#!/usr/bin/env python3

import sys
import json
import os
import subprocess
import re
import time
import socket
import requests
import openai
from openai import OpenAI


# ---------------------------------------------------------------------
# Configuration constants for model/endpoint/API keys
# ---------------------------------------------------------------------

# --- CeC --- #
# centralize to a single authoritative alcf_chat_models list

from inference_auth_token import get_access_token
from alcf_inference_utilities import get_names_of_alcf_chat_models

# Initialize the shared globals for ALCF models.
ALCF_ACCESS_TOKEN = get_access_token()
ALCF_CHAT_MODELS = get_names_of_alcf_chat_models(ALCF_ACCESS_TOKEN)
#print("DEBUG (model_access.py): ALCF_CHAT_MODELS is", ALCF_CHAT_MODELS, flush=True)
# --- CeC --- #

OPENAI_EP  = 'https://api.openai.com/v1'


class Model:
    def __init__(self, model_name):
        self.base_model = None
        self.tokenizer  = None
        self.model_name = model_name
        self.temperature = 0.7
        self.headers = { 'Content-Type': 'application/json' }
        self.endpoint = None

        # Model to be run locally via VLLM
        if model_name.startswith('local:'):
            self.model_name = model_name.split('local:')[1]
            print('\nLocal model:', model_name)
            self.key        = None
            self.endpoint   = 'http://localhost:8000/v1/chat/completions'
            self.model_type = 'vLLM'
    
        elif model_name.startswith('pb:'):
            """Submit the model job to PBS and store the job ID"""
            self.model_name = model_name.split('pb:')[1]
            self.model_type = 'HuggingfacePBS'
            self.model_script="run_model.pbs"
            self.job_id = None
            self.status = "PENDING"
            self.client_socket = None

            # Submit the PBS job and capture the job ID
            print(f'\nHuggingface model {self.model_name} to be run on HPC system: Starting model server.')
            result = subprocess.run(["qsub", self.model_script], capture_output=True, text=True, check=True)
            self.job_id = result.stdout.strip().split(".")[0]  # Extract job ID

            print(f"Job submitted with ID: {self.job_id}")
            # Wait until job starts running
            self.wait_for_job_to_start()
            self.connect_to_model_server()

            # Send model name to server
            #print(f"Sending model name: {self.model_name}")
            self.client_socket.sendall(self.model_name.encode())

            # Receive response
            response = self.client_socket.recv(1024).decode()
            if response != 'ok':
                print('Unexpected response:', response)
                exit(1)
            print(f"Model server initialized")

        # Model to be downloaded from HF and run via PBS job
        elif model_name.startswith('hf:'):
            self.model_type = 'Huggingface'

            from huggingface_hub import login
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer
            )

            self.model_name = model_name.split('hf:')[1]
            print('\nHF model running locally:', model_name)
            self.endpoint = 'http://huggingface.co'

            with open("hf_access_token.txt", "r") as file:
                self.key = file.read().strip()
            login(self.key)
    
            max_seq_length = 2048  # e.g. for models that support RoPE scaling
    
            # Define the model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    
            # Load base model with quantization
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
            )
    
            # Ensure pad token is set correctly
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
    
            # Ensure model has correct pad token ID
            self.base_model.config.pad_token_id = self.tokenizer.pad_token_id
        
            # Ensure tokenizer uses correct padding side
            self.tokenizer.padding_side = "right"  # Recommended for LLaMA
    
        elif model_name.startswith('alcf'):
            self.model_name = model_name.split('alcf:')[1]
            print('\nALCF Inference Service Model:', self.model_name)

            #from inference_auth_token import get_access_token
            #self.key = get_access_token()
#
            #from alcf_inference_utilities import get_names_of_alcf_chat_models
            #alcf_chat_models = get_names_of_alcf_chat_models(self.key)
            #if self.model_name not in alcf_chat_models:
                #print('Bad ALCF model', self.model_name)
                #exit(1)
            #self.model_type = 'ALCF'
#
            #self.endpoint = 'https://data-portal-dev.cels.anl.gov/resource_server/sophia/vllm/v1'
            self.key = ALCF_ACCESS_TOKEN

            if self.model_name not in ALCF_CHAT_MODELS:
                print('Bad ALCF model', self.model_name)
                exit(1)
            self.model_type = 'ALCF'
            self.endpoint = 'https://data-portal-dev.cels.anl.gov/resource_server/sophia/vllm/v1'
    
        elif model_name.startswith('openai'):
            self.model_name = model_name.split('openai:')[1]
            print('\nOpenAI model to be run at OpenAI:', self.model_name)
            self.model_type = 'OpenAI'
            #if self.model_name not in ['gpt-4o']:
            #    print('Bad OpenAI model', self.model_name)
            #    exit(1)
            with open('openai_access_token.txt', 'r') as file:
                self.key = file.read().strip()
            self.endpoint = OPENAI_EP
    
        else:
            print('Bad model:', model_name)
            exit(1)
    
    def wait_for_job_to_start(self):
        """Monitor job status and get assigned compute node"""
        while True:
            qstat_output = subprocess.run(["qstat", "-f", self.job_id], capture_output=True, text=True).stdout

            # Extract compute node name
            match = re.search(r"exec_host = (\S+)", qstat_output)
            if match:
                self.compute_node = match.group(1).split("/")[0]  # Get the node name
                print(f"Job {self.job_id} is running on {self.compute_node}")
                self.status = "RUNNING"
                break

            print(f"Waiting for job {self.job_id} to start...")
            time.sleep(5)  # Check every 5 seconds

    def connect_to_model_server(self):
        """Establish a persistent TCP connection to the model server"""
        if self.status != "RUNNING":
            raise RuntimeError("Model is not running. Ensure the PBS job is active.")

        print(f"Connecting to {self.compute_node} on port 50007...")
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        for count in range(10):
            try:
                self.client_socket.connect((self.compute_node, 50007))  # Connect to compute node
                break
            except:
                time.sleep(5)
                print(f"Trying connection again {count}")


    def details(self):
        print(f'Model {self.model_name}:')
        print(f'    Model type  = {self.model_type}')
        # print(f'    Key         = {self.key}')
        print(f'    Endpoint    = {self.endpoint}')
        print(f'    Temperature = {self.temperature}')
        print(f'    Base model  = {self.base_model}')
        print(f'    Tokenizer   = {self.tokenizer}')

    def close(self):
        """Close the cached connection when done"""
        if self.client_socket:
            print("Closing connection to model server.")
            self.client_socket.close()
            self.client_socket = None

    def run(self, user_prompt='Tell me something interesting', system_prompt='You are a helpful assistant', temperature=0.7) -> str:
        """
        Calls model (configured with MODEL_NAME, ENDPOINT_MODEL, API_KEY_MODEL)
        to generate an answer to the given question.
        """
        
        # Debug info
        #self.details()
    
        if self.model_type == 'Huggingface':
            #print('Calling HF model', hf_info)
            response = run_hf_model(user_prompt, self.base_model, self.tokenizer)
            #print('HF response =', response)
            return response

        elif self.model_type == 'HuggingfacePBS':
            """Send a request to the running PBS job via cached connection"""
            if self.status != "RUNNING":
                raise RuntimeError("Model is not running. Ensure the PBS job is active.")
            if self.client_socket is None:
                raise RuntimeError("Socket is not connected")
    
            #print(f"Sending input to model: {user_prompt}")
            self.client_socket.sendall(user_prompt.encode())

            # Receive response
            response = self.client_socket.recv(1024).decode()
            #print(f"Received response from model: {response}")
            return response
    
        elif self.model_type == 'vLLM':
            #print('HERE')
            data = {
                "model": self.model_name,
                "messages": [
                             {"role": "system", "content": system_prompt},
                             {"role": "user", "content": user_prompt}
                            ],
                'temperature': self.temperature
            }
            try:
                #print(f'Running {self.endpoint}\n\tHeaders = {self.headers}\n\tData = {json.dumps(data)}')
                response = requests.post(self.endpoint, headers=self.headers, data=json.dumps(data))
                #print('Response:', response)
                response = response.json()
                #print('JSON:', response)
                message = response['choices'][0]['message']['content']
                #print('MESSAGE:', message)
                #exit(1)
            except Exception as e:
                print(f'Exception: {e}')
                #exit(1)
                message = ''
            return message
    
        elif self.model_type == 'OpenAI' or self.model_type=='ALCF':
            # Configure the OpenAI client for model 1
            client = OpenAI(
                api_key  = self.key,
                base_url = self.endpoint
            )
    
            # For chat models:
            if self.model_type == 'OpenAI' and self.model_name[:2] == 'o1':
                messages=[
                    {"role": "user", "content": user_prompt},
                ]
                temperature = 1.0
            else:
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                temperature=self.temperature

            try:
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    timeout = 60
                )
            except APITimeoutError as e:
                # This will catch timeouts from the request_timeout parameter
                print(f"OpenAI API request timed out: {e}, with {self.model_name}, index {index}, 60 sec timeout, and prompt={eval_prompt}")
                return None
            except Exception as e:
                # Optionally catch other errors
                print(f"Some other error occurred: {e}")
                return None
    
            # Extract the assistant's response
            generated_text = response.choices[0].message.content.strip()
            return generated_text

        else:
            print('Unknown model type:', self.model_type)
            exit(1)
    

def initialize_hf_model(model_name):
    with open("hf_access_token.txt", "r") as file:
        hf_access_token = file.read().strip()
    login(hf_access_token)

    max_seq_length = 2048  # e.g. for models that support RoPE scaling

    # Define the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load base model with quantization
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
    )

    # Ensure pad token is set correctly
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Ensure model has correct pad token ID
    base_model.config.pad_token_id = tokenizer.pad_token_id

    # Ensure tokenizer uses correct padding side
    tokenizer.padding_side = "right"  # Recommended for LLaMA

    return(base_model, tokenizer)


def run_hf_model(input_text, base_model, tokenizer):
    # Prepare input for generation
    inputs = tokenizer(input_text, return_tensors="pt", padding=True)

    # Explicitly move tensors to GPU
    input_ids = inputs["input_ids"].to("cuda")
    attention_mask = inputs["attention_mask"].to("cuda")  # Pass attention mask

    # Explicitly pass attention_mask
    output = base_model.generate(input_ids, attention_mask=attention_mask, max_length=512, pad_token_id=tokenizer.pad_token_id)

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text


