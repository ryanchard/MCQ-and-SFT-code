#!/usr/bin/env python3

import sys
import json
import os
#import statistics
import requests
import openai
from openai import OpenAI
#import time

# import torch


# ---------------------------------------------------------------------
# Configuration constants for model/endpoint/API keys
# ---------------------------------------------------------------------
OPENAI_EP  = 'https://api.openai.com/v1'


class Model:
    def __init__(self, model_name):
        self.base_model = None
        self.tokenizer  = None
        self.model_name = model_name
        self.temperature = 0.7
        self.headers = { 'Content-Type': 'application/json' }

        # Model to be run locally via VLLM
        if model_name.startswith('local:'):
            self.model_name = model_name.split('local:')[1]
            print('Local model:', model_name)
            self.key        = None
            self.endpoint   = 'http://localhost:8000/v1/chat/completions'
            self.model_type = 'vLLM'
    
        # Model to be downloaded from HF and run locally
        elif model_name.startswith('hf:'):
            self.model_type = 'Huggingface'
            from huggingface_hub import login
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer
            )

            self.model_name = model_name.split('hf:')[1]
            print('HF model:', model_name)
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
            print('ALCF Inference Service Model:', self.model_name)

            from inference_auth_token import get_access_token
            self.key = get_access_token()

            from alcf_inference_utilities import get_names_of_alcf_chat_models
            alcf_chat_models = get_names_of_alcf_chat_models(self.key)
            if self.model_name not in alcf_chat_models:
                print('Bad ALCF model', self.model_name)
                exit(1)
            self.model_type = 'ALCF'

            self.endpoint = 'https://data-portal-dev.cels.anl.gov/resource_server/sophia/vllm/v1'
    
        elif model_name.startswith('openai'):
            self.model_name = model_name.split('openai:')[1]
            print('OpenAI model:', self.model_name)
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
    

    def details(self):
        print(f'Model {self.model_name}:')
        print(f'    Model type  = {self.model_type}')
        # print(f'    Key         = {self.key}')
        print(f'    Endpoint    = {self.endpoint}')
        print(f'    Temperature = {self.temperature}')
        print(f'    Base model  = {self.base_model}')
        print(f'    Tokenizer   = {self.tokenizer}')


    def run(self, question: str, system_prompt='You are a helpful assistant', temperature=0.7) -> str:
        """
        Calls model (configured with MODEL_NAME, ENDPOINT_MODEL, API_KEY_MODEL)
        to generate an answer to the given question.
        """
        
        # Debug info
        #self.details()
    
        if self.model_type == 'Huggingface':
            #print('Calling HF model', hf_info)
            response = run_hf_model(question, self.base_model, self.tokenizer)
            #print('HF response =', response)
            return response
    
        elif self.model_type == 'vLLM':
            #print('HERE')
            data = {
                "model": self.model_name,
                "messages": [
                             {"role": "system", "content": system_prompt},
                             {"role": "user", "content": question}
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
                    {"role": "user", "content": question},
                ]
                temperature = 1.0
            else:
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
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
                print(f"OpenAI API request timed out: {e}, with {modelname}, index {index}, 60 sec timeout, and prompt={eval_prompt}")
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
    output = base_model.generate(input_ids, attention_mask=attention_mask, max_length=200, pad_token_id=tokenizer.pad_token_id)

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text


