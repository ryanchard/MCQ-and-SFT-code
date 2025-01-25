#!/usr/bin/env python3

"""
Usage:
    python script.py questions.json

Where questions.json is a JSON file with an array of objects, each having
"question" and "answer" fields, for example:

[
    {
        "question": "What is the capital of France?",
        "answer": "Paris"
    },
    {
        "question": "Who wrote 'Pride and Prejudice'?",
        "answer": "Jane Austen"
    }
]
"""

import sys
import json
import os
import statistics
import requests
import openai
from openai import OpenAI
import time

from alcf_inference_utilities import get_names_of_alcf_chat_models

from inference_auth_token import get_access_token
alcf_access_token = get_access_token()


# ---------------------------------------------------------------------
# Configuration constants for model/endpoint/API keys
# ---------------------------------------------------------------------
OPENAI_EP  = 'https://api.openai.com/v1'

with open('openai_access_token.txt', 'r') as file:
    openai_access_token = file.read().strip()

alcf_chat_models = get_names_of_alcf_chat_models(alcf_access_token)


# ---------------------------------------------------------------------

def generate_answer(model, question: str) -> str:
    """
    Calls the first model (configured with MODEL_NAME_1, ENDPOINT_MODEL_1, API_KEY_MODEL_1)
    to generate an answer to the given question.
    """
    (modelname, key, ep) = model
    # Configure the OpenAI client for model 1
    client = OpenAI(
        api_key  = key,
        base_url = ep
    )

    # For chat models:
    #print(f'CALL {modelname}')
    response = client.chat.completions.create(
        model=modelname,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question},
        ],
        temperature=0.7,
    )
    #print(f'RETURN {modelname}')

    # Extract the assistant's response
    generated_text = response.choices[0].message.content.strip()
    #print(f'Generated answer for {question}:\n\t{generated_text}\n')
    return generated_text


def get_model_parameters(model):
    # if model == 'mistralai/Mistral-7B-Instruct-v0.3':
    if model in alcf_chat_models:
        key      = alcf_access_token
        endpoint = 'https://data-portal-dev.cels.anl.gov/resource_server/sophia/vllm/v1'
    elif model == 'gpt-4o':
        key      = openai_access_token
        endpoint = OPENAI_EP
    else:
        print('Bad model:', model)
        exit(1)
    return (model, key, endpoint)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Program to use LLM B to rate answers provided by LLM A') 
    parser.add_argument('-i','--input', help='QA input file', required=True)
    parser.add_argument('-m','--model', help='model', required=True)
    parser.add_argument('-o','--output', help='Output directory', required=True)
    parser.add_argument('-s','--start', help='Number to start at in QA file', default='0')
    parser.add_argument('-e','--end', help='End number in QA file', default='all')
    args = parser.parse_args()

    json_file = args.input
    output_dir   = args.output

    model_name = args.model
    model = get_model_parameters(model_name)

    # Load question-answer pairs
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    qa_pairs = []

    start_time = time.time()
    total_time = 0

    if args.end == 'all':
        data = data[int(args.start):]
    else:
        data = data[int(args.start):int(args.end)]

    print(f'Generating {len(data)} answers with model {model_name}')
    for (qa_pair, index) in zip(data, range(1, len(data) + 1)):
        question = qa_pair.get("question", "")
        reference_answer = qa_pair.get("answer", "")
        filename         = qa_pair.get("file", "")
        filenum          = qa_pair.get("filenum", "")
        chunknum         = qa_pair.get("chunknum", "")

        if not question or not reference_answer:
            continue  # skip malformed items

        # Use the model to generate an answer
        model_answer = generate_answer(model, question)

        gen_time    = time.time() - start_time
        total_time += gen_time
        start_time  = time.time()
        if index%10==0:
            avg_time = total_time / index  # Average time per item so far
            print(f'{index} ({avg_time:.2f} s)', end =' ', flush=True) 

        new_tuple = {'file':filename, 'filenum':filenum, 'chunknum':chunknum, 'gen_time': f'{gen_time:.3f}',
                     'question':question, 'reference': reference_answer, 'model': model_answer}
        qa_pairs.append(new_tuple)

    print()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if int(args.start)==0 and args.end=='all':
        output_file = f'answers_{model_name.replace("/","+")}.json'
    else:
        output_file = f'answers_{model_name.replace("/","+")}_{args.start}_{args.end}.json'
    with open(output_dir+'/'+output_file, 'w', encoding='utf-8') as out_f:
        json.dump(qa_pairs, out_f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
