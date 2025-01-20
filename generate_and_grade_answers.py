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

# ---------------------------------------------------------------------
# Configuration constants for TWO different models/endpoints/API keys
# ---------------------------------------------------------------------
OPENAI_EP  = 'https://api.openai.com/v1'

with open('alcf_access_token.txt', 'r') as file:
    alcf_access_token = file.read().strip()

with open('openai_access_token.txt', 'r') as file:
    openai_access_token = file.read().strip()

# Define the URL and headers for ALCF Inference Service list-endpoints
url = "https://data-portal-dev.cels.anl.gov/resource_server/list-endpoints"
headers = {
    "Authorization": f"Bearer {alcf_access_token}"
}

# Make the GET request
response = requests.get(url, headers=headers)

# Check the response
if response.status_code == 200:
    status = response.json()
else:
    print("Error:", response.status_code, response.text)
    exit(1)

alcf_chat_models = status['clusters']['sophia']['frameworks']['vllm']['models']


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


def evaluate_answer(model, question: str, reference_answer: str, user_answer: str) -> float:
    """
    Calls the second model (configured with MODEL_NAME_2, ENDPOINT_MODEL_2, API_KEY_MODEL_2)
    to evaluate how consistent the user_answer is with the reference_answer.
    Returns a numeric score (float) from 1 to 10.
    """
    # Configure the OpenAI client for model 2
    (modelname, key, ep) = model
    client = OpenAI(
        api_key  = key,
        base_url = ep
    )

    # We ask the model to strictly return a single number from 1 to 10
    # indicating how well the user_answer matches the reference_answer.
    eval_prompt = f"""
You are a strict grader. 

Question: {question}
Reference Answer: {reference_answer}
User's Answer: {user_answer}

On a scale of 1 to 10 (10 = exactly matches the reference answer, 
1 = completely incorrect), provide ONLY the numeric score that reflects
how well the User's Answer matches the Reference Answer. No extra text.
"""

    # print(f'CALL {modelname}')
    response = client.chat.completions.create(
        model=modelname,
        messages=[
            {"role": "system", "content": "You are a strict grader. Respond with only the number."},
            {"role": "user", "content": eval_prompt},
        ],
        temperature=0.0,
    )
    # print(f'RETURN {modelname}')

    # Extract the numeric score from the assistant's response
    # raw_score = response["choices"][0]["message"]["content"].strip()
    raw_score = response.choices[0].message.content.strip()
    try:
        score = float(raw_score)
    except ValueError:
        # If the model didn't return a pure number, attempt a fallback or default to 0
        score = 0.0
    
    #print(f'Evaluated answer: {score}')
    return score


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
    parser.add_argument('-a','--modelA', help='modelA', required=True)
    parser.add_argument('-b','--modelB', help='modelB', required=True)
    parser.add_argument('-q','--quiet', help='Do not show details', action='store_true', default=False)
    parser.add_argument('-c','--csv', help='Generate CSV summary', action='store_true', default=False)
    parser.add_argument('-s','--start', help='Number to start at in QA file', default='0')
    parser.add_argument('-e','--end', help='End number in QA file', default='all')
    parser.add_argument('-o','--output', help='Output directory', default='output_files')
    args = parser.parse_args()

    json_file = args.input
    show_details = not args.quiet
    generate_csv = args.csv
    output_dir   = args.output

    modelA = get_model_parameters(args.modelA)
    modelB = get_model_parameters(args.modelB)

    # Load question-answer pairs
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    scores   = []
    qa_pairs = []

    start_time = time.time()
    total_time = 0
    gen_answer_total_time = 0
    eval_answer_total_time = 0

    if args.end == 'all':
        data = data[int(args.start):]
    else:
        data = data[int(args.start):int(args.end)]

    # print(f'Processing {len(data)} QA pairs')
    for (qa_pair, index) in zip(data, range(1, len(data) + 1)):
        question = qa_pair.get("question", "")
        reference_answer = qa_pair.get("answer", "")

        if not question or not reference_answer:
            continue  # skip malformed items

        # Step 1: Use the first model to generate an answer
        gen_answer_start_time = time.time()
        model_answer = generate_answer(modelA, question)
        gen_answer_time = time.time() - gen_answer_start_time
        gen_answer_total_time += gen_answer_time

        # Step 2: Use the second model to evaluate/grade the generated answer 
        # against the reference answer
        eval_answer_start_time = time.time()
        score = evaluate_answer(modelB, question, reference_answer, model_answer)
        eval_answer_time = time.time() - eval_answer_start_time
        eval_answer_total_time += eval_answer_time
        scores.append(score)
        qa_pairs.append({'modelA': modelA[0], 'modelB': modelB[0], 'index': index, 'question': question, 'ref':reference_answer, 'model':model_answer, 'score':score, 'gen_time': f'{gen_answer_time:.4f}', 'eval_time': f'{eval_answer_time:.4f}'})
        if show_details:
            print(f"Question {index}: {question}\n")
            print(f"Reference answer: {reference_answer}\n")
            print(f"Model A answer: {model_answer}\n")
            print(f"Model B Evaluation: {score}")
            print("--------------------------------------------------")

        total_time += time.time() - start_time
        start_time = time.time()
        if index%10==0:
            avg_time = total_time / index  # Average time per item so far
            avg_gen_time = gen_answer_total_time / index  # Average time per item so far
            avg_eval_time = eval_answer_total_time / index  # Average time per item so far
            print(f'{index} ({avg_time:.2f}; {avg_gen_time:.2f}; {avg_eval_time:.2f})', end =' ', flush=True) 

    print()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = f'out_{args.modelA.replace("/","+")}:{args.modelB}_{args.start}_{args.end}.json'
    with open(output_dir+'/'+output_file, 'w', encoding='utf-8') as out_f:
        json.dump(qa_pairs, out_f, ensure_ascii=False, indent=2)

    if scores:
        mean_score = statistics.mean(scores)
        variance_score = statistics.pvariance(scores)  # population variance
        if generate_csv:
            print(f'{len(data)},{args.start},{args.end},{modelA[0]},{modelB[0]},{mean_score:.2f},{variance_score:.2f}')
        else:
            print(f'Evaluated {len(data)} responses from {modelA[0]} with {modelB[0]}:')
            print(f"  Mean score: {mean_score:.2f}")
            print(f"  Variance: {variance_score:.2f}")
    else:
        print("No valid QA pairs found or no scores computed.")


if __name__ == "__main__":
    main()
