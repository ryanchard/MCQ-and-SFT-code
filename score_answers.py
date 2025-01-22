#!/usr/bin/env python3

import sys
import json
import os
import statistics
import requests
import openai
from openai import OpenAI
import time
import glob

# ---------------------------------------------------------------------
# Configuration constants for TWO different models/endpoints/API keys
# ---------------------------------------------------------------------
from inference_auth_token import get_access_token
alcf_access_token = get_access_token()

from alcf_inference_utilities import get_names_of_alcf_chat_models
alcf_chat_models = get_names_of_alcf_chat_models(alcf_access_token)

OPENAI_EP  = 'https://api.openai.com/v1'

with open('openai_access_token.txt', 'r') as file:
    openai_access_token = file.read().strip()


# ---------------------------------------------------------------------

def evaluate_answer(model, question: str, reference_answer: str, user_answer: str) -> float:
    """
    Calls the model to evaluate how consistent the user_answer is with the reference_answer.
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

    response = client.chat.completions.create(
        model=modelname,
        messages=[
            {"role": "system", "content": "You are a strict grader. Respond with only the number."},
            {"role": "user", "content": eval_prompt},
        ],
        temperature=0.0,
    )

    # Extract the numeric score from the assistant's response
    # raw_score = response["choices"][0]["message"]["content"].strip()
    raw_score = response.choices[0].message.content.strip()
    try:
        score = float(raw_score)
    except ValueError:
        # If the model didn't return a pure number, attempt a fallback or default to 0
        score = 0.0
    
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
        print('Valid models are:', alcf_chat_models)
        exit(1)
    return (model, key, endpoint)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Program to use LLM B to rate answers provided previously by LLM A') 
    parser.add_argument('-a','--modelA', help='modelA', required=True)
    parser.add_argument('-b','--modelB', help='modelB', required=True)
    parser.add_argument('-o','--output', help='Output directory', required=True)
    parser.add_argument('-z','--zero',   help='Process empty files', action="store_true")
    args = parser.parse_args()

    output_dir   = args.output

    modelA_name = args.modelA
    modelB_name = args.modelB
    modelB      = get_model_parameters(modelB_name)

    # Load previously generated answers from modelA
    answer_file = output_dir+'/answers_'+modelA_name.replace('/', '+')+'.json'
    print(f'Looking for {answer_file}')
    if not os.path.exists(answer_file):
        print(f'No answers file for {modelA_name}')
        exit(1)

    score_file = f'{output_dir}/scores_{modelA_name.replace("/","+")}:{modelB_name.replace("/","+")}.json'
    print(f'Looking for {score_file}')
    if os.path.exists(score_file) and not args.zero:
        print('Already exists:', score_file)
        exit(1)

    out_f = open(score_file, 'w', encoding='utf-8') 

    # Load question-answer pairs
    with open(answer_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    scores   = []
    qa_pairs = []

    start_time = time.time()
    total_time = 0
    eval_answer_total_time = 0

    print(f'Processing {len(data)} QA pairs')
    for (qa_pair, index) in zip(data, range(1, len(data) + 1)):
        question         = qa_pair.get("question", "")
        reference_answer = qa_pair.get("reference", "")
        model_answer     = qa_pair.get("model", "")
        gen_time         = qa_pair.get("gen_time", "")
        file             = qa_pair.get("file", "")
        filenum          = qa_pair.get("filenum", "")
        chunknum         = qa_pair.get("chunknum", "")

        if not question or not reference_answer or not model_answer:
            print('Bad item:')
            print('QQQ', question)
            print('RRR', reference_answer)
            print('MMM', model_answer)
            exit(1)
            continue  # skip malformed items

        # Use model to evaluate/grade the generated answer in file
        # against the reference answer
        eval_answer_start_time = time.time()
        score = evaluate_answer(modelB, question, reference_answer, model_answer)
        eval_answer_time = time.time() - eval_answer_start_time
        eval_answer_total_time += eval_answer_time
        scores.append(score)
        qa_pairs.append({'modelA': modelA_name, 'modelB': modelB[0], 'index': index, 'question': question, 'reference':reference_answer, 'model':model_answer, 'score':score, 'gen_time': gen_time, 'eval_time': f'{eval_answer_time:.4f}', 'file':file, 'filenum':filenum, 'chunknum':chunknum})

        total_time += time.time() - start_time
        start_time = time.time()
        if index%10==0:
            avg_time = total_time / index  # Average time per item so far
            avg_eval_time = eval_answer_total_time / index  # Average time per item so far
            print(f'{index} ({avg_time:.2f})', end =' ', flush=True) 

    print()

    json.dump(qa_pairs, out_f, ensure_ascii=False, indent=2)

    if scores:
        mean_score = statistics.mean(scores)
        variance_score = statistics.pvariance(scores)  # population variance
    else:
        print("No valid QA pairs found or no scores computed.")


if __name__ == "__main__":
    main()
