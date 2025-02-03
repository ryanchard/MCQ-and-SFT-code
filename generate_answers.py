#!/usr/bin/env python3

import sys
import json
import os
import statistics
import requests
import time

import argparse
#import torch

from model_access import Model


# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Program to use LLM to provide answers to MCQs')
    parser.add_argument('-i','--input', help='MCQs input file', required=True)
    parser.add_argument('-m','--model', help='model', required=True)
    parser.add_argument('-o','--output', help='Output directory', required=True)
    parser.add_argument('-s','--start', help='Number to start at in MCQs file', default='0')
    parser.add_argument('-e','--end', help='End number in MCQs file', default='all')
    args = parser.parse_args()

    json_file = args.input
    output_dir   = args.output

    model_name = args.model

    model = Model(model_name)
    model.details()

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
        model_answer = model.run(question)

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
