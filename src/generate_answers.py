#!/usr/bin/env python

import sys
import json
import os
import statistics
import requests
import time
import argparse

import config
from model_access import Model
from tqdm import tqdm

"""
import hydra
from omegaconf import DictConfig

@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(cfg.general.noise_threshold)  # Accessing config values

if __name__ == "__main__":
    main()
"""

# other functions

# add a "no op" progress bar for quiet mode
class NoOpTqdm:
    """A do-nothing progress bar class that safely ignores all tqdm calls."""
    def __init__(self, total=0, desc="", unit=""):
        self.total = total  # Store total count
        self.n = 0  # Keep track of progress count

    def update(self, n=1):
        self.n += n  # Simulate tqdm's progress tracking

    def set_postfix_str(self, s):
        pass  # No-op

    def close(self):
        pass  # No-op

# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Program to use LLM to provide answers to MCQs')
    parser.add_argument('-m','--model', help='model', required=True)
    parser.add_argument('-s','--start', help='Number to start at in MCQs file', default='0')
    parser.add_argument('-e','--end', help='End number in MCQs file', default='all')
    parser.add_argument('-c', "--cache-dir", type=str, default=os.getenv("HF_HOME"),
                        help="Custom cache directory for Hugging Face")
    parser.add_argument('-i', '--input',  help='file containing MCQs', required=True)  
    parser.add_argument('-o', '--output', help='Output directory for Results',
                        default=config.results_dir)
    parser.add_argument('-q','--quiet',   action='store_true',   
                        help='No progress bar or messages')
    parser.add_argument('-v','--verbose', action='store_true',    
                        help='Enable verbose logging')    

    args = parser.parse_args()

    # Decide logging level and whether to show a progress bar
    if args.verbose:
        config.logger.setLevel(logging.INFO)
        use_progress_bar = False
    elif args.quiet:
        config.logger.setLevel(logging.CRITICAL)
        use_progress_bar = False
    else:  # default case
        config.logger.setLevel(logging.WARNING)
        use_progress_bar = True

    # Set HF_HOME if using custom cache directory
    if args.cache_dir:
        os.environ["HF_HOME"] = args.cache_dir
        print(f"Using Hugging Face cache directory: {args.cache_dir}")

    json_file = args.input
    output_dir   = args.output

    model_name = args.model

    model = Model(model_name)
    model.details()

    # Load question-answer pairs
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except:
        print(f"ERROR: file {json_file} not found.")
        sys.exit(0)

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
        try:
            model_answer = model.run(question)
        except KeyboardInterrupt:
            print("EXIT: Execution interrupted by user")
            sys.exit(0)
        except Exception as e:
            print(f"ERROR: {e}")
            sys.exit(0)

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

