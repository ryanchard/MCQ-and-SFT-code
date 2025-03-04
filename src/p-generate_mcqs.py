#!/usr/bin/env python

# parallel version of generate_mcqs.py

import os
import sys
import json
import re
import time
import spacy
import argparse
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import config
from model_access import Model

##############################################################################
# Global constants and counters
##############################################################################
CHUNK_SIZE = config.chunkSize
chunks_successful = 0
chunks_failed     = 0

nlp = spacy.load("en_core_web_sm")

# NoOpTqdm for quiet mode
class NoOpTqdm:
    """A do-nothing progress bar that ignores tqdm calls."""
    def __init__(self, total=0, desc="", unit=""):
        self.total = total
        self.n = 0
    def update(self, n=1):
        self.n += n
    def set_postfix_str(self, s):
        pass
    def close(self):
        pass

def human_readable_time(seconds: float) -> str:
    """Convert time in seconds into a more human-friendly format."""
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f} minutes"
    elif seconds < 86400:
        hours = seconds / 3600
        return f"{hours:.2f} hours"
    else:
        days = seconds / 86400
        return f"{days:.2f} days"

def split_text_into_chunks(text: str, chunk_size: int = CHUNK_SIZE) -> list:
    """
    Split text into chunks of ~chunk_size words, respecting sentence
    boundaries using spaCy sentence segmentation.
    """
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        word_count = len(sentence.split())
        if current_length + word_count > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = word_count
        else:
            current_chunk.append(sentence)
            current_length += word_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def generate_mcqs(model, path, filename, linenum, chunks, chunk_progress_callback=None) -> list:
    """
    Processes each chunk:
      1) Summarize/expand
      2) Generate multiple choice question
      3) Verify (score) question
      Returns list of QA pairs (JSON records).
    """
    global chunks_successful, chunks_failed
    qa_pairs = []

    for chunknum, chunk in enumerate(chunks, start=1):
        # Step 1: Summarize & expand
        try:
            formatted_user_message = config.user_message.format(chunk=chunk)
            step1_output = model.run(
                user_prompt=formatted_user_message,
                system_prompt=config.system_message
            )
            augmented_chunk = step1_output
            # (Optional check if it has "augmented_chunk: ...")
            if "augmented_chunk:" in str(step1_output).lower():
                # etc.
                pass
        except Exception as e:
            config.logger.warning(f"Error summarizing chunk: {e}")
            if "401" in str(e) or "Unauthorized" in str(e):
                sys.exit(f"Model API Authentication failed. ({str(e)}) Exiting.")
            chunks_failed += 1
            if chunk_progress_callback:
                chunk_progress_callback()
            continue

        # Step 2: Generate question
        try:
            user_msg_2 = config.user_message_2.format(augmented_chunk=augmented_chunk)
            generated_question = model.run(
                user_prompt=user_msg_2,
                system_prompt=config.system_message_2
            )
        except Exception as e:
            config.logger.warning(f"Error generating question: {e}")
            if "401" in str(e) or "Unauthorized" in str(e):
                sys.exit("Model API Authentication failed. Exiting.")
            chunks_failed += 1
            if chunk_progress_callback:
                chunk_progress_callback()
            continue

        # Step 3: Verify question
        try:
            user_msg_3 = config.user_message_3.format(
                augmented_chunk=augmented_chunk,
                generated_question=generated_question
            )
            step3_output = model.run(
                user_prompt=user_msg_3,
                system_prompt=config.system_message_3
            )
            if not step3_output:
                raise ValueError("No output returned from model.run() in step3.")

            # Clean/parse JSON
            step3_output = step3_output.replace("```json", "").replace("```", "")
            # etc. parse logic
            parsed_json = json.loads(step3_output)

            model_answer = str(parsed_json.get("answer", "")).strip()
            model_score  = parsed_json.get("score", 0)

            if isinstance(model_score, int) and model_score > config.minScore:
                chunks_successful += 1
                qa_pairs.append({
                    "file": filename,
                    "path": path,
                    "line": linenum,
                    "chunk": chunknum,
                    "model": model.model_name,
                    "question": generated_question,
                    "answer": model_answer,
                    "text": augmented_chunk
                })
            else:
                chunks_failed += 1

        except Exception as e:
            config.logger.info(f"Chunk fail: Error verifying question/answer: {e}")
            chunks_failed += 1

        finally:
            # Let the main thread update progress, if we gave a callback
            if chunk_progress_callback:
                chunk_progress_callback()

    return qa_pairs

def process_one_file(model, filename, input_dir, output_dir):
    """
    Processes a single file (JSON or JSONL) and returns:
      - number_of_chunks processed (for progress bar)
      - the file index or name
      - time spent, etc.
    """
    file_path = os.path.join(input_dir, filename)
    start_time = time.time()
    all_prompt_answer_pairs = []
    num_chunks = 0

    # Read file lines
    with open(file_path, 'r', encoding='utf-8') as f:
        if filename.lower().endswith(".json"):
            json_str = f.read()
            lines = [json_str]
        else:
            lines = f.readlines()

    # For each line
    for j, line in enumerate(lines, start=1):
        try:
            record = json.loads(line.strip())
        except json.JSONDecodeError as e:
            config.logger.info(f"JSON decode error in {filename} line {j}: {e}")
            continue

        text = record['text']
        path = record['path']

        # Split into chunks and process
        chunks = split_text_into_chunks(text, CHUNK_SIZE)
        num_chunks += len(chunks)

        # We call generate_mcqs. Note that we do NOT do the tqdm update here;
        # we’ll do it in the main thread. We’ll simply gather results.
        qa_pairs = generate_mcqs(
            model, path, filename, j, chunks,
            chunk_progress_callback=None  # no direct tqdm call here
        )
        all_prompt_answer_pairs.extend(qa_pairs)

    # Write out results
    out_file = f'{output_dir}/{filename}.out.json'
    with open(out_file, 'w', encoding='utf-8') as out_f:
        json.dump(all_prompt_answer_pairs, out_f, ensure_ascii=False, indent=2)

    elapsed = time.time() - start_time
    return num_chunks, elapsed, filename

def process_directory(model, input_dir, output_dir,
                      use_progress_bar=True,
                      num_workers=1):
    """
    Parallel version: 
    - gather all files
    - create a global tqdm progress bar
    - use ThreadPoolExecutor to process each file in parallel
    """
    # Gather JSON/JSONL files
    json_files  = [f for f in os.listdir(input_dir) if f.lower().endswith(".json")]
    jsonl_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".jsonl")]
    all_files = json_files + jsonl_files
    total_files = len(all_files)

    if total_files == 0:
        config.logger.warning("No suitable files found in directory.")
        return

    # Approximate total chunk count for the entire directory
    approximate_chunk_count = max(
        1,
        sum(os.stat(os.path.join(input_dir, f)).st_size
            for f in all_files) // CHUNK_SIZE
    )

    # Setup tqdm
    if use_progress_bar:
        pbar = tqdm(total=approximate_chunk_count, desc="Chunks processed", unit="chunk")
    else:
        pbar = NoOpTqdm(total=approximate_chunk_count)

    overall_start_time = time.time()

    # We'll use a ThreadPoolExecutor with `max_workers=num_workers`.
    # Each future returns (num_chunks, elapsed_time, filename).
    futures = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for filename in all_files:
            f = executor.submit(process_one_file, model, filename, input_dir, output_dir)
            futures.append(f)

        # As each file completes, we know how many chunks it processed.
        for future in as_completed(futures):
            # Grab the results from that file
            num_chunks, elapsed, filename = future.result()
            # Update the global tqdm
            pbar.update(num_chunks)
            config.logger.info(f"Finished {filename} in {human_readable_time(elapsed)} with {num_chunks} chunks.")

    # Done with all files
    total_time = time.time() - overall_start_time
    pbar.close()

    config.logger.info(
        f"Finished parallel processing {len(all_files)} files in {human_readable_time(total_time)}."
    )
    config.logger.info(
        f"{chunks_successful} chunks succeeded, {chunks_failed} failed."
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate MCQs from JSONL/JSON files in parallel.')
    parser.add_argument('-i', '--input',  help='Input directory with JSON/JSONL files',
                        default=config.json_dir)
    parser.add_argument('-o', '--output', help='Output directory for MCQs',
                        default=config.mcq_dir)
    parser.add_argument('-m','--model', help='Model to use',
                        default=config.defaultModel)
    parser.add_argument('-q','--quiet',   action='store_true',
                        help='No progress bar or messages')
    parser.add_argument('-v','--verbose', action='store_true',
                        help='Enable verbose logging')
    parser.add_argument('-p', '--parallel', type=int, default=1,
                        help='Number of files to process in parallel')

    args = parser.parse_args()

    # Decide logging level and whether to show a progress bar
    if args.verbose:
        config.logger.setLevel(logging.INFO)
        use_progress_bar = False  # or True if you want both info & bar
    elif args.quiet:
        config.logger.setLevel(logging.CRITICAL)
        use_progress_bar = False
    else:
        config.logger.setLevel(logging.WARNING)
        use_progress_bar = True

    input_directory = args.input
    output_directory = args.output
    model_name = args.model
    num_workers = args.parallel

    from model_access import Model
    model = Model(model_name)
    model.details()

    os.makedirs(output_directory, exist_ok=True)

    try:
        process_directory(
            model,
            input_directory,
            output_directory,
            use_progress_bar=use_progress_bar,
            num_workers=num_workers
        )
    except KeyboardInterrupt:
        print("EXIT: Execution interrupted by user")
        sys.exit(1)

