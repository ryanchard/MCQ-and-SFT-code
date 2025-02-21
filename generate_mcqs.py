#!/usr/bin/env python3

import os
import sys
import json
import re
import time  # For timing
from openai import OpenAI
import spacy
import argparse
import config
import logging
import argparse
from tqdm import tqdm  # CeC: tqdm for progress bar

from model_access import Model
from inference_auth_token import get_access_token
alcf_access_token = get_access_token()

from alcf_inference_utilities import get_names_of_alcf_chat_models
alcf_chat_models = get_names_of_alcf_chat_models(alcf_access_token)

##############################################################################
# Global constants
##############################################################################
CHUNK_SIZE = 1000  # approximate number of words per chunk


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

######
# Functions
######

def human_readable_time(seconds: float) -> str:
    """
    Convert time in seconds into a more human-friendly format
    (seconds, minutes, hours, days).
    """
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

def approximate_total_chunks(input_dir, bytes_per_chunk=5000):
    """
    Returns an approximate total chunk count by summing file sizes
    of .json or .jsonl files and dividing by `bytes_per_chunk`.
    """
    total_bytes = 0
    for f in os.listdir(input_dir):
        if f.lower().endswith((".json", ".jsonl")):
            path = os.path.join(input_dir, f)
            size = os.stat(path).st_size  # file size in bytes
            total_bytes += size
    return total_bytes // bytes_per_chunk

def split_text_into_chunks(text: str, chunk_size: int = CHUNK_SIZE) -> list:
    """
    Split the text into chunks of ~chunk_size words, respecting sentence
    boundaries using spaCy for sentence segmentation.
    """
    nlp = spacy.load("en_core_web_sm")
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

def generate_mcqs(model, path, filename, linenum, chunks: list, pbar) -> list:
    """
    For each chunk:
      1) Summarize and expand the chunk => augmented_chunk
      2) Generate a MULTIPLE-CHOICE question with 5 possible answers,
         marking the correct answer for later grading.
      3) Verify by prompting GPT with (question + augmented_chunk) to get
         an answer & a score (1-10). Keep only those with score > 7.

    Returns a list of dicts with the QA pair if they pass the score threshold.
    """
    qa_pairs = []

    for chunknum, chunk in enumerate(chunks, start=1):
        # Step 1: Summarize & expand the chunk
        system_message = (
            "You are a helpful assistant that summarizes text in bullet points "
            "and expands on them using your broader knowledge. "
            "Name this result 'augmented_chunk'."
        )
        user_message = (
            f"Given the following chunk of text, please:\n\n"
            f"1. Summarize the text in bullet points.\n"
            f"2. Expand on the summary using your parametric knowledge.\n\n"
            f"Chunk:\n{chunk}\n\n"
            f"Return the result as plain text labeled 'augmented_chunk:' at the start."
        )
        try:
            step1_output = model.run(user_prompt=user_message, system_prompt=system_message)
            augmented_chunk = step1_output
            if "augmented_chunk:" in str(step1_output).lower():
                augmented_chunk = re.split(
                    r'augmented_chunk\s*:\s*',
                    step1_output,
                    flags=re.IGNORECASE,
                    maxsplit=1
                )[-1].strip()
        except Exception as e:
            config.logger.warning(f"Error summarizing and expanding chunk: {e}")
            # Check for fatal errors
            if "401" in str(e) or "Unauthorized" in str(e):
                sys.exit("Model API Authentication failed. Exiting.")
            pbar.update(1)
            continue

        # Step 2: Generate a MULTIPLE-CHOICE question with 5 answers
        system_message_2 = (
            "You are a helpful assistant that generates exactly ONE multiple-choice question "
            "based on the provided text (augmented_chunk). The question must have 5 possible answers, "
            "numbered 1 to 5. Exactly one of these 5 choices is correct. "
            "Mark the correct choice with '(*)' at the end for later grading."
        )
        user_message_2 = (
            f"Below is some content called augmented_chunk.\n"
            f"Please:\n"
            f"1) Create exactly one multiple-choice question that can be answered by the augmented_chunk.\n"
            f"2) Provide five distinct options (1 to 5) as answers.\n"
            f"3) Mark the correct answer with '(*)' at the end of that particular option.\n\n"
            f"Constraints:\n"
            f"- The question and answers must be self-contained and understandable without referencing the chunk.\n"
            f"- Do not mention 'chunk' or 'augmented_chunk' or 'article' or 'study' in the final output.\n\n"
            f"augmented_chunk:\n{augmented_chunk}\n"
        )
        try:
            generated_question = model.run(user_prompt=user_message_2, system_prompt=system_message_2)
        except Exception as e:
            config.logger.warning(f"Error generating question: {e}")
            # Check for fatal errors
            if "401" in str(e) or "Unauthorized" in str(e):
                sys.exit("Model API Authentication failed. Exiting.")
            pbar.update(1)
            continue

        # Step 3: Verify the question by prompting GPT with the augmented_chunk
        system_message_3 = (
            "You are a helpful assistant that evaluates how well an answer "
            "matches the question in context of the augmented_chunk. "
            "Return your answer and a score from 1 to 10 in JSON form like:\n"
            '{"answer":"...","score":9}'
        )
        user_message_3 = (
            f"augmented_chunk:\n{augmented_chunk}\n\n"
            f"question:\n{generated_question}\n\n"
            f"Please provide:\n"
            f"1. An appropriate answer to the multiple-choice question above. "
            f"Your answer should identify which option is correct and why. "
            f"2. A single integer 'score' from 1 to 10 for how well the answer "
            f"addresses the question based on the augmented_chunk.\n\n"
            f"Output must be valid JSON in the form:\n"
            f'{{"answer":"...","score":9}}'
        )
        try:
            step3_output = model.run(user_prompt=user_message_3, system_prompt=system_message_3)
            step3_output = step3_output.replace("```json", "").replace("```", "")
            step3_output = step3_output.replace('\\"', "XXXABCXXX")
            step3_output = step3_output.replace("\\", "\\\\")
            step3_output = step3_output.replace("XXXABCXXX", '\\"')

            parsed_json = json.loads(step3_output)
            #model_answer = parsed_json.get("answer", "").strip()
            # replace line above with this one to force a string and save an exception when answer is an int
            model_answer = str(parsed_json.get("answer", "")).strip()
            model_score = parsed_json.get("score", 0)

            # Instead of printing inline, update the progress bar postfix
            pbar.set_postfix_str(f"Score: {model_score}")

            if isinstance(model_score, int) and model_score > 7:
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

        except json.JSONDecodeError:
            config.logger.warning("JSON parsing failed. Trying to fix output...")
            fix_prompt = f"""
            Convert the following text strictly into valid JSON of the form:
            {{"answer":"...","score":9}}
            Nothing else, no additional text.

            TEXT TO FIX:
            {step3_output}
            """
            try:
                fixed_json_output = model.run(system_prompt="You are a strict JSON converter.", user_prompt=fix_prompt)
                parsed_json = json.loads(fixed_json_output)
                model_answer = parsed_json.get("answer", "").strip()
                model_score = parsed_json.get("score", 0)
                pbar.set_postfix_str(f"Score: {model_score}")

                if isinstance(model_score, int) and model_score > 7:
                    qa_pairs.append({
                        "question": generated_question,
                        "answer": model_answer,
                        "text": augmented_chunk
                    })
            except Exception as e:
                config.logger.warning(f"Could not fix JSON automatically: {e}")
                pbar.update(1)
                continue

        except Exception as e:
            config.logger.warning(f"Error in verifying question/answer: {e}")
            pbar.update(1)
            continue

        # Update the progress bar after processing this chunk
        pbar.update(1)

    # Add a newline after processing all chunks in this call
    config.logger.info("")
    return qa_pairs

def process_directory(model, input_dir: str, output_dir: str = "output_files"):
    """
    Main function to:
    1) Iterate over all JSON/JSONL files in a directory.
    2) Extract text.
    3) Split into chunks.
    4) Generate Q/A pairs (including summarization & scoring).
    5) Save to JSON only those that pass the score threshold.
    """
    json_files  = [f for f in os.listdir(input_dir) if f.lower().endswith(".json")]
    jsonl_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".jsonl")]
    all_files = json_files + jsonl_files
    total_files = len(all_files)

    if total_files == 0:
        config.logger.warning("No suitable files found in directory.")
        return

    overall_start_time = time.time()
    cumulative_time = 0.0
    processed_count = 0

    if len(jsonl_files) > 0:
        line_counts = []
        for i, filename in enumerate(jsonl_files, start=1):
            file_path = os.path.join(input_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                line_count = sum(1 for _ in file)
            line_counts.append(line_count)
        config.logger.info(f'{len(jsonl_files)} JSONL files, with {sum(line_counts)} lines in total: {line_counts}')

    if len(json_files) > 0:
        approximate_chunk_count = approximate_total_chunks(input_dir, bytes_per_chunk=5000)
        config.logger.info(f"\nTotal JSON files: {total_files}, ~{approximate_chunk_count} chunks\n")
    else:
        # Fallback if only JSONL files exist (estimate by summing lines)
        approximate_chunk_count = sum(line_counts)

    # Create a global tqdm progress bar for chunk processing
#    pbar = tqdm(total=approximate_chunk_count, desc="Chunks processed", unit="chunk")
    if config.get_quiet_mode():
        # In quiet mode, we want to show the progress bar.
        pbar = tqdm(total=approximate_chunk_count, desc="Chunks processed", unit="chunk")
    else:
        # In non-quiet mode, we suppress the progress bar.
        pbar = NoOpTqdm()


    # Iterate over files
    for i, filename in enumerate(all_files, start=1):
        all_prompt_answer_pairs = []
        num_chunks = 0
        file_path = os.path.join(input_dir, filename)
        file_start_time = time.time()

        config.logger.info(f"Processing file {i}/{total_files}: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as file:
            if filename.lower().endswith(".json"):
                json_str = file.read()
                lines = [json_str]
            else:
                lines = file.readlines()

            for j, line in enumerate(lines, start=1):
                # JSON file will get read as one line most of the time so this msg is
                # misleading and not helpful
                #config.logger.info(f"Processing line {j} of {len(lines)} in file {i}")
                try:
                    record = json.loads(line.strip())
                except json.JSONDecodeError as e:
                    config.logger.info(f"JSON decode error in file {filename} line {j}: {e}")
                    continue

                text = record['text']
                path = record['path']
                chunks = split_text_into_chunks(text, CHUNK_SIZE)
                num_chunks += len(chunks)

                prompt_answer_pairs = generate_mcqs(model, path, filename, j, chunks, pbar)
                all_prompt_answer_pairs.extend(prompt_answer_pairs)

        out_file = f'{output_dir}/file_{i}.json'
        config.logger.info(f"Writing output for file {i} with {num_chunks} chunks to {out_file}")
        with open(out_file, 'w', encoding='utf-8') as out_f:
            json.dump(all_prompt_answer_pairs, out_f, ensure_ascii=False, indent=2)

        file_end_time = time.time()
        file_time_taken = file_end_time - file_start_time
        processed_count += 1
        cumulative_time += file_time_taken
        avg_time_per_file_so_far = cumulative_time / processed_count
        remaining_files = total_files - processed_count
        estimated_time_remaining = remaining_files * avg_time_per_file_so_far

        config.logger.info(
            f"Time for this file: {human_readable_time(file_time_taken)} | "
            f"Processed: {processed_count}/{total_files} | "
            f"Estimated remaining: {human_readable_time(estimated_time_remaining)}"
        )

    total_time = time.time() - overall_start_time
    config.logger.info(
        f"\nDone! Processed {processed_count}/{total_files} files in "
        f"{human_readable_time(total_time)}.\n"
        f"Prompt/answer pairs (score > 7) saved to {output_dir}."
    )

    if processed_count > 0:
        final_avg_time_per_file = total_time / processed_count
        config.logger.info(f"Average time to process each file: {human_readable_time(final_avg_time_per_file)}")

    # Final logging (after processing all files)
    total_time = time.time() - overall_start_time
    config.logger.info(
        f"\nDone! Processed {processed_count}/{total_files} files in "
        f"{human_readable_time(total_time)}.\n"
        f"Prompt/answer pairs (score > 7) saved to {output_dir}."
    )
    if processed_count > 0:
        final_avg_time_per_file = total_time / processed_count
        config.logger.info(f"Average time to process each file: {human_readable_time(final_avg_time_per_file)}")

    # Force the progress bar to complete by updating it to the total if necessary
    remaining = pbar.total - pbar.n
    if remaining > 0:
        pbar.update(remaining)
    # Close the progress bar
    pbar.close()


def get_model_parameters(model):
    if model in alcf_chat_models:
        key      = alcf_access_token
        endpoint = 'https://data-portal-dev.cels.anl.gov/resource_server/sophia/vllm/v1'
    elif model == 'gpt-4o':
        key      = openai_access_token
        endpoint = OPENAI_EP
    else:
        config.logger.warning('Bad model:', model)
        config.logger.warning('Valid models are:', alcf_chat_models + 'gpt-4o')
        exit(1)
    return (model, key, endpoint)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Program to generate MCQs from JSONL or JSON files')
    parser.add_argument('-i','--input', help='QA input file', required=True)
    parser.add_argument('-o','--output', help='Output directory', required=True)
    parser.add_argument('-m','--model', help='Model to use to generate MCQs',
                        default=config.defaultModel)
    parser.add_argument('-q','--quiet', help='Suppress informational msgs',
                        action="store_true")
    args = parser.parse_args()

    #if args.quiet:
    #    pbar = tqdm(total=100, desc="Processing", unit="chunk")  # ...and progress bar
    #else:
    #    pbar = NoOpTqdm()         # Default: Don't show progress bar (too noisy w/ INFO)

    config.set_quiet_mode(args.quiet)
    #logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')

    input_directory = args.input
    output_json     = args.output

    model_name = args.model
    model = Model(model_name)
    model.details()

    os.makedirs(output_json, exist_ok=True)

    try:
        process_directory(model, input_directory, output_json)
    except KeyboardInterrupt:
        print("EXIT: Execution interrupted by user")
        sys.exit(0)

