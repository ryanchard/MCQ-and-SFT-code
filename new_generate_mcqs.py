#!/usr/bin/env python3
"""
generate_mcqs.py

A script to generate multiple-choice questions (MCQs) from JSON or JSONL files.
This version integrates configuration via a YAML file (default: generate_mcqs_config.yaml)
and allows command-line arguments to override configuration values.

Usage example:
    python generate_mcqs.py -i input_directory -o output_directory [-m model_name] [-c config_file]
"""

import os
import sys
import json
import re
import time
import argparse
import yaml  # Requires PyYAML: pip install pyyaml
import spacy

from openai import OpenAI
from model_access import Model
from inference_auth_token import get_access_token
from alcf_inference_utilities import get_names_of_alcf_chat_models

# ---------------------------------------------------------------------------
# Load tokens and model lists (unchanged from your original code)
# ---------------------------------------------------------------------------
alcf_access_token = get_access_token()
alcf_chat_models = get_names_of_alcf_chat_models(alcf_access_token)

# Read OpenAI access token from file
with open('openai_access_token.txt', 'r') as file:
    openai_access_token = file.read().strip()

# ---------------------------------------------------------------------------
# Helper functions for configuration
# ---------------------------------------------------------------------------
def load_config(config_file):
    """
    Load the YAML configuration file if it exists.
    Returns a dictionary (or an empty dict if not found).
    """
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    return {}

def merge_config(defaults, config_file_settings, cli_args):
    """
    Merge the configuration in the following order:
      1. Built-in defaults,
      2. YAML configuration file settings,
      3. Command-line arguments (highest priority).
    Returns the unified configuration dictionary.
    """
    # Start with a copy of defaults
    merged = defaults.copy()

    # Update with settings from the config file
    for key, value in config_file_settings.items():
        merged[key] = value

    # Override with required command-line arguments
    merged['input'] = cli_args.input
    merged['output'] = cli_args.output

    # Override the model if provided via CLI; otherwise, use the value in merged['defaults']['model']
    if cli_args.model:
        if 'defaults' not in merged:
            merged['defaults'] = {}
        merged['defaults']['model'] = cli_args.model

    return merged

# ---------------------------------------------------------------------------
# Built-in defaults (matching your original hard-coded values)
# ---------------------------------------------------------------------------
DEFAULTS = {
    'api': {
        'openai_endpoint': 'https://api.openai.com/v1',
        'openai_access_token_file': 'openai_access_token.txt',
        'alcf_model_endpoint': 'https://data-portal-dev.cels.anl.gov/resource_server/sophia/vllm/v1'
    },
    'chunking': {
        'chunk_size': 1000,
        'spacy_model': 'en_core_web_sm'
    },
    'prompts': {
        'summarization_system': (
            "You are a helpful assistant that summarizes text in bullet points "
            "and expands on them using your broader knowledge. "
            "Name this result 'augmented_chunk'."
        ),
        'summarization_user': (
            "Given the following chunk of text, please:\n"
            "1. Summarize the text in bullet points.\n"
            "2. Expand on the summary using your parametric knowledge.\n"
            "Chunk:\n{chunk}\n"
            "Return the result as plain text labeled 'augmented_chunk:' at the start."
        ),
        'mcq_generation_system': (
            "You are a helpful assistant that generates exactly ONE multiple-choice question "
            "based on the provided text (augmented_chunk). The question must have 5 possible answers, "
            "numbered 1 to 5. Exactly one of these 5 choices is correct. "
            "Mark the correct choice with '(*)' at the end for later grading."
        ),
        'mcq_generation_user': (
            "Below is some content called augmented_chunk.\n"
            "Please:\n"
            "1) Create exactly one multiple-choice question that can be answered by the augmented_chunk.\n"
            "2) Provide five distinct options (1 to 5) as answers.\n"
            "3) Mark the correct answer with '(*)' at the end of that particular option.\n"
            "Constraints:\n"
            "- The question and answers must be self-contained and understandable without referencing the chunk.\n"
            "- Do not mention 'chunk' or 'augmented_chunk' or 'article' or 'study' in the final output.\n"
            "augmented_chunk:\n{augmented_chunk}"
        ),
        'verification_system': (
            "You are a helpful assistant that evaluates how well an answer "
            "matches the question in context of the augmented_chunk. "
            "Return your answer and a score from 1 to 10 in JSON form like:\n"
            '{"answer":"...","score":9}'
        ),
        'verification_user': (
            "augmented_chunk:\n{augmented_chunk}\n\n"
            "question:\n{question}\n\n"
            "Please provide:\n"
            "1. An appropriate answer to the multiple-choice question above. Your answer should identify which option is correct and why.\n"
            "2. A single integer 'score' from 1 to 10 for how well the answer addresses the question based on the augmented_chunk.\n\n"
            "Output must be valid JSON in the form: {\"answer\":\"...\",\"score\":9}"
        ),
        'json_fix_system': "You are a strict JSON converter.",
        'json_fix_user': (
            "Convert the following text strictly into valid JSON of the form: {\"answer\":\"...\",\"score\":9} "
            "Nothing else, no additional text.\nTEXT TO FIX:\n{text_to_fix}"
        )
    },
    'scoring': {
        'score_threshold': 7
    },
    'generation': {
        'temperature': 0.7
    },
    'defaults': {
        'model': "openai:gpt-4o"
    }
}

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def human_readable_time(seconds: float) -> str:
    """
    Convert time in seconds into a more human-friendly format.
    """
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        return f"{seconds/60:.2f} minutes"
    elif seconds < 86400:
        return f"{seconds/3600:.2f} hours"
    else:
        return f"{seconds/86400:.2f} days"

def split_text_into_chunks(text: str, chunk_size: int) -> list:
    """
    Split the text into chunks of ~chunk_size words, respecting sentence
    boundaries using spaCy for sentence segmentation.
    """
    # Load the specified spaCy model (from config)
    nlp = spacy.load(DEFAULTS['chunking']['spacy_model'])
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

# ---------------------------------------------------------------------------
# Core processing functions
# ---------------------------------------------------------------------------
def generate_mcqs(model, path, filename, linenum, chunks: list, config: dict) -> list:
    """
    For each text chunk:
      1. Summarize & expand the chunk into an 'augmented_chunk'
      2. Generate a multiple-choice question with 5 options, marking the correct one
      3. Verify the question by obtaining an answer and score (keeping only those with score > threshold)
    Returns a list of QA pairs.
    """
    qa_pairs = []
    threshold = config['scoring']['score_threshold']

    for chunknum, chunk in enumerate(chunks, start=1):
        # --- Step 1: Summarize & Expand ---
        system_message = config['prompts']['summarization_system']
        user_message = config['prompts']['summarization_user'].format(chunk=chunk)
        try:
            step1_output = model.run(user_prompt=user_message, system_prompt=system_message)
            augmented_chunk = step1_output
            if "augmented_chunk:" in step1_output.lower():
                augmented_chunk = re.split(
                    r'augmented_chunk\s*:\s*',
                    step1_output,
                    flags=re.IGNORECASE,
                    maxsplit=1
                )[-1].strip()
        except Exception as e:
            print(f"Error summarizing and expanding chunk: {e}")
            continue

        # --- Step 2: Generate MCQ ---
        system_message_2 = config['prompts']['mcq_generation_system']
        user_message_2 = config['prompts']['mcq_generation_user'].format(augmented_chunk=augmented_chunk)
        try:
            generated_question = model.run(user_prompt=user_message_2, system_prompt=system_message_2)
        except Exception as e:
            print(f"Error generating question: {e}")
            continue

        # --- Step 3: Verify the MCQ ---
        system_message_3 = config['prompts']['verification_system']
        user_message_3 = config['prompts']['verification_user'].format(
            augmented_chunk=augmented_chunk,
            question=generated_question
        )
        try:
            step3_output = model.run(user_prompt=user_message_3, system_prompt=system_message_3)
            # Clean up output formatting
            step3_output = step3_output.replace("```json", "").replace("```", "")
            step3_output = step3_output.replace('\\"', "XXXABCXXX")
            step3_output = step3_output.replace("\\", "\\\\")
            step3_output = step3_output.replace("XXXABCXXX", '\\"')
            parsed_json = json.loads(step3_output)
            model_answer = parsed_json.get("answer", "").strip()
            model_score = parsed_json.get("score", 0)
            print(f"{model_score}", end=" ")
            sys.stdout.flush()
            if isinstance(model_score, int) and model_score > threshold:
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
            print("\nDEBUG: JSON parsing failed. Trying to fix output...")
            fix_prompt = (
                "Convert the following text strictly into valid JSON of the form: "
                '{"answer":"...","score":9} '
                "Nothing else, no additional text.\n\nTEXT TO FIX:\n" + step3_output
            )
            try:
                fixed_json_output = model.run(
                    system_prompt=config['prompts']['json_fix_system'],
                    user_prompt=config['prompts']['json_fix_user'].format(text_to_fix=step3_output)
                )
                parsed_json = json.loads(fixed_json_output)
                model_answer = parsed_json.get("answer", "").strip()
                model_score = parsed_json.get("score", 0)
                print(f"{model_score}", end=" ")
                sys.stdout.flush()
                if isinstance(model_score, int) and model_score > threshold:
                    qa_pairs.append({
                        "question": generated_question,
                        "answer": model_answer,
                        "text": augmented_chunk
                    })
            except Exception as e:
                print(f"Could not fix JSON automatically: {e}")
                continue
        except Exception as e:
            print(f"\nError in verifying question/answer: {e}")
            continue

    print()  # For clarity after printing scores
    return qa_pairs

def process_directory(model, input_dir: str, output_dir: str, config: dict):
    """
    Process all JSON/JSONL files in the input directory:
      - Extract text from each file
      - Split the text into chunks using the configured chunk size
      - Generate QA pairs for each chunk
      - Save results to JSON files in the output directory
    """
    # Identify JSON and JSONL files
    json_files  = [f for f in os.listdir(input_dir) if f.lower().endswith(".json")]
    jsonl_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".jsonl")]
    all_files = json_files + jsonl_files
    total_files = len(all_files)

    if total_files == 0:
        print("No suitable files found in directory.")
        return

    overall_start_time = time.time()
    cumulative_time = 0.0
    processed_count = 0

    if jsonl_files:
        line_counts = []
        for i, filename in enumerate(jsonl_files, start=1):
            file_path = os.path.join(input_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                line_count = sum(1 for _ in file)
            line_counts.append(line_count)
        print(f'{len(jsonl_files)} JSONL files, with {sum(line_counts)} lines in total: {line_counts}')

    if json_files:
        print(f'We have {len(json_files)} JSON files')

    for i, filename in enumerate(all_files, start=1):
        all_prompt_answer_pairs = []
        num_chunks = 0
        file_path = os.path.join(input_dir, filename)
        file_start_time = time.time()
        print(f"\nProcessing file {i}/{total_files}: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as file:
            if filename.lower().endswith(".json"):
                json_str = file.read()
                lines = [json_str]
            else:
                lines = file.readlines()

            for j, line in enumerate(lines, start=1):
                print(f'Processing line {j} of {len(lines)} in file {i}')
                try:
                    dict_obj = json.loads(line.strip())
                except json.JSONDecodeError as e:
                    print(f"JSON decode error in file {filename}: {e}")
                    continue
                text = dict_obj.get('text', '')
                path = dict_obj.get('path', '')
                # Use the chunk size from config
                chunk_size = config['chunking']['chunk_size']
                chunks = split_text_into_chunks(text, chunk_size)
                num_chunks += len(chunks)
                prompt_answer_pairs = generate_mcqs(model, path, filename, j, chunks, config)
                all_prompt_answer_pairs.extend(prompt_answer_pairs)

        out_file = os.path.join(output_dir, f'file_{i}.json')
        print(f'Writing output for file {i} with {num_chunks} chunks to {out_file}')
        with open(out_file, 'w', encoding='utf-8') as out_f:
            json.dump(all_prompt_answer_pairs, out_f, ensure_ascii=False, indent=2)

        file_time_taken = time.time() - file_start_time
        processed_count += 1
        cumulative_time += file_time_taken
        avg_time = cumulative_time / processed_count
        remaining_files = total_files - processed_count
        estimated_time_remaining = remaining_files * avg_time
        print(
            f"Time for this file: {human_readable_time(file_time_taken)} | "
            f"Processed: {processed_count}/{total_files} | "
            f"Estimated remaining: {human_readable_time(estimated_time_remaining)}"
        )

    total_time = time.time() - overall_start_time
    print(
        f"\nDone! Processed {processed_count}/{total_files} files in "
        f"{human_readable_time(total_time)}.\n"
        f"Prompt/answer pairs (score > {config['scoring']['score_threshold']}) saved to {output_dir}."
    )
    if processed_count > 0:
        final_avg_time = total_time / processed_count
        print(f"Average time to process each file: {human_readable_time(final_avg_time)}")

def get_model_parameters(model_name):
    """
    Determine API key and endpoint based on the model.
    """
    if model_name in alcf_chat_models:
        key = alcf_access_token
        endpoint = DEFAULTS['api']['alcf_model_endpoint']
    elif model_name == 'gpt-4o' or model_name.startswith("openai"):
        key = openai_access_token
        endpoint = DEFAULTS['api']['openai_endpoint']
    else:
        print('Bad model:', model_name)
        print('Valid models are:', alcf_chat_models + ['gpt-4o'])
        sys.exit(1)
    return (model_name, key, endpoint)

def generate_mcqs_main(config: dict):
    """
    Main function that sets up the model and initiates the MCQ generation process.
    """
    input_directory = config['input']
    output_directory = config['output']
    model_name = config['defaults']['model']

    # Instantiate the model; your Model class should internally use get_model_parameters() as needed.
    model = Model(model_name)
    model.details()

    os.makedirs(output_directory, exist_ok=True)
    process_directory(model, input_directory, output_directory, config)

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate MCQs from JSON/JSONL files with configuration.')
    parser.add_argument('-i', '--input', required=True, help='Input directory or file.')
    parser.add_argument('-o', '--output', required=True, help='Output directory.')
    parser.add_argument('-m', '--model', help='Model to use for generating MCQs.')
    parser.add_argument('-c', '--config', default='generate_mcqs_config.yaml',
                        help='Path to the YAML configuration file (default: generate_mcqs_config.yaml)')
    args = parser.parse_args()

    # Load configuration from YAML file (if it exists) and merge with built-in defaults and CLI args.
    file_config = load_config(args.config)
    config = merge_config(DEFAULTS, file_config, args)

    # Uncomment the following lines to print the merged configuration for debugging:
    # import pprint
    # pprint.pprint(config)

    generate_mcqs_main(config)

