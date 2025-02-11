#!/usr/bin/env python3

import os
import sys
import json
import re
import time  # For timing
from openai import OpenAI
import spacy

from model_access import Model

#from inference_auth_token import get_access_token
#alcf_access_token = get_access_token()
#
#from alcf_inference_utilities import get_names_of_alcf_chat_models
#alcf_chat_models = get_names_of_alcf_chat_models(alcf_access_token)
#print("DEBUG: alcf_chat_models is", alcf_chat_models, flush=True)

from model_access import Model, ALCF_CHAT_MODELS, ALCF_ACCESS_TOKEN
print("DEBUG (gm.py): ALCF_CHAT_MODELS is", ALCF_CHAT_MODELS, flush=True)

OPENAI_EP  = 'https://api.openai.com/v1'
with open('openai_access_token.txt', 'r') as file:
    openai_access_token = file.read().strip()

##############################################################################
# Global constants
##############################################################################
CHUNK_SIZE = 1000  # approximate number of words per chunk

def human_readable_time(seconds: float) -> str:
    """
    Convert time in seconds into a more human-friendly format
    (seconds, minutes, hours, days).
    """
    # Less than a minute
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    # Less than an hour
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f} minutes"
    # Less than a day
    elif seconds < 86400:
        hours = seconds / 3600
        return f"{hours:.2f} hours"
    else:
        days = seconds / 86400
        return f"{days:.2f} days"


def split_text_into_chunks(text: str, chunk_size: int = CHUNK_SIZE) -> list:
    """
    Split the text into chunks of ~chunk_size words, respecting sentence
    boundaries using spaCy for sentence segmentation.
    """
    # Load spaCy English model
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)

    sentences = [sent.text.strip() for sent in doc.sents]

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        word_count = len(sentence.split())
        
        # If adding this sentence exceeds the chunk_size, start a new chunk
        if current_length + word_count > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = word_count
        else:
            current_chunk.append(sentence)
            current_length += word_count

    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def generate_mcqs(model, path, filename, linenum, chunks: list) -> list:
    """
    For each chunk:
      1) Summarize and expand the chunk => augmented_chunk
      2) Generate a MULTIPLE-CHOICE question with 5 possible answers, 
         marking the correct answer for later grading.
      3) Verify by prompting GPT with (question + augmented_chunk) to get
         an answer & a score (1-10). Keep only those with score > 7.

    Returns a list of dicts with {"question": question, "answer": model_answer, "text": augmented_chunk}
    if they pass the score threshold.
    """
    qa_pairs = []
    scores_this_round = []  # to hold scores per chunk

    for chunknum, chunk in enumerate(chunks, start=1):
        # --------------------------------------------------------------------
        # Step 1: Summarize & expand the chunk => augmented_chunk
        # --------------------------------------------------------------------
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

        """
        (modelname, key, ep) = model
        client = OpenAI(
            api_key  = key,
            base_url = ep
        )

        # Skip the old recovery code as it isn't in the Model class
        try:
            response_1 = client.chat.completions.create(
                model=modelname,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.7,
            )

            # Attempt to parse out the augmented chunk from the model's response
            step1_output = response_1.choices[0].message.content.strip()
        """
        try:
            step1_output = model.run(user_prompt=user_message, system_prompt=system_message)
            # We'll assume the model starts with "augmented_chunk:"
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

        # --------------------------------------------------------------------
        # Step 2 (MODIFIED): Generate a MULTIPLE-CHOICE question with 5 answers
        # --------------------------------------------------------------------
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

        """
        try:
            response_2 = client.chat.completions.create(
                model=modelname,
                messages=[
                    {"role": "system", "content": system_message_2},
                    {"role": "user", "content": user_message_2},
                ],
                temperature=0.7,
            )
            generated_question = response_2.choices[0].message.content.strip()
        """
        try:
            generated_question = model.run(user_prompt=user_message_2, system_prompt=system_message_2)
        except Exception as e:
            print(f"Error generating question: {e}")
            continue

        # --------------------------------------------------------------------
        # Step 3: Verify the question by prompting GPT with the augmented_chunk
        # and requesting an answer + score (1-10).
        # --------------------------------------------------------------------
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

        """
        try:
            response_3 = client.chat.completions.create(
                model=modelname,
                messages=[
                    {"role": "system", "content": system_message_3},
                    {"role": "user", "content": user_message_3},
                ],
                temperature=0.7,
            )
            step3_output = response_3.choices[0].message.content.strip()
        """    

        try:
            step3_output = model.run(user_prompt=user_message_3, system_prompt=system_message_3)
            step3_output = step3_output.replace("```json", "")
            step3_output = step3_output.replace("```", "")
            step3_output = step3_output.replace('\\"', "XXXABCXXX")
            step3_output = step3_output.replace("\\", "\\\\")
            step3_output = step3_output.replace("XXXABCXXX", '\\"')

            # Attempt to parse the JSON
            parsed_json = json.loads(step3_output)
            model_answer = parsed_json.get("answer", "").strip()
            model_score = parsed_json.get("score", 0)

            # Print the score on the same line (no newline)
            print(f"{model_score}", end=" ")
            sys.stdout.flush()
            
            # Keep only if score > 7
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
            print("\nDEBUG: JSON parsing failed. Trying to fix output...")

            # Attempt a second pass to fix the JSON
            fix_prompt = f"""
            Convert the following text strictly into valid JSON of the form:
            {{"answer":"...","score":9}} 
            Nothing else, no additional text.

            TEXT TO FIX:
            {step3_output}
            """
            try:
                fixed_json_output = model.run(system_prompt="You are a strict JSON converter.", user_prompt=fix_prompt)
                """
                fix_response = client.chat.completions.create(
                    model=modelname,
                    messages=[
                        {"role": "system", "content": "You are a strict JSON converter."},
                        {"role": "user", "content": fix_prompt},
                    ],
                    temperature=0.0,
                )
                fixed_json_output = fix_response.choices[0].message.content.strip()
                """
                parsed_json = json.loads(fixed_json_output)
                model_answer = parsed_json.get("answer", "").strip()
                model_score = parsed_json.get("score", 0)

                # Print the score on the same line (no newline)
                print(f"{model_score}", end=" ")
                sys.stdout.flush()
                
                if isinstance(model_score, int) and model_score > 7:
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

    # At the end of processing all chunks, print a newline for clarity
    print()
    return qa_pairs


def process_directory(model, input_dir: str, output_dir: str = "output_files"):
    """
    Main function to:
    1) Iterate over all PDFs or TXT files in a directory.
    2) Extract text (for PDFs, convert; for TXT, read directly).
    3) Split into chunks.
    4) Generate Q/A pairs (including summarization & scoring).
    5) Save to JSON only those that pass the score threshold.

    Adds:
    - Running count of files processed
    - Time to process each file
    - Estimated remaining time based on average processing time (human-friendly)
    - Print average time to process a file at the end
    """

    # As AdaParse generates JSONL files, we allow for both
    json_files  = [ f for f in os.listdir(input_dir) if f.lower().endswith(".json") ]
    jsonl_files = [ f for f in os.listdir(input_dir) if f.lower().endswith(".jsonl")]

    all_files = json_files + jsonl_files
    total_files = len(all_files)

    if total_files == 0:
        print("No suitable files found in directory.")
        return

    # Track timing
    overall_start_time = time.time()
    cumulative_time = 0.0
    processed_count = 0

    if len(jsonl_files) > 0:
        line_counts = []
        for i, filename in enumerate(jsonl_files, start=1):
            file_path = os.path.join(input_dir, filename)
            with open(os.path.join(input_dir, filename), 'r', encoding='utf-8') as file:
                line_count = 0
                for line in file:
                    # Parse each line as a JSON object and append to the list
                    line_count += 1
            line_counts.append(line_count)
        print(f'{len(jsonl_files)} JSONL files, with {sum(line_counts)} lines in total: {line_counts}')

    if len(json_files) > 0:
        print(f'We have {len(json_files)} JSON files')

    # Iterate over files
    for i, filename in enumerate(all_files, start=1):
        all_prompt_answer_pairs = []
        num_chunks = 0

        file_path = os.path.join(input_dir, filename)

        # Timestamp before processing this file jsonl
        file_start_time = time.time()

        print(f"\nProcessing file {i}/{total_files}: {file_path}")

        # 1) Extract text 
        with open(os.path.join(input_dir, filename), 'r', encoding='utf-8') as file:
            if filename.lower().endswith(".json"):
                json_str = file.read()
                lines = [json_str]
            else:
                lines = file.readlines()

            for j, line in enumerate(lines, start=1):
                print(f'Processing line {j} of {len(lines)} in file {i}')
                # Parse each line as a JSON object and append to the list
                dict = json.loads(line.strip())
                text = dict['text']
                path = dict['path']

                # 2) Split into chunks
                chunks = split_text_into_chunks(text, CHUNK_SIZE)
                num_chunks += len(chunks)

                # 3) Generate Q/A pairs
                prompt_answer_pairs = generate_mcqs(model, path, filename, j, chunks)

                # 4) Accumulate results
                all_prompt_answer_pairs.extend(prompt_answer_pairs)

        # 5) Write all results to a JSON file
        out_file = f'{output_dir}/file_{i}.json'
        print(f'Writing output for file {i} with {num_chunks} chunks to {out_file}')
        with open(out_file, 'w', encoding='utf-8') as out_f:
            json.dump(all_prompt_answer_pairs, out_f, ensure_ascii=False, indent=2)

        # Timestamp after processing
        file_end_time = time.time()
        file_time_taken = file_end_time - file_start_time

        # Update counters
        processed_count += 1
        cumulative_time += file_time_taken

        # Calculate average time per processed file so far
        avg_time_per_file_so_far = cumulative_time / processed_count

        # Estimate remaining time
        remaining_files = total_files - processed_count
        estimated_time_remaining = remaining_files * avg_time_per_file_so_far

        # Print timing stats (human-friendly)
        print(
            f"Time for this file: {human_readable_time(file_time_taken)} | "
            f"Processed: {processed_count}/{total_files} | "
            f"Estimated remaining: {human_readable_time(estimated_time_remaining)}"
        )

    total_time = time.time() - overall_start_time
    # Final stats
    print(
        f"\nDone! Processed {processed_count}/{total_files} files in "
        f"{human_readable_time(total_time)}.\n"
        f"Prompt/answer pairs (score > 7) saved to {output_dir}."
    )

    # Print the average time to process a file
    if processed_count > 0:
        final_avg_time_per_file = total_time / processed_count
        print(f"Average time to process each file: {human_readable_time(final_avg_time_per_file)}")

def get_model_parameters(model):
    # Force immediate printing of available models
    print("Available models:", alcf_chat_models, flush=True)

    if model in alcf_chat_models:
        key = alcf_access_token
        endpoint = 'https://data-portal-dev.cels.anl.gov/resource_server/sophia/vllm/v1'
    elif model == 'gpt-4o':
        key = openai_access_token
        endpoint = OPENAI_EP  # Ensure OPENAI_EP is defined
    else:
        print('Bad model:', model)
        # Concatenate lists properly
        print('Valid models are:', alcf_chat_models + ["gpt-4o"])
        exit(1)
    return (model, key, endpoint)


def old_get_model_parameters(model):

   #if model == 'mistralai/Mistral-7B-Instruct-v0.3':
   if model in alcf_chat_models:
       key      = alcf_access_token
       endpoint = 'https://data-portal-dev.cels.anl.gov/resource_server/sophia/vllm/v1'
   elif model == 'gpt-4o':
       key      = openai_access_token
       endpoint = OPENAI_EP
   else:
       print('Bad model:', model)
       print('Valid models are:', alcf_chat_models + ["gpt-4o"]) 
       exit(1)
   return (model, key, endpoint)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Program to generate MCQs from JSONL or JSON files')
    parser.add_argument('-i','--input', help='QA input file', required=True)
    parser.add_argument('-o','--output', help='Output directory', required=True)
    parser.add_argument('-m','--model', help='Model to use to generate MCQs', default='openai:gpt-4o')
    args = parser.parse_args()
                                            
    input_directory = args.input
    output_json     = args.output

    model_name = args.model
    model = Model(model_name)
    model.details()

    os.makedirs(output_json, exist_ok=True)
    process_directory(model, input_directory, output_json)
