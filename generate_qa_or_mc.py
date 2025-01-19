#!/usr/bin/env python3

import os
import sys
import json
import re
import time  # For timing
import PyPDF2
import spacy
from openai import OpenAI

##############################################################################
# Explicit API settings
##############################################################################
# MODEL_NAME = "gpt-3.5-turbo"
TEMPERATURE = 0.7

OPENAI_EP  = 'https://api.openai.com/v1'

with open('alcf_access_token.txt', 'r') as file:
    alcf_access_token = file.read().strip()

with open('openai_access_token.txt', 'r') as file:
    openai_access_token = file.read().strip()


##############################################################################
# Global constants
##############################################################################
CHUNK_SIZE = 1000  # approximate number of words per chunk

alcf_chat_models = ['Qwen/Qwen2.5-14B-Instruct',
                    'Qwen/Qwen2.5-7B-Instruct',
                    'Qwen/QwQ-32B-Preview',
                    # Meta Llama Family
                    'meta-llama/Meta-Llama-3-70B-Instruct',
                    'meta-llama/Meta-Llama-3-8B-Instruct',
                    'meta-llama/Meta-Llama-3.1-70B-Instruct',
                    'meta-llama/Meta-Llama-3.1-8B-Instruct',
                    'meta-llama/Meta-Llama-3.1-405B-Instruct',
                    'meta-llama/Llama-3.3-70B-Instruct',
                    # Mistral Family
                    'mistralai/Mistral-7B-Instruct-v0.3',
                    'mistralai/Mistral-Large-Instruct-2407',
                    'mistralai/Mixtral-8x22B-Instruct-v0.1',
                    # Nvidia Nemotron Family
                    'mgoin/Nemotron-4-340B-Instruct-hf',
                    # Aurora GPT Family
                    'auroragpt/auroragpt-0.1-chkpt-7B-Base'
                    ]

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

def query_model(model, system_message, user_message, temperature, error_message):
    (modelname, key, ep) = model
    client = OpenAI(
        api_key  = key,
        base_url = ep
    )

    response = client.chat.completions.create(
        model=modelname,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user",   "content": user_message},
        ],
        temperature=temperature,
    )
    content = response.choices[0].message.content.strip()
    return content


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract all text from a PDF file, given its file path.
    """
    text_content = []
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_content.append(page_text)
    return "\n".join(text_content)


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

def summarize_and_expand_chunk(model, chunk):
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

    temperature = TEMPERATURE
    error_message = "Error summarizing and expanding chunk"

    step1_output = query_model(model, system_message, user_message, temperature, error_message)

    try:
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
        return None

    return augmented_chunk

def generate_qa_or_mc(errorfile, model, filenum, filename, chunks: list) -> list:
    """
    For each chunk:
      1) Summarize and expand the chunk => augmented_chunk
      2) Generate a question that can be answered by augmented_chunk
      3) Verify by prompting GPT with (question + augmented_chunk) to get
         an answer & a score (1-10). Keep only those with score > 7.

    Returns a list of dicts with {"prompt": question, "answer": model_answer}
    if they pass the score threshold.

    Prints scores inline as each chunk is processed.
    """
    qa_pairs = []
    scores_this_round = []  # to hold scores per chunk

    for chunknum, chunk in enumerate(chunks, start=1):
        print(f'\tChunk {chunknum}:', end=" ")
        sys.stdout.flush()
        # --------------------------------------------------------------------
        # Step 1: Summarize & expand the chunk => augmented_chunk
        # --------------------------------------------------------------------
        augmented_chunk = summarize_and_expand_chunk(model, chunk)

        # --------------------------------------------------------------------
        # Step 2: Generate a question that can be answered by the augmented_chunk
        # --------------------------------------------------------------------
        if qa_or_mc == 'qa':
            system_message_2 = (
                "You are a helpful assistant that generates a single question "
                "that can be answered by the provided text (augmented_chunk)."
            )
            user_message_2 = (
                f"Given the following augmented_chunk, please generate one question that "
                f"this augmented_chunk can answer:\n\n{augmented_chunk}\n"
                f"Please do not refer to the augmented_chunk nor to the article or study"
                f"Instead frame the question so that is it indepdent of any paper or study"
                f"and is understandable without additional context."
            )
        else:
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

        temperature = TEMPERATURE
        error_message = 'Error generating question'
        generated_question = query_model(model, system_message_2, user_message_2, temperature, error_message)

        # --------------------------------------------------------------------
        # Step 3: Verify the question by prompting GPT with the augmented_chunk
        # and requesting an answer + score (1-10).
        # --------------------------------------------------------------------
        if qa_or_mc == 'qa':
            system_message_3 = (
                "You are a helpful assistant that evaluates how well an answer "
                "matches the question in context of the augmented_chunk. "
                "Return your answer and a score from 1 to 10 in JSON form like:\n"
                '{"answer":"...","score":9}. Make very sure to generate valid JSON. Do NOT include Markdown formatting like ```json or any backticks.'
            )
            user_message_3 = (
                f"augmented_chunk:\n{augmented_chunk}\n\n"
                f"question:\n{generated_question}\n\n"
                f"Please provide:\n"
                f"1. An appropriate answer to the question above. The answer should be a complete sentence and contain enough of the augmented chunk to stand alone as a statement. \n"
                f"2. A single integer 'score' from 1 to 10 for how well the answer addresses the question based on the augmented_chunk.\n\n"
                f"Output MUST be valid JSON in the form:\n"
                f'{{"answer":"...","score":9}}.'
                f'Do NOT include Markdown formatting like "```json". do NOT include any backticks ("`").'
            )
       else:
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

        temperature = TEMPERATURE
        error_message = 'Error generating question'
        step3_output = query_model(model, system_message_3, user_message_3, temperature, error_message)

        #print(f'\n\tSTEP3 Output (chunk {chunknum} of file {filenum}: {filename}):\n    {step3_output}')

        try:
            # Attempt to parse the JSON
            parsed_json = json.loads(step3_output)
        except json.JSONDecodeError:
            #print("\nDEBUG: JSON parsing failed. Trying to fix output...")
            print("... fixing...", end=" ")
            # Attempt a second pass to fix the JSON
            step3_output = step3_output.replace("```json", '') # Common error in LLM output
            step3_output = step3_output.replace("```", '')     # Common error in LLM output
            step3_output = step3_output.replace(" \\( ", ' ')  # Seem to be product of bad PDF extraction
            step3_output = step3_output.replace(" \\) ", ' ')  # Seem to be product of bad PDF extraction
            try:
                parsed_json = json.loads(step3_output)
            except:
                print(f'Re-parse also failed on {step3_output}')
                error_string = f'Fail on chunk {chunknum} of file {filename}: {step3_output}\n'
                errorfile.write(error_string)
                continue

        model_answer = parsed_json.get("answer", "").strip()
        model_score = parsed_json.get("score", 0)
       # Print the score on the same line (no newline)
        #print(f"{model_score}", end=" ")
        print(f"{model_score}")
        sys.stdout.flush()

        # Keep only if score > 7
        if isinstance(model_score, int) and model_score > 7:
            qa_pairs.append({
                "question": generated_question,
                "answer": model_answer,
                "text" : augmented_chunk,
                "file" : filename,
                "filenum" : filenum,
                "chunknum" : chunknum
            })
            #d = { "question": generated_question, "answer": model_answer, "text" : augmented_chunk, "file" : filename, "filenum" : filenum, "chunknum" : chunknum }
            #with open(f'test_{filenum}_{chunknum}.json', 'w', encoding='utf-8') as out_f:
            #    json.dump(d, out_f, ensure_ascii=False, indent=2)

    return qa_pairs


def process_directory(errorfile, model, qa_or_mc, chunksize, input_dir: str, output_file: str = "output.json"):
    """
    Main function to:
    1) Iterate over all PDFs or TXT files in a directory.
    2) Extract text (for PDFs, convert; for TXT, read directly).
    3) Split into chunks.
    4) Generate Q/A pairs (including summarization & scoring) -- or generate MCs.
    5) Save to JSON only those that pass the score threshold.

    Adds:
    - Running count of files processed
    - Time to process each file
    - Estimated remaining time based on average processing time (human-friendly)
    - Print average time to process a file at the end
    """

    # Gather all PDF/TXT files
    files = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith(".pdf") or f.lower().endswith(".txt")
    ]
    total_files = len(files)
    if total_files == 0:
        print("No PDF or TXT files found in directory.")
        return

    all_prompt_qa_or_mc = []

    # Track timing
    overall_start_time = time.time()
    cumulative_time = 0.0
    processed_count = 0

    # Iterate over files
    for filenum, filename in enumerate(files, start=1):
        file_path = os.path.join(input_dir, filename)

        # Timestamp before processing this file
        file_start_time = time.time()

        print(f"\nProcessing ({filenum}/{total_files}): {file_path}")

        # 1) Extract text depending on file type
        if filename.lower().endswith(".pdf"):
            text = extract_text_from_pdf(file_path)
        else:
            # .txt file
            text = extract_text_from_txt(file_path)

        # 2) Split into chunks
        chunks = split_text_into_chunks(text, chunksize)

        # 3) Generate Q/A pairs
        prompt_qa_or_mc = generate_qa_or_mc(errorfile, model, filenum, filename, chunks)

        with open('tmp_file', 'a+', encoding='utf-8') as out_f:
            json.dump(prompt_qa_or_mc, out_f, ensure_ascii=False, indent=2)

        # 4) Accumulate results
        all_prompt_qa_or_mc.extend(prompt_qa_or_mc)

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

    # 5) Write all results to a JSON file
    with open(output_file, 'w', encoding='utf-8') as out_f:
        json.dump(all_prompt_qa_or_mc, out_f, ensure_ascii=False, indent=2)

    total_time = time.time() - overall_start_time
    # Final stats
    print(
        f"\nDone! Processed {processed_count}/{total_files} files in "
        f"{human_readable_time(total_time)}.\n"
        f"Prompt/answer pairs (score > 7) saved to {output_file}."
    )

    # Print the average time to process a file
    if processed_count > 0:
        final_avg_time_per_file = total_time / processed_count
        print(f"Average time to process each file: {human_readable_time(final_avg_time_per_file)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Program to generate QA pairs or MC questions from PDF')
    parser.add_argument('-i','--inputdir', help='Input PDF directory', required=True)
    parser.add_argument('-q','--qa', help='Generate QA pairs rather than MCQs', action="store_true")
    parser.add_argument('-o','--output', help='Output file for JSON (default=output_file)', default='output_file')
    parser.add_argument('-c','--chunksize', help=f'Chunk size (default={CHUNK_SIZE})', default=CHUNK_SIZE)
    args = parser.parse_args()

    if args.qa:
        qa_or_mc = 'qa'
    else:
        qa_or_mc = 'mc'

    input_directory = args.inputdir
    output_json     = args.output + '.json'
    error_file      = args.output + '.error'
    model           = get_model_parameters('gpt-4o')

    print(f'Generating {qa_or_mc} from {input_directory} to {output_json} with {model[0]} and chunk size {args.chunksize}')

    errorfile = open(error_file, "w")

    process_directory(errorfile, model, qa_or_mc, args.chunksize, input_directory, output_json)
