from __future__ import annotations

import os
import sys
import json
import time
import random
import logging
import threading
from concurrent.futures import Future

from aeris.behavior import action
from aeris.behavior import Behavior
from aeris.behavior import loop
from aeris.exchange.thread import ThreadExchange
from aeris.launcher.thread import ThreadLauncher
from aeris.logging import init_logging
from aeris.manager import Manager

import config

from simple_parse import extract_text_from_pdf
from generate_mcqs import split_text_into_chunks, generate_mcqs, NoOpTqdm
# from select_mcqs_at_random import select_random_entries
from model_access import Model

logger = logging.getLogger(__name__)


class PDFParser(Behavior):
    count: int

    def on_setup(self) -> None:
        self.count = 0

    @action
    def parse_pdf(self, file_path) -> str:

        basename, _ = os.path.splitext(file_path)
        out_file = basename + ".json"
        out_path  = os.path.join("", out_file)
        if os.path.isfile(out_path):
            logger.warning(f'Already exists: {out_path}')
            return 0

        logger.info(f"Processing file: {file_path}")

        (text, parser) = extract_text_from_pdf(file_path)

        json_structure = {'path': file_path, 'text': text, 'parser':parser}

        with open(out_path, 'w', encoding='utf-8') as out_f:
            json.dump(json_structure, out_f, ensure_ascii=False, indent=2)
        return out_file


class MCQGenerator(Behavior):
    model: Model
    all_prompt_answer_pairs: list

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def on_setup(self) -> None:
        self.model = Model(self.model_name)
        self.all_prompt_answer_pairs = []

    @action
    def generate_mcqs(self, file_path) -> str:
        CHUNK_SIZE = 1000
        num_chunks = 0
        with open(file_path, 'r', encoding='utf-8') as file:
            if file_path.lower().endswith(".json"):
                json_str = file.read()
                lines = [json_str]
            else:
                lines = file.readlines()

            for j, line in enumerate(lines, start=1):

                try:
                    record = json.loads(line.strip())
                except json.JSONDecodeError as e:
                    logger.info(f"JSON decode error in file {file_path} line {j}: {e}")

                text = record['text']
                path = record['path']
                chunks = split_text_into_chunks(text, CHUNK_SIZE)
                num_chunks += len(chunks)

                prompt_answer_pairs = generate_mcqs(self.model, path, file_path, j, chunks, NoOpTqdm())
                #prompt_answer_pairs = {'hello': 'test'}
                self.all_prompt_answer_pairs.extend(prompt_answer_pairs)

        out_file = f'mcqs_{file_path}'
        logger.info(f"Writing output to {out_file}")
        with open(out_file, 'w', encoding='utf-8') as out_f:
            json.dump(self.all_prompt_answer_pairs, out_f, ensure_ascii=False, indent=2)
        return out_file



class MCQSelector(Behavior):
    
    @action
    def select_mcqs(self, input_file, output_file, n=5) -> str:

        with open(input_file, 'r', encoding='utf-8') as infile:
            data = json.load(infile)

        # Ensure there are enough entries to sample
        if n > len(data):
            raise ValueError(f"Requested {n} entries, but the file only contains {len(data)} entries.")

        # Randomly sample N entries without replacement
        selected_entries = random.sample(data, n)

        # Write the sampled entries to the output file
        with open(output_file, 'w', encoding='utf-8') as outfile:
            json.dump(selected_entries, outfile, indent=4, ensure_ascii=False)

        logger.info(f"Selected {n} random entries written to {output_file}")

        return output_file
    


class MCQAnswerer(Behavior):
    model: Model

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def on_setup(self) -> None:
        self.model = Model(self.model_name)

    @action
    def answer_mcqs(self, file_path) -> str:
        return "dog"




class AnswerGenerator(Behavior):
    
    model: Model

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def on_setup(self) -> None:
        self.model = Model(self.model_name)

    @action
    def generate_answers(self, input_file, output_file, n=5) -> str:

        return 1
    

class AnswerScorer(Behavior):
    
    @action
    def score_answers(self, input_file, output_file, n=5) -> str:

        return 1

def main() -> int:
    init_logging(logging.INFO)

    with Manager(
        exchange=ThreadExchange(),
        launcher=ThreadLauncher(),
    ) as manager:
        pdf_parser = PDFParser()
        mcq_generator = MCQGenerator(model_name="alcf:mistralai/Mistral-7B-Instruct-v0.3")
        mcq_selector = MCQSelector()
        generate_answers = AnswerGenerator(model_name="alcf:meta-llama/Meta-Llama-3-70B-Instruct")
        score_answers = AnswerScorer()

        parser_agent = manager.launch(pdf_parser)
        mcq_gen_agent = manager.launch(mcq_generator)
        mcq_selector_agent = manager.launch(mcq_selector)

        future: Future[int] = parser_agent.action('parse_pdf', "example1.pdf")
        parsed_output = future.result()

        logger.info('Parser Agent parsed json: %s', future.result())

        future: Future[int] = mcq_gen_agent.action('generate_mcqs', parsed_output)
        
        mcq_output = future.result()

        logger.info('MCQ Agent generated MCQs: %s', future.result())

    return 0




if __name__ == '__main__':
    raise SystemExit(main())
