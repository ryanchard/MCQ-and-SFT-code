from __future__ import annotations

import os
import sys
import json
import time
import random
import logging
import threading
import statistics
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
from score_answers import score_answer

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
    def answer_mcqs(self, input_file, start_num=0) -> str:
        # Load question-answer pairs
        try:
            with open(input_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except:
            config.logger.error(f"ERROR: file {input_file} not found.")
            sys.exit(0)

        qa_pairs = []

        start_time = time.time()
        total_time = 0

        data = data[start_num:]

        config.logger.info(f'Generating {len(data)} answers with model {self.model_name}')

        pbar = NoOpTqdm(total=len(data))

        # Use JSON Lines file format, append each result immediately rather
        # than waiting to write everything at the end
        # Determine the output file name (changed extension to .jsonl).
        if int(start_num) == 0:
            output_file = f'answers_{self.model_name.replace("/","+")}.json'
        # else:
        #     output_file = f'answers_{model_name.replace("/","+")}_{start_num}_{args.end}.jsonl'
        output_path = os.path.join('', output_file)

        # Remove existing output file if present; start fresh
        if os.path.exists(output_path):
            os.remove(output_path)

        for (qa_pair, index) in zip(data, range(1, len(data) + 1)):
            question = qa_pair.get("question", "")
            reference_answer = qa_pair.get("answer", "")
            filename         = qa_pair.get("file", "")
            filenum          = qa_pair.get("filenum", "")
            chunknum         = qa_pair.get("chunknum", "")

            if not question or not reference_answer:
                config.logger.info("not a question or ref_answer")
                continue  # skip malformed items

            # Use the model to generate an answer
            try:
                model_answer = self.model.run(question)
            except KeyboardInterrupt:
                config.logger.warning("EXIT: Execution interrupted by user")
                sys.exit(0)
            except Exception as e:
                config.logger.error(f"ERROR: {e}")
                sys.exit(0)

            gen_time    = time.time() - start_time
            total_time += gen_time
            start_time  = time.time()
            if index%10==0:
                avg_time = total_time / index  # Average time per item so far
                config.logger.info(f'{index} ({avg_time:.2f} s)', end =' ', flush=True) 

            new_tuple = {'file':filename, 'filenum':filenum, 'chunknum':chunknum,
                        'gen_time': f'{gen_time:.3f}',
                        'question':question, 'reference': reference_answer,
                        'model': model_answer}
            qa_pairs.append(new_tuple)


            # Append the new result immediately to the output file.

            with open(output_path, 'a', encoding='utf-8') as out_f:
                out_f.write(json.dumps(new_tuple, ensure_ascii=False) + "\n")

        logger.info("Processing complete")
        return output_path



class AnswerScorer(Behavior):
    modela: Model
    modelb: Model

    def __init__(self, model_a_name: str, model_b_name: str) -> None:
        self.model_a_name = model_b_name
        self.model_b_name = model_b_name

    def on_setup(self) -> None:
        self.modela = Model(self.model_a_name)
        self.modelb = Model(self.model_b_name)

    @action
    def score_answers(self) -> str:

         # Load previously generated answers from modelA
        answer_file = 'answers_'+self.model_a_name.replace('/', '+')+'.json'
        print(f'Looking for {answer_file}')
        if not os.path.exists(answer_file):
            print(f'No answers file for {self.model_a_name}')
            exit(1)

        score_file = f'scores_{self.model_a_name.replace("/","+")}={self.model_b_name.replace("/","+")}.json'
        # if os.path.exists(score_file) and not args.force:
        #     print('Score file already exists:', score_file)
        #     exit(1)

        logger.info(f"Writing scores to {score_file}")
        out_f = open(score_file, 'w', encoding='utf-8') 



        logger.info(f"Loading question-answer pairs from {answer_file}")
        # Load question-answer pairs
        with open(answer_file, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
            # data = json.load(f)

        scores   = []
        qa_pairs = []

        start_time = time.time()
        total_time = 0
        eval_answer_total_time = 0

        logger.info(f'Processing {len(data)} Q pairs')
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
                print('Question ', question)
                print('Reference', reference_answer)
                print('Model    ', model_answer)
                exit(1)
                continue  # skip malformed items

            # Use model to evaluate/grade the generated answer in file
            # against the reference answer
            eval_answer_start_time = time.time()
            score = score_answer(index, self.modelb, question, reference_answer, model_answer)
            eval_answer_time = time.time() - eval_answer_start_time
            eval_answer_total_time += eval_answer_time
            if score != None:
                scores.append(score)
                qa_pairs.append({'modelA': self.model_a_name, 'modelB': self.model_b_name, 'index': index, 'question': question, 'reference':reference_answer, 'model':model_answer, 'score':score, 'gen_time': gen_time, 'eval_time': f'{eval_answer_time:.4f}', 'file':file, 'filenum':filenum, 'chunknum':chunknum})

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

        return scores



def main() -> int:
    init_logging(logging.INFO)

    with Manager(
        exchange=ThreadExchange(),
        launcher=ThreadLauncher(),
    ) as manager:
        pdf_parser = PDFParser()
        mcq_generator = MCQGenerator(model_name="alcf:mistralai/Mistral-7B-Instruct-v0.3")
        mcq_selector = MCQSelector()
        mcq_answerer = MCQAnswerer(model_name="alcf:meta-llama/Meta-Llama-3-70B-Instruct")
        answer_scorer = AnswerScorer(model_a_name='alcf:mistralai/Mistral-7B-Instruct-v0.3', 
                                     model_b_name='alcf:meta-llama/Meta-Llama-3-70B-Instruct')

        parser_agent = manager.launch(pdf_parser)
        mcq_gen_agent = manager.launch(mcq_generator)
        mcq_selector_agent = manager.launch(mcq_selector)
        mcq_answerer_agent = manager.launch(mcq_answerer)
        answer_scorer_agent = manager.launch(answer_scorer)

        future: Future[int] = parser_agent.action('parse_pdf', "example1.pdf")
        parsed_output = future.result()

        logger.info('Parser Agent parsed json: %s', future.result())

        future: Future[int] = mcq_gen_agent.action('generate_mcqs', parsed_output)
        mcq_output = future.result()

        # mcq_output = "mcqs_example1.json"

        logger.info('MCQ Agent generated MCQs: %s', mcq_output)

        future: Future[int] = mcq_selector_agent.action('select_mcqs', mcq_output, "selected_mcqs.json", n=3)

        selected_mcqs = future.result()
        logger.info('Selected MCQs: %s', selected_mcqs)


        future: Future[int] = mcq_answerer_agent.action('answer_mcqs', selected_mcqs)

        answered_questions = future.result()

        logger.info('Answered questions: %s', answered_questions)


        future: Future[int] = answer_scorer_agent.action('score_answers')

        scored_answers = future.result()

        logger.info('Scored questions: %s', scored_answers)


    return 0




if __name__ == '__main__':
    raise SystemExit(main())
