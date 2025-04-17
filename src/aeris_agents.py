from __future__ import annotations

import os
import sys
import json
import time
import uuid
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

    def __init__(self):
        self.agent_name = "PDFParser-" + str(uuid.uuid4())[:8]

    def on_setup(self) -> None:
        self.count = 0

    @action
    def parse_pdf(self, file_path) -> str:
        task_id = str(uuid.uuid4())[:8]
        logger.info(f"TIMEINFO: {self.agent_name} agent start - task_id {task_id}")

        basename, _ = os.path.splitext(file_path)
        out_file = basename + ".json"
        out_path  = os.path.join("", out_file)
        if os.path.isfile(out_path):
            logger.warning(f'Already exists: {out_path}')
            return 0

        logger.info(f"Processing file: {file_path}")

        (text, parser) = extract_text_from_pdf(file_path)

        json_structure = [{'path': file_path, 'text': text, 'parser':parser}]


        logger.info(f"TIMEINFO: {self.agent_name} agent end - task_id {task_id}")

        return json_structure
        # with open(out_path, 'w', encoding='utf-8') as out_f:
        #     json.dump(json_structure, out_f, ensure_ascii=False, indent=2)
        # return out_file


class MCQGenerator(Behavior):
    model: Model
    all_prompt_answer_pairs: list

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.agent_name = "MCQGenerator-" + str(uuid.uuid4())[:8]

    def on_setup(self) -> None:
        self.model = Model(self.model_name)
        self.all_prompt_answer_pairs = []

    @action
    def generate_mcqs(self, parsed_input, file_path) -> str:
        task_id = str(uuid.uuid4())[:8]
        logger.info(f"TIMEINFO: {self.agent_name}-{self.model_name} agent start - task_id {task_id}")

        CHUNK_SIZE = 1000
        num_chunks = 0
        # with open(file_path, 'r', encoding='utf-8') as file:
        #     if file_path.lower().endswith(".json"):
        #         json_str = file.read()
        #         lines = [json_str]
        #     else:
        #         lines = file.readlines()

        # for record, key in enumerate(parsed_input, start=1):
        count = 0
        # print(parsed_input)
        for record in parsed_input:
            # print(record)
            count += 1
            # print(record)
            # try:
            #     record = json.loads(line.strip())
            # except json.JSONDecodeError as e:
            #     logger.info(f"JSON decode error in file {file_path} line {j}: {e}")

            text = record['text']
            path = record['path']
            chunks = split_text_into_chunks(text, CHUNK_SIZE)
            num_chunks += len(chunks)

            prompt_answer_pairs = generate_mcqs(self.model, path, file_path, count, chunks, NoOpTqdm())
            #prompt_answer_pairs = {'hello': 'test'}
            self.all_prompt_answer_pairs.extend(prompt_answer_pairs)

        logger.info(f"TIMEINFO: {self.agent_name}-{self.model_name} agent end - task_id {task_id}")

        return self.all_prompt_answer_pairs
        # out_file = f'mcqs_{file_path}'
        # logger.info(f"Writing output to {out_file}")
        # with open(out_file, 'w', encoding='utf-8') as out_f:
        #     json.dump(self.all_prompt_answer_pairs, out_f, ensure_ascii=False, indent=2)

        # return out_file




class MCQSelector(Behavior):
    
    def __init__(self):
        self.agent_name = "MCQSelector-" + str(uuid.uuid4())[:8]

    @action
    def select_mcqs(self, data, n=3) -> str:
        task_id = str(uuid.uuid4())[:8]
        logger.info(f"TIMEINFO: {self.agent_name} agent start - task_id {task_id}")

        # with open(input_file, 'r', encoding='utf-8') as infile:
        #     data = json.load(infile)

        # Ensure there are enough entries to sample
        if n > len(data):
            raise ValueError(f"Requested {n} entries, but the file only contains {len(data)} entries.")

        # Randomly sample N entries without replacement
        selected_entries = random.sample(data, n)

        # # Write the sampled entries to the output file
        # with open(output_file, 'w', encoding='utf-8') as outfile:
        #     json.dump(selected_entries, outfile, indent=4, ensure_ascii=False)

        # logger.info(f"Selected {n} random entries written to {output_file}")

        # return output_file
        logger.info(f"TIMEINFO: {self.agent_name} agent end - task_id {task_id}")
        return selected_entries
    


class MCQAnswerer(Behavior):
    model: Model

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.agent_name = "MCQAnswerer-" + str(uuid.uuid4())[:8]

    def on_setup(self) -> None:
        self.model = Model(self.model_name)

    @action
    def answer_mcqs(self, data, start_num=0) -> str:
        task_id = str(uuid.uuid4())[:8]
        # Load question-answer pairs
        # try:
        #     with open(input_file, "r", encoding="utf-8") as f:
        #         data = json.load(f)
        # except:
        #     config.logger.error(f"ERROR: file {input_file} not found.")
        #     sys.exit(0)

        logger.info(f"TIMEINFO: {self.agent_name}-{self.model_name} agent start - task_id {task_id}")

        qa_pairs = []

        start_time = time.time()
        total_time = 0

        data = data[start_num:]

        config.logger.info(f'Generating {len(data)} answers with model {self.model_name}')

        pbar = NoOpTqdm(total=len(data))

        # Use JSON Lines file format, append each result immediately rather
        # # than waiting to write everything at the end
        # # Determine the output file name (changed extension to .jsonl).
        # if int(start_num) == 0:
        #     output_file = f'answers_{self.model_name.replace("/","+")}.json'
        # # else:
        # #     output_file = f'answers_{model_name.replace("/","+")}_{start_num}_{args.end}.jsonl'
        # output_path = os.path.join('', output_file)

        # # Remove existing output file if present; start fresh
        # if os.path.exists(output_path):
        #     os.remove(output_path)

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
                # config.logger.info(f'{index} ({avg_time:.2f} s)', end =' ', flush=True) 

            new_tuple = {'file':filename, 'filenum':filenum, 'chunknum':chunknum,
                        'gen_time': f'{gen_time:.3f}',
                        'question':question, 'reference': reference_answer,
                        'model': model_answer}
            qa_pairs.append(new_tuple)


            # Append the new result immediately to the output file.

            # with open(output_path, 'a', encoding='utf-8') as out_f:
            #     out_f.write(json.dumps(new_tuple, ensure_ascii=False) + "\n")
        
        logger.info(f"TIMEINFO: {self.agent_name}-{self.model_name} agent end - task_id {task_id}")

        return qa_pairs
        # return output_path



class AnswerScorer(Behavior):
    modela: Model
    modelb: Model

    def __init__(self, model_a_name: str, model_b_name: str) -> None:
        self.model_a_name = model_a_name
        self.model_b_name = model_b_name

    def on_setup(self) -> None:
        self.modela = Model(self.model_a_name)
        self.modelb = Model(self.model_b_name)
        self.agent_name = "AnswerScorer-" + str(uuid.uuid4())[:8]

    @action
    def score_answers(self, data) -> str:
        task_id = str(uuid.uuid4())[:8]
        logger.info(f"TIMEINFO: {self.agent_name}-{self.model_a_name}-{self.model_b_name} agent start - task_id {task_id}")

         # Load previously generated answers from modelA
        # answer_file = 'answers_'+self.model_a_name.replace('/', '+')+'.json'
        # print(f'Looking for {answer_file}')
        # if not os.path.exists(answer_file):
        #     print(f'No answers file for {self.model_a_name}')
        #     return ""

        # score_file = f'scores_{self.model_a_name.replace("/","+")}={self.model_b_name.replace("/","+")}.json'
        # # if os.path.exists(score_file) and not args.force:
        # #     print('Score file already exists:', score_file)
        # #     exit(1)

        # logger.info(f"Writing scores to {score_file}")
        # out_f = open(score_file, 'w', encoding='utf-8') 

        # logger.info(f"Loading question-answer pairs from {answer_file}")
        # # Load question-answer pairs
        # with open(answer_file, "r", encoding="utf-8") as f:
        #     data = [json.loads(line) for line in f]
        #     # data = json.load(f)

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

        # json.dump(qa_pairs, out_f, ensure_ascii=False, indent=2)

        if scores:
            mean_score = statistics.mean(scores)
            variance_score = statistics.pvariance(scores)  # population variance
        else:
            print("No valid QA pairs found or no scores computed.")

        logger.info(f"TIMEINFO: {self.agent_name}-{self.model_a_name}-{self.model_b_name} agent end - task_id {task_id}")

        return scores, qa_pairs


class Coordinator(Behavior):


    def __init__(
        self,
        generator_a: Handle[MCQGenerator],
        generator_b: Handle[MCQGenerator],
        selector: Handle[MCQSelector],
        answerer_a: Handle[MCQAnswerer],
        answerer_b: Handle[MCQAnswerer],
        scorer_a: Handle[AnswerScorer],
        scorer_b: Handle[AnswerScorer],
    ) -> None:
        self.generator_a = generator_a
        self.generator_b = generator_b
        self.selector = selector
        self.answerer_a = answerer_a
        self.answerer_b = answerer_b
        self.scorer_a = scorer_a
        self.scorer_b = scorer_b

        self.agent_name = "Coordinator-" + str(uuid.uuid4())[:8]

    @action
    def process(self, parsed_output: str, file_path: str) -> str:
        """
        Generate a set of MCQs from the parsed text. Select a subset of MCQs and score the answers to them.
        Use the best model to answer all MCQs and return the result
        """
        task_id = str(uuid.uuid4())[:8]
        logger.info(f"{self.agent_name} agent start - task_id {task_id}")

        import random
        use_model_a = random.choice([True, False])

        if use_model_a:
            mcq_output = self.generator_a.action('generate_mcqs', parsed_output, file_path).result()
        else:
            mcq_output = self.generator_b.action('generate_mcqs', parsed_output, file_path).result()
        # logger.info('MCQ Agent generated MCQs: %s', mcq_output)

        selected_mcqs = self.selector.action('select_mcqs', mcq_output, n=3).result()
        logger.info('Selected MCQs: %s', selected_mcqs)

        answered_questions_a_fut = self.answerer_a.action('answer_mcqs', selected_mcqs)
        answered_questions_b_fut = self.answerer_b.action('answer_mcqs', selected_mcqs)
        
        # logger.info('Answered questions A: %s', answered_questions_a)
        # logger.info('Answered questions B: %s', answered_questions_b)

        fut_a = self.scorer_b.action('score_answers', answered_questions_a_fut.result())
        fut_b = self.scorer_a.action('score_answers', answered_questions_b_fut.result())

        scored_answers_a, scored_output_a = fut_a.result()
        scored_answers_b, scored_output_b = fut_b.result()

        logger.info('Scored questions: A: %s B: %s', scored_answers_a, scored_answers_b)

        avg_score_a = sum(scored_answers_a) / len(scored_answers_a)
        avg_score_b = sum(scored_answers_b) / len(scored_answers_b)

        # Now use the best model to answer all of the generated MCQs
        selected_mcqs = self.selector.action('select_mcqs', mcq_output, n=len(mcq_output)).result()

        if avg_score_a >= avg_score_b:
            logger.info("Generating answers with A")
            answered_questions = self.answerer_a.action('answer_mcqs', selected_mcqs).result()
        else:
            logger.info("Generating answers with B")
            answered_questions = self.answerer_b.action('answer_mcqs', selected_mcqs).result()

        logger.info(f"{self.agent_name} agent end - task_id {task_id}")
        return answered_questions


def main() -> int:
    init_logging(logging.INFO)

    model_a_name = 'alcf:mistralai/Mistral-7B-Instruct-v0.3'
    model_b_name = 'alcf:meta-llama/Meta-Llama-3-70B-Instruct'

    with Manager(
        exchange=ThreadExchange(),
        launcher=ThreadLauncher(),
    ) as manager:
        pdf_parser = PDFParser()
        mcq_generator_a = MCQGenerator(model_name=model_a_name)
        mcq_generator_b = MCQGenerator(model_name=model_b_name)
        mcq_selector = MCQSelector()
        mcq_answerer_a = MCQAnswerer(model_name=model_a_name)
        mcq_answerer_b = MCQAnswerer(model_name=model_b_name)
        answer_scorer_a = AnswerScorer(model_a_name=model_a_name, 
                                     model_b_name=model_b_name)
        answer_scorer_b = AnswerScorer(model_a_name=model_b_name, 
                                     model_b_name=model_a_name)

        parser_agent = manager.launch(pdf_parser)
        mcq_gen_a_agent = manager.launch(mcq_generator_a)
        mcq_gen_b_agent = manager.launch(mcq_generator_b)
        mcq_selector_agent = manager.launch(mcq_selector)
        mcq_answerer_a_agent = manager.launch(mcq_answerer_a)
        mcq_answerer_b_agent = manager.launch(mcq_answerer_b)
        answer_scorer_a_agent = manager.launch(answer_scorer_a)
        answer_scorer_b_agent = manager.launch(answer_scorer_b)

        coordinator = manager.launch(Coordinator(mcq_gen_a_agent, mcq_gen_b_agent, 
                                                 mcq_selector_agent, mcq_answerer_a_agent, mcq_answerer_b_agent, answer_scorer_a_agent, answer_scorer_b_agent))

        results = []
        for x in range(1,11):
            filename = f"example{x}.pdf"
            parsed_output_fut = parser_agent.action('parse_pdf', filename)
            results.append(coordinator.action('process', parsed_output_fut.result(), filename))

        for r in results:
            logger.info('Scored questions: %s', r.result())

    return 0

if __name__ == '__main__':
    raise SystemExit(main())
