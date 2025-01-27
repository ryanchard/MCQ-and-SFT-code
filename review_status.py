import os
import requests
import glob
import subprocess

from inference_auth_token import get_access_token
alcf_access_token = get_access_token()
from alcf_inference_utilities import get_names_of_alcf_chat_models, get_alcf_inference_service_model_queues
alcf_chat_models = get_names_of_alcf_chat_models(alcf_access_token)


def extract_model_a_from_scores_file_name(folder, file):
    part1 = file.split(f'{folder}/scores_')[1]
    part2 = part1.split(':')[0]
    return part2.replace('+', '/')


def extract_model_a_from_answers_file_name(folder, file):
    part1 = file.split(f'{folder}/answers_')[1]
    part2 = part1.split(':')[0]
    return part2.replace('+', '/')


def extract_model_b_from_scores_file_name(folder, file):
    part1 = file.split(f'{folder}/scores_')[1]
    part2 = part1.split(':')[1]
    part3 = part2.split('.json')[0]
    return part3.replace('+', '/')


def generate_scores_file_name(folder, model_a, model_b):
    return f'{folder}/scores_{model_a.replace("/","+")}:{model_b.replace("/","+")}.json'


"""
We want to run potentially many different modelAs and evaluate with many different modelBs.

If a modelA has already been run once and scored with one modelB, it can be rescored with another.

"""
def main():
    import argparse
    parser = argparse.ArgumentParser(description='Program to run LLM B to rate answers provided by LLM A')
    parser.add_argument('-i','--inputs', help='MCQ file', required=True)
    parser.add_argument('-o','--outputdir', help='Directory to look for run results', required=True)
    parser.add_argument('-s','--silent', help='Just show things to run', action='store_true')
    parser.add_argument('-x','--execute', help='Execute commands', action='store_true')
    parser.add_argument('-m','--more', help='Also look at non-running/queued models', action='store_true')
    args = parser.parse_args()

    # Folder containing the output files
    inputs = args.inputs
    folder = args.outputdir
    silent = args.silent
    execute= args.execute
    other  = args.more 

    running_model_list, queued_model_list = get_alcf_inference_service_model_queues(alcf_access_token)
    running_model_list = [model for model in running_model_list if model in alcf_chat_models]
    running_model_list = [model for model in running_model_list if 'batch' not in model]
    running_model_list = [model for model in running_model_list if 'auroragpt-0.1-chkpt' not in model]
    running_model_list = [model for model in running_model_list if model != 'N/A'] + ['gpt-4o']

    # List available generated answers
    answers_files = [file.replace('.json', '') for file in glob.glob(f'{folder}/answers_*')]
    if not silent:
        print(f'\nReviewing answers and scores files in {folder}.')
        print(f'\nFocused on currently running models, minus mgoin/Nemotron-4-340B-Instruct-hf (slow) and auroragpt-0.1-chkpt-* (buggy):\n')

    models_scored = {}

    if not silent:
        print(f'====== Models running, queued, available at ALCF inference service ====')
        print(f'Running models: {running_model_list}')
        print(f'Queued models : {queued_model_list}')
        other_models = [model for model in alcf_chat_models if model not in running_model_list and model not in queued_model_list]
        print(f'Other models : {other_models}')

        # List for each set of answers which models have reviewed it
        print(f'\n====== Answers and scores obtained to date for {folder} ========')
        for file in answers_files:
            model_a = file.split("answers_")[1].replace('+','/')
            print(f'{model_a}')
            score_files = glob.glob(f'{folder}/scores_{model_a.replace("/","+")}:*')
            m_list = []
            for score_file in score_files:
                f = score_file.split(f'{folder}/scores_{model_a.replace("/","+")}:')[1]
                model_b = f.split("_")[0].replace('+','/').replace('.json','')
                print(f'\t{model_b}')
                m_list.append(model_b)
            models_scored[model_a] = m_list

    # List running models that have not generated answers
    no_answer_list = [f'python generate_answers.py -o {folder} -i {inputs} -m {model_a}' for model_a in running_model_list if not os.path.isfile(f'{folder}/answers_{model_a.replace("/","+")}.json')]
    if no_answer_list != []:
        if not silent: print('\n====== Generating answers for running models without them ======')
        for command in no_answer_list:
            if execute:
                print(f'\nExecuting {command}')
                try:
                    subprocess.run(command, shell=True)
                except OSError as e:
                    print(f'    Error {e}')
                    return -1
            else:
                print(f'\n{command}')

    # List for each possible reviewer (i.e., a running model) which answers it has not reviewed
    if not silent: print('\n====== Score answers with any un-applied running model ======')
    for model_b in running_model_list:
        for filename in answers_files:
            model_a = extract_model_a_from_answers_file_name(folder, filename)
            if not os.path.isfile(generate_scores_file_name(folder, model_a, model_b)):
                score_filename = generate_scores_file_name(folder, model_a, model_b)
                command = f'python score_answers.py -o {folder} -a {model_a} -b {model_b}'
                if execute:
                    print(f'\nExecuting {command}')
                    try:
                        subprocess.run(command, shell=True)
                    except OSError as e:
                        print(f'    Error {e}')
                        return -1
                else:
                    print(f'\n{command}')


    if not silent: print()

    if other:
       print('\n====== Non-running/queued models ======')
       for model_a in other_models:
           if not os.path.isfile(f'{folder}/answers_{model_a.replace("/","+")}.json'):
               print(f'\npython generate_answers.py -o {folder} -i {folder}.json -m {model_a}')

if __name__ == "__main__":
    main()
