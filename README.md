# Code for generating QA pairs and MCQs from PDFs, etc.

This repository provides access to programs as follows:
* `generate_qa_or_mc.py`: Prepare question-answer pairs (QA) or multiple-choice questions (MCQ) from PDF documents
* `generate_and_grade_answers.py`: Apply an LLM hosted on the ALCF inference service to MCQs, and another LLM to score results
* `run_missing_generates.py`: Perform runs of `generate_and_grade_answers.py` to generate missing outputs.
* `check_alcf_service_status.py`: Determine what models are currently running on ALCF inference service.

The original programs were written by Rick Stevens and then adapted for large-scale use at ALCF by Ian Foster.

## Generate QA pairs or MCQs from PDFs

The program `generate_qa_or_mc.py` calls an LLM (currently GPT4o) to generate either question-answer pairs (QA) or multiple-choice questions (MCQ) from a set of PDFs. Options are as follows:
```
% python generate_qa_or_mc.py -h
  -h, --help            show this help message and exit
  -i INPUTDIR, --inputdir INPUTDIR
                        Input PDF directory
  -q, --qa              Generate QA pairs rather than MCQs
  -o OUTPUT, --output OUTPUT
                        Output file for JSON (default=output_file)
  -c CHUNKSIZE, --chunksize CHUNKSIZE
                        Chunk size (default=1000 tokens)
```
For example, the following generates MCQs for each PDF in directory PDFs, creating files `my_output.json` for generated MCQs and `my_output.error` to list any parse errors.
```
python generate_qa_or_mc.py -i PDFs -o my_output
```
Notes:
* You need a file `openai_access_token.txt` that contains your OpenAI access token.

## `generate_and_grade_answers.py`: Generate and Grade Answers with an LLM

This program uses a specified LLM A to generate answers for a supplied set of MCQs and evaluates the generated answers with a specified LLM B. The program has options as follows:
```
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        MCQ input file
  -a MODELA, --modelA MODELA
                        modelA
  -b MODELB, --modelB MODELB
                        modelB
  -q, --quiet           Do not show details
  -c, --csv             Generate CSV summary
  -s START, --start START
                        MCQ number to start at in file (default 0)
  -e END, --end END     MCQ end number in file (default all)
  -o OUTPUT, --output OUTPUT
                        Output directory
```
For example, the following program reads MCQs from `my_inputs.json`, generates answers with model `meta-llama/Meta-Llama-3.1-405B-Instruct`, and writes outputs directory `my_outputs`. The flag `-c` asks for periodic summary CSVs and the flag `-q` means "quiet", i.e., do not print answers as generated.
```
python generate_and_grade_answers.py -i my_qa_file.json -o my_outputs -a 'meta-llama/Meta-Llama-3.1-405B-Instruct' -b 'gpt-4o' -c -q
```
As running this program can take a while, you can request to process just the MCQs from number `<start>` to number `<end>`. E.g., the following will process just the first 100:
```
python generate_and_grade_answers.py -i my_qa_file.json -o my_outputs -a 'model/A' -b 'gpt-4o' -c -q -e 100
```
Answers are stored in a file in the specified outputs directory, named `<modelA>:<modelB>_<start>_<end>.json`, with any "/" in `<modelA>` replaced with a "+." The `<start>` and `<end>` are as just described, so for example, the preceding call would generate a file `model-+A:gpt-4o_0_100.json`. 

## Generate and grade many MCQs over time

When you have to process 1000s of MCQs, it can be challenging to track what you have processed. This program examines file names in the output directory to determine what runs need to be performed and optionally runs them. E.g., the following call asks what runs need to be performed to process all MCQs up to number 200. 
```
% python run_missing_generates.py -i my_qa.json -o outputs -a 'model/A' -m 300
```
If we had already processed, as just indicated, those for numbers 0 to 100, it would output the following string to process numbers 100 to 200. Note that it always use gpt-4o as modelB. 
```
python generate_and_grade_answers.py -i my-qa.json -o outputs -a 'model/A' -b 'gpt-4o' -c -q -s 100 -e 200
python generate_and_grade_answers.py -i my-qa.json -o outputs -a 'model/A' -b 'gpt-4o' -c -q -s 200 -e 300
```
If you want it to **execute** these commands, you add a `-x`:
```
% python run_missing_generates.py -i my_qa.json -o outputs -a 'model/A' -m 300 -x
```
### Requesting to run many models with `run_missing_generates.py`

The ALCF inference service hosts many models, as [listed here](https://github.com/argonne-lcf/inference-endpoints?tab=readme-ov-file#-available-models). At any one time, zero or more *running*, zero or more are *queued*, and the rest are neither running not queued. (See below for how to use `check_alcf_service_status.py` to determine which.)
You may want to run against all available models. To do so, you can specify `-a all`, which works out what commands are needed to process specified MCQs with all *running models*. Adding `-q` also considers *queued models*, and `-s` *non-running models*. For example, when I ran the following command I was informed of the commands to run three models for which results are not found:
```
% python run_missing_generates.py -i 100-papers-qa.json -o output_files -a all -m 100 -s
python generate_and_grade_answers.py -i 100-papers-qa.json -a 'Qwen/Qwen2-VL-72B-Instruct' -b 'gpt-4o' -c -q -s 0 -e 100
python generate_and_grade_answers.py -i 100-papers-qa.json -a 'deepseek-ai/DeepSeek-V3' -b 'gpt-4o' -c -q -s 0 -e 100
python generate_and_grade_answers.py -i 100-papers-qa.json -a 'mgoin/Nemotron-4-340B-Instruct-hf' -b 'gpt-4o' -c -q -s 0 -e 100
```

### Summary of arguments for `run_missing_generates.py`
The program has options as follows:

```
  -h, --help            show this help message and exit
  -a MODELA, --modelA MODELA
                        modelA
  -o OUTPUTDIR, --outputdir OUTPUTDIR
                        Directory to look for run results
  -i INPUTFILE, --inputfile INPUTFILE
                        File to look for inputs
  -x, --execute         Run program
  -q, --queued          Process queued models
  -m MAX, --max MAX     Max to process
  -s, --start           Request to non-running models
```

## Determine which models are running on ALCF inference service 

The program `check_alcf_service_status.py` retrieves and processes status information from the [ALCF Inference service](https://github.com/argonne-lcf/inference-endpoints), and lists models currently running or queued to run. E.g., as follows, which shows six models running and one queued. Models that are not accessed for some period are shut down and queued models started. A request to a model that is not running adds it to the queue.

```
% python check_alcf_service_status.py
Running: ['meta-llama/Meta-Llama-3-70B-Instruct', 'meta-llama/Meta-Llama-3-8B-Instruct', 'mistralai/Mistral-7B-Instruct-v0.3']
Starting: ['N/A']
Queued : []
```

Note:
* You will need a valid ALCF access token stored in a file `alcf_access_token.txt`.  See [how to generate an ALCF access token](https://github.com/argonne-lcf/inference-endpoints?tab=readme-ov-file#authentication).
* Here is a list of [models supported by the ALCF inference service](https://github.com/argonne-lcf/inference-endpoints?tab=readme-ov-file#-available-models).
* "N/A" is a test model used by ALCF, it can be ignored.

