# Code for MCQ-based evaluation, etc.

Here we describe Python programs for:
* Generating and evaluating MCQs
* Fine-tuning models based on supplied data
* Other useful things

Please email stevens@anl.gov and foster@uchicago.edu if you see things that are unclear or missing.

## Code for generating and evaluating MCQs

Programs to run PDFs &rarr; JSON &rarr; LLM-generated MCQs &rarr; LLM-generated answers &rarr; LLM-scored answers

Details on programs follow. Use `-h` to learn about other options.
```
# 1) Extract text from PDFs to create JSON files
python simple_parse.py -i <PDF-directory> -o <JSON-directory>

# 1a) Or: Extract text from PDFs with AdaParse for higher quality
#     See https://github.com/7shoe/AdaParse/tree/main

# 2) Use specified LLM to generate MCQs for papers, after dividing paper text into chunks
#    and augmenting each chunk with extra info
python generate_mcqs.py -i <JSON-directory> -o <JSON-file> -m <model>

# 2a) Next is useful if you run `generate_mcqs.py` multiple times and thus have multiple JSON files
python combine_json_files.py -i <JSON-directory> -o <JSON-file>

# 3) Select subset of MCQs from output of step 2, for subsequent use
python select_mcqs_at_random.py -i <JSON-file> -o <JSON-file> -n <N>

# 4) Use specified LLM to generate answers to MCQs generated in step 2
#    Read MCQs in <input-json>
#    Place results in "<result-directory>/answers_<model-A>.json"
python generate_answers.py -i <input-json> -o <result-directory> -m <model>

# 5) Use specified LLM to score answers to MCQs generated in step 4
#    Look for file "answers_<model-A>.json" in <result-directory>
#    Produce file "scores_<model-A>_<model-B>.json"
python score_answers.py -o <result-directory> -a <model-A> -b <model-B>
```
Note:
* You need a file `openai_access_token.txt` that contains your OpenAI access token if you are to use `gpt-4o`.

## Code for fine-tuning programs
```
# LORA fine-tuning
python lora_fine_tune.py -i <json-file> -o <model-directory>

# Full fine tune
python full_fine_tune.py -i <json-file> -o <model-directory>
```
Note:
* You need a file `hf_access_token.txt` if you want to publish models to HuggingFace.
* You need to edit the file to specify where to publish models in HuggingFace

## Code for other useful things

Determine what models are currently running on ALCF inference service (see below for more info)
```
python check_alcf_service_status.py
```
Determine what answers have been generated and scored, and what additional runs could be performed, _given running models_, to generate and score additional answers. (You may want to submit runs to start models. Use `-m` flag to see what could be useful to submit.) 
```
python review_status.py -o <result-directory>
```
Perform runs of `generate_answers` and `grade_answers.py` to generate missing outputs. (See below for more info)
```
python run_missing_generates.py -o <result-directory>
```

### More on `check_alcf_service_status.py` 

The program `check_alcf_service_status.py` retrieves and processes status information from the [ALCF Inference service](https://github.com/argonne-lcf/inference-endpoints), and lists models currently running or queued to run. E.g., as follows, which shows six models running and one queued. Models that are not accessed for some period are shut down and queued models started. A request to a model that is not running adds it to the queue.
```
% python check_alcf_service_status.py
Running: ['meta-llama/Meta-Llama-3-70B-Instruct', 'meta-llama/Meta-Llama-3-8B-Instruct', 'mistralai/Mistral-7B-Instruct-v0.3']
Starting: ['N/A']
Queued : []
```
Note:
* You need a valid ALCF access token stored in a file `alcf_access_token.txt`.  See [how to generate an ALCF access token](https://github.com/argonne-lcf/inference-endpoints?tab=readme-ov-file#authentication).
* Here is a list of [models supported by the ALCF inference service](https://github.com/argonne-lcf/inference-endpoints?tab=readme-ov-file#-available-models).
* "N/A" is a test model used by ALCF, it can be ignored.

### More on `run_missing_generates.py`

The ALCF inference service hosts many models, as [listed here](https://github.com/argonne-lcf/inference-endpoints?tab=readme-ov-file#-available-models). At any one time, zero or more *running*, zero or more are *queued*, and the rest are neither running not queued. (See below for how to use `check_alcf_service_status.py` to determine which.)
You may want to run against all available models. To do so, you can specify `-a all`, which works out what commands are needed to process specified MCQs with all *running models*. Adding `-q` also considers *queued models*, and `-s` *non-running models*. For example, when I ran the following command I was informed of the commands to run three models for which results are not found:
```
% python run_missing_generates.py -i 100-papers-qa.json -o output_files -a all -m 100 -s
python generate_and_grade_answers.py -i 100-papers-qa.json -o outputs -a 'Qwen/Qwen2-VL-72B-Instruct' -b 'gpt-4o' -c -q -s 0 -e 100
python generate_and_grade_answers.py -i 100-papers-qa.json -o outputs -a 'deepseek-ai/DeepSeek-V3' -b 'gpt-4o' -c -q -s 0 -e 100
python generate_and_grade_answers.py -i 100-papers-qa.json -o outputs -a 'mgoin/Nemotron-4-340B-Instruct-hf' -b 'gpt-4o' -c -q -s 0 -e 100
```

`run_missing_generates.py` has options as follows:

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




