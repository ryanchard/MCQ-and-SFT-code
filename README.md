## (this page under construction)

# Code for MCQ-based evaluation, etc.

Here we describe Python programs for:
* Generating and evaluating MCQs
* Fine-tuning models based on supplied data
* Other useful things

Please email foster@anl.gov and stevens@anl.gov if you see things that are unclear or missing.

## Set up to access ALCF Inference Service 

**Before you start:** We recommend you follow the instructions for 
[ALCF Inference Service Prerequisites](https://github.com/argonne-lcf/inference-endpoints?tab=readme-ov-file#%EF%B8%8F-prerequisites)
to set up your ALCF auth token, required to access models via the inference service.
(You need to download and run `inference_auth_token.py`.)

## Code for generating and evaluating MCQs

### Workflow Overview
This pipeline converts scientific papers in **PDF format** into JSON and then uses AI models
of your choice to generate **multiple-choice questions (MCQs)**, **answers**,
and **scores** of those answers.

**Workflow Steps:**
1. Convert PDFs to JSON representations.
2. Generate MCQs from JSON representations.
3. Combine multiple MCQ JSON files (if needed).
4. Select a subset of MCQs.
5. Generate additonal  answers for MCQs (using a different model than
used to generate the initial MCQs and answers).
6. Score AI-generated answers using another AI model.
7. Review the status of MCQ generation and scoring.

---

### 1. Set Up Your Working Directory
Ensure your working directory has subdirectories for storing input and output files.

- `myPDFdir/` → Stores **original PDF papers**.
- `myJSONdir/` → Stores **parsed text in JSON format**.
- `myJSON-MCQdir/` → Stores **generated MCQs in JSON format**.
- `myRESULTSdir/` → Stores **AI-generated answers and scores**.

If you're just starting (and don't already have these or equivalent directories),
, create these directories manually. If yours are named differently, substitute your
directory names as you follow the instruction sequence below::
```bash
mkdir myPDFdir myJSONdir myJSON-MCQdir myRESULTSdir
```
(**Note:** Some of the scripts below create their output directories automatically if they don’t
already exist, but we will create them just to be sure..)

At this stage you'll want to place some papers in PDF form into **myPDFdir**.

---

### 2. Set Up and Activate Your Conda Environment
If you already set up a Conda environment, update it with the latest dependencies:
```bash
conda env update --name <your_conda_env> --file environment.yml
```
Otherwise, create a new Conda environment:
```bash
conda env create -f environment.yml
conda activate globus_env
```
(**Note:** If you get `CondaValueError: prefix already exists`, edit`environment.yml` and change the `name:`,
ten activate that env after creating it.)

---

### 3. Convert PDFs to JSON
Extract text from PDFs using a simple parser:
```bash
python simple_parse.py -i myPDFdir -o myJSONdir
```
Alternatively, you can use **AdaParse** (higher-quality parser, still in testing). 
[More details](https://github.com/7shoe/AdaParse/tree/main)

---

### 4. Generate MCQs Using an AI Model
To generate MCQs from parsed JSON files:

1. **Authenticate with ALCF inference service (if not already done):**
   ```bash
   wget https://raw.githubusercontent.com/argonne-lcf/inference-endpoints/refs/heads/main/inference_auth_token.py
   python inference_auth_token.py authenticate
   ```
2. **(Optional) Check which models are running**
You may wish to check to see which models are currently running as waiting for a model to load can
take 10-15 minutes (see 
[ALCF Inference service](https://github.com/argonne-lcf/inference-endpoints)). Get the list of running
and queued models as follows:
   ```bash
   access_token=$(python inference_auth_token.py get_access_token)
   curl -X GET "https://data-portal-dev.cels.anl.gov/resource_server/sophia/jobs" \
       -H "Authorization: Bearer ${access_token}"
   ```

3. **Run MCQ generation:**
You may wish to check to see which models are currently running as waiting for a model to load can
take 10-15 minutes (see 
[ALCF Inference service](https://github.com/argonne-lcf/inference-endpoints)).
For this example
we are using `Mistral-7B-Instruct-v0.3`.


   ```bash
   python generate_mcqs.py -i myJSONdir \
        -o myJSON-MCQdir \
        -m 'alcf:mistralai/Mistral-7B-Instruct-v0.3'
   ```
   - This script divides text into **chunks**, **generates MCQs**, and **includes reference answers**.

4. **(Optional) Combine multiple MCQ JSON files into a single file:**
   ```bash
   python combine_json_files.py -i myJSON-MCQdir -o MCQ-JSON-file
   ```

---

### 5. Select a Subset of MCQs for Further Processing
If you want to randomly select a subset of MCQs from the generated JSON files, use 
`select_mcqs_at_random.py`, specifying the number of MCQs to select.  For example, to select
17 MCQs::
```bash
python select_mcqs_at_random.py -i MCQ-JSON-file -o Selected-MCQs.json -n 17
```

---

### 6. Generate Answers for MCQs Using a Different Model
This step uses an AI model to generate **new answers** for the selected MCQs. We will
use a differnet model than above here. Note the form for specifying the model is 
`<locn>:<model>` and in this example we will use `meta-llama/Meta-Llama-3-70B-Instruct`,
whose endpoint is running at <locn> = `alcf`..

```bash
python generate_answers.py -i Selected-MCQs.json \
       -o myRESULTSdir \
       -m 'alcf:meta-llama/Meta-Llama-3-70B-Instruct'
```
- **Input:** `Selected-MCQs.json` (or `MCQ-JSON-file` if no subset was chosen).
- **Output:** `myRESULTSdir/answers_<model>.json` (AI-generated answers).

---

### 7. Score AI-Generated Answers
An AI model evaluates and scores the generated answers against reference answers. Here we
will use
`alcf:mistralai/Mistral-7B-Instruct-v0.3`
to evaluate the answers we created in the previous step with
`alcf:meta-llama/Meta-Llama-3-70B-Instruct`

```bash
python score_answers.py -o myRESULTSdir \
       -a 'alcf:meta-llama/Meta-Llama-3-70B-Instruct' \
       -b 'alcf:mistralai/Mistral-7B-Instruct-v0.3'
```
- **Input:** `myRESULTSdir/answers_<model-A>.json`
- **Output:** `myRESULTSdir/scores_<locn-A>:<model-A>_<locn-B>:<model-B>.json`
- **Note:** Any `/` in model names is replaced with `+` in filenames.

---

### 8. Review MCQ Generation and Scoring Status
To check progress and see which MCQs are answered/scored:
```bash
python review_status.py -i MCQ-JSON-file -o myRESULTSdir
```
- This script identifies missing or incomplete processing steps.

---

## Final Notes
- This pipeline ensures **high-quality multiple-choice questions** are generated and scored using AI.
- The steps allow for **comparison of AI-generated answers against reference answers**.
- The scoring step provides a **numerical evaluation (1-10)** of answer accuracy.

**CeC edits stop here**

**Note:**
* You need a file *openai_access_token.txt* that contains your OpenAI access token if you
are to use an OpenAI model like *gpt-4o*.

Examples of running *generate_answers.py*:
* `python generate_answers.py -o RESULTS -i MCQs.json -m openai:o1-mini.json`
  * Uses the OpenAI model `o1-mini` to generate answers for MCQs in `MCQs.json` and stores results in the `RESULTS` directory, in a file named `answers_openai:o1-mini.json`
* `python generate_answers.py -o RESULTS -i MCQs.json -m "pb:argonne-private/AuroraGPT-IT-v4-0125`
  * Uses the Huggingface model `argonne-private/AuroraGPT-IT-v4-0125`, running on a Polaris compute node started via PBS, to generate answers for the same MCQs. Results are placed in `RESULTS/answers_pb:argonne-private+AuroraGPT-IT-v4-0125.json`
 
Examples of running `score_answers.py`:
* `python score_answers.py -o RESULTS -i MCQs.json -a openai:o1-mini.json -b openai:gpt-4o`
  * Uses the OpenAI model `gpt-4o` to score answers for MCQs in `MCQs.json` and stores results in `RESULTS` directory, in a file named `answers_openai:o1-mini.json`
* `python score_answers.py -o RESULTS -a pb:argonne-private/AuroraGPT-IT-v4-0125 -b openai:gpt-4o`
  * Uses the OpenAI model gpt-4o to score answers previously generated for model `pb:argonne-private/AuroraGPT-IT-v4-0125`, and assumed to be located in a file `RESULTS/answers_pb:argonne-private+AuroraGPT-IT-v4-0125.json`, as above. Places results in file `RESULTS/scores_pb:argonne-private+AuroraGPT-IT-v4-0125:openai:gpt-4o.json`.
 

## Notes on different model execution locations
The class `Model` (in `model_access.py`) implements init and run methods that allow for use of different models. 
```
model = Model(modelname)
response = model.run(user_prompt='Tell me something interesting')
```
where `modelname` has a prefix indicating the model type/location:
* **alcf**: Model served by the ALCF Inference Service. You need an ALCF project to charge to.
* **hf**: Huggingface model downloaded and run on Polaris login node (not normally a good thing).
* **pb**: Huggingface model downloaded and run on a Polaris compute node. You need an ALCF project to charge to.
* **vllm**: Huggingface model downloaded and run via VLLM on Polaris compute node. Not sure that works at present.
* **openai**: An OpenAI model, like gpt-4o or o1-mini. You need an OpenAI account to charge to.


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
* We are still debugging how to download and run published models

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

The program `check_alcf_service_status.py` retrieves and processes status information from the
[ALCF Inference service](https://github.com/argonne-lcf/inference-endpoints),
and lists models currently running or queued to run. E.g., as follows, which shows six
models running and one queued. Models that are not accessed for some period are shut
down and queued models started. A request to a model that is not running adds it to the queue.
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




