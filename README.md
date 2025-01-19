# Code for generating QA pairs and MCQs from PDFs, etc.

## Administrative aids

The program `check_alcf_service_status.py` retrieves and processes status information from the [ALCF Inference service](https://github.com/argonne-lcf/inference-endpoints), and lists models currently running or queued to run. E.g., as follows, which shows six models running and one queued. Models that are not accessed for some period are shut down and queued models started. A request to a model that is not running adds it to the queue.

```
% python check_alcf_service_status.py
Running: ['meta-llama/Meta-Llama-3-70B-Instruct', 'meta-llama/Meta-Llama-3-8B-Instruct', 'mistralai/Mistral-7B-Instruct-v0.3']
Starting: ['N/A']
Queued : []
```

Note:
* You will need a valid ALCF access token stored in a file `alcf_access_token.txt`.  [How to generate an ALCF access token](https://github.com/argonne-lcf/inference-endpoints?tab=readme-ov-file#authentication).
* [The models supported by the ALCF inference service](https://github.com/argonne-lcf/inference-endpoints?tab=readme-ov-file#-available-models).
* "N/A" is a test model used by ALCF, it can be ignored.

## Generate QA pairs and MCQs from PDFs

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
* You need a file `openai_access_token.txt` that contains your OpenAI access token
