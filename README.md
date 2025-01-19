# Code for generating QA pairs and MCQs from PDFs, etc.

## Administrative aids

The program `check_alcf_service_status.py` retrieves and processes status information from the ALCF Inference service, and lists models currently running or queued to run. E.g., as follows, which shows six models running and one queued. Models that are not accessed for some period are shut down and queued models started. A request to a model that is not running adds it to the queue.

```
% python check_alcf_service_status.py
Running: ['Qwen/Qwen2.5-7B-Instruct', 'Qwen/Qwen2.5-14B-Instruct', 'Qwen/QwQ-32B-Preview', 'mistralai/Mistral-Large-Instruct-2407', 'meta-llama/Meta-Llama-3.1-70B-Instruct', 'meta-llama/Meta-Llama-3.1-8B-Instruct']
Queued : ['mistralai/Mixtral-8x22B-Instruct-v0.1']
```

Note:
* You will need a valid ALCF access token stored in a file `alcf_access_token.txt`

## Generate QA pairs and MCQs from PDFs

The program `generate_qa_or_mc.py` calls an LLM to generate either question-answer pairs (QA) or multiple-choice questions (MCQ) from a set of PDFs. Options are as follows:
```
  -h, --help            show this help message and exit
  -i INPUTDIR, --inputdir INPUTDIR
                        Input PDF directory
  -q, --qa              Generate QA pairs rather than MCQs
  -o OUTPUT, --output OUTPUT
                        Output file for JSON (default=output_file.json)
  -c CHUNKSIZE, --chunksize CHUNKSIZE
                        Chunk size (default=500 tokens)
```
For example, the following generates MCQs for each PDF in directory PDFs.
```
python generate_qa_or_mc.py -i PDFs -o my_output.json
```

