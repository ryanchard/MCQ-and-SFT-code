import sys
import torch
from trl import SFTTrainer
from transformers import TrainingArguments, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

from datasets import load_dataset
import json

from huggingface_hub import login

###############################################################################
# 1. Parse command-line arguments for dataset file and output directory/model name
###############################################################################
if len(sys.argv) < 3:
    print("Usage: python train.py <dataset_json_file> <output_model_dir>")
    sys.exit(1)

dataset_file = sys.argv[1]  # e.g. "text.json"
output_name   = sys.argv[2]  # e.g. "llama-3.1-8B-merged-sft-rick"

###############################################################################
# 2. Log in to Hugging Face (optional - only needed if you push to hub)
###############################################################################
with open('hf_access_token.txt', 'r') as file:
    hf_access_token = file.read().strip()
login(hf_access_token)

max_seq_length = 2048  # For example, supports RoPE scaling internally

###############################################################################
# 3. Load the dataset from the user-specified file
###############################################################################
print(f'Loading dataset from {dataset_file}')
dataset = load_dataset("json", data_files=dataset_file, split="train")
#dataset = load_dataset("json", data_files=dataset_file)
num_rows = dataset.num_rows
print(f"Number of rows: {num_rows}")

num_steps = num_rows  # Modify as desired (will run over 4x the data

print(f"Number of steps: {num_steps}")

###############################################################################
# 4. Configure 4-bit quantization
###############################################################################
#config = BitsAndBytesConfig(
#    load_in_4bit=True,
#    bnb_4bit_quant_type="nf4",
#    bnb_4bit_use_double_quant=True,
#    bnb_4bit_compute_dtype=torch.bfloat16,
#)

###############################################################################
# 5. Define the model and tokenizer
###############################################################################
model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the base model with quantization and device mapping
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
)

# Fix for infinite generation issue by adding and setting a pad token

print(f"Pad Token id: {tokenizer.pad_token_id} and Pad Token: {tokenizer.pad_token}")
print(f"EOS Token id: {tokenizer.eos_token_id} and EOS Token: {tokenizer.eos_token}")

tokenizer.add_special_tokens({"pad_token": "<|end_of_text|>"})

print(f"Pad Token id: {tokenizer.pad_token_id} and Pad Token: {tokenizer.pad_token}")
print(f"EOS Token id: {tokenizer.eos_token_id} and EOS Token: {tokenizer.eos_token}")

base_model.config.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = 'right'

print(f"Pad Token id: {tokenizer.pad_token_id} and Pad Token: {tokenizer.pad_token}")
print(f"EOS Token id: {tokenizer.eos_token_id} and EOS Token: {tokenizer.eos_token}")


###############################################################################
# 7. Create and run the SFTTrainer

###############################################################################

trainer = SFTTrainer(
    model=base_model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    args=TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=num_steps,
        bf16=True,
        logging_steps=1,
        output_dir="SFT-outputs",  # Intermediate checkpoints
        optim="adamw_8bit",
        seed=3407,
    ),
)

trainer.train()

# 2. IMPORTANT: Switch padding side to left
#tokenizer.padding_side = "left"

print(f"Pad Token id: {tokenizer.pad_token_id} and Pad Token: {tokenizer.pad_token}")
print(f"EOS Token id: {tokenizer.eos_token_id} and EOS Token: {tokenizer.eos_token}")

#tokenizer.pad_token = tokenizer.eos_token  ## we are using EOS for padding earlier as well.

print(f"Pad Token id: {tokenizer.pad_token_id} and Pad Token: {tokenizer.pad_token}")
print(f"EOS Token id: {tokenizer.eos_token_id} and EOS Token: {tokenizer.eos_token}")

#tokenizer.pad_token_id = tokenizer.eos_token_id  ## we are using EOS for padding earlier as well.

print(f"Pad Token id: {tokenizer.pad_token_id} and Pad Token: {tokenizer.pad_token}")
print(f"EOS Token id: {tokenizer.eos_token_id} and EOS Token: {tokenizer.eos_token}")

base_model.save_pretrained(output_name)
tokenizer.save_pretrained(output_name)

###############################################################################
# 10. (Optional) Push merged model and tokenizer to Hugging Face Hub
###############################################################################
# Adjust the repo name "ianfoster/..." or remove these calls if you do not want to push
base_model.push_to_hub("ianfoster/" + output_name)
tokenizer.push_to_hub("ianfoster/" + output_name)
