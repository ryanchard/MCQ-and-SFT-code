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
output_dir   = sys.argv[2]  # e.g. "llama-3.1-8B-merged-sft-rick"

###############################################################################
# 2. Log in to Hugging Face (optional - only needed if you push to hub)
###############################################################################
access_token = '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'  # your HF token here
login(access_token)

max_seq_length = 2048  # For example, supports RoPE scaling internally

###############################################################################
# 3. Load the dataset from the user-specified file
###############################################################################
dataset = load_dataset("json", data_files=dataset_file, split="train")
num_rows = dataset.num_rows
print(f"Number of rows: {num_rows}")

num_steps = num_rows % 4  # Modify as desired

###############################################################################
# 4. Configure 4-bit quantization
###############################################################################
config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

###############################################################################
# 5. Define the model and tokenizer
###############################################################################
model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the base model with quantization and device mapping
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=config,
    device_map="auto",
)

# Fix for infinite generation issue by adding and setting a pad token
tokenizer.add_special_tokens({"pad_token": "<|reserved_special_token_0|>"})
base_model.config.pad_token_id = tokenizer.pad_token_id
tokenizer.padding_side = 'right'

# Prepare for k-bit LoRA training
base_model = prepare_model_for_kbit_training(base_model)

###############################################################################
# 6. Create the PEFT LoRA configuration and wrap the base model
###############################################################################
lora_config = LoraConfig(
    r=16,
    target_modules=["q_proj", "v_proj"],
    task_type=TaskType.CAUSAL_LM,
    lora_alpha=32,
    lora_dropout=0.05
)

peft_model = get_peft_model(base_model, lora_config)

###############################################################################
# 7. Create and run the SFTTrainer
###############################################################################
trainer = SFTTrainer(
    model=peft_model,
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

###############################################################################
# 8. Save the LoRA adapter and tokenizer to the user-specified output directory
###############################################################################
peft_model.save_pretrained(output_dir, save_adapter=True, save_config=True)
tokenizer.save_pretrained(output_dir)

###############################################################################
# 9. Merge LoRA weights with the base model and save the final merged model
###############################################################################
model_to_merge = peft_model.from_pretrained(
    AutoModelForCausalLM.from_pretrained(model_name).to("cuda"),
    output_dir
)

merged_model = model_to_merge.merge_and_unload()
merged_model.save_pretrained(output_dir, save_method="merged_16bit")
tokenizer.save_pretrained(output_dir)

###############################################################################
# 10. (Optional) Push merged model and tokenizer to Hugging Face Hub
###############################################################################
# Adjust the repo name "Stevens/..." or remove these calls if you do not want to push
merged_model.push_to_hub("ianfoster/" + output_dir)
tokenizer.push_to_hub("ianfoster/" + output_dir)
