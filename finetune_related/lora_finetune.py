import os
import time
import torch
import json
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
import wandb

# ------------------------------
# Environment configuration
# ------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_MODE"] = "offline"  # Use offline mode to avoid sending secrets

# ------------------------------
# Unique ID for checkpoints
# ------------------------------
unique_id = str(time.strftime('%m%d%H%M%S'))

# ------------------------------
# Configurable paths (replace with your own)
# ------------------------------
model_path = os.getenv("MODEL_PATH", "/path/to/local/model")  # Path to pretrained model
json_file_path = os.getenv("DATASET_PATH", "/path/to/dataset.json")  # Path to JSON dataset

# ------------------------------
# Load model and tokenizer
# ------------------------------
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16
)

# ------------------------------
# LoRA configuration
# ------------------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ------------------------------
# Load dataset
# ------------------------------
with open(json_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Ensure data is a list
if isinstance(data, dict):
    data = [data]

dataset = Dataset.from_list(data)

# ------------------------------
# Preprocess dataset
# ------------------------------
def preprocess_function(examples):
    inputs = [f"Question: {q}\nAnswer: " for q in examples["input"]]
    model_inputs = tokenizer(inputs, truncation=True, padding="max_length", max_length=2048)
    labels = tokenizer(examples["output"], truncation=True, padding="max_length", max_length=8192)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# ------------------------------
# Training arguments
# ------------------------------
training_args = TrainingArguments(
    output_dir=f"./checkpoints_{unique_id}",
    num_train_epochs=50,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    save_steps=200,
    save_total_limit=10,
    learning_rate=1e-5,
    prediction_loss_only=True,
    fp16=True,
    logging_steps=10,
    warmup_steps=100,
    weight_decay=0.01,
    dataloader_num_workers=4,
    ddp_find_unused_parameters=False,
    report_to="wandb"
)

# ------------------------------
# Data collator
# ------------------------------
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ------------------------------
# Trainer
# ------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

# ------------------------------
# Initialize Weights & Biases (offline mode)
# ------------------------------
wandb.init(project='ds_qwen_14b_kidney', name=unique_id)

# ------------------------------
# Train the model
# ------------------------------
trainer.train()

# ------------------------------
# Save LoRA model
# ------------------------------
model.save_pretrained("./lora_model")

# Merge LoRA weights into base model
model = model.merge_and_unload()

# Save merged model
model.save_pretrained("./merged_model")
tokenizer.save_pretrained("./merged_model")
