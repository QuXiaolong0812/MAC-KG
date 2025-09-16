"""
Fine-tune a pre-trained causal language model using DeepSpeed optimizer on a custom JSON dataset.
Training process and metrics are logged using Weights & Biases (WandB).
"""

import os
import time
import torch
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
from modelscope import AutoModelForCausalLM, AutoTokenizer
import deepspeed
import wandb

# Load DeepSpeed CPUAdam ops
deepspeed.ops.op_builder.CPUAdamBuilder().load()

# ------------------------------
# Environment configuration
# ------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("CUDA_DEVICES", "0,1,2,3,4,5,6,7")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ------------------------------
# Unique ID for checkpoints
# ------------------------------
unique_id = str(time.strftime('%m%d%H%M%S'))

# ------------------------------
# Model and dataset paths
# ------------------------------
model_path = os.getenv("MODEL_PATH", "/path/to/qwen2.5-7b-instruct")
dataset_path = os.getenv("DATASET_PATH", "/path/to/dataset")

# ------------------------------
# Load model and tokenizer
# ------------------------------
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# ------------------------------
# Load dataset from JSON file
# ------------------------------
dataset_file = os.path.join(dataset_path, "dhc_lung_dataset_122959.jsonl")
dataset = load_dataset('json', data_files=dataset_file)

# Split dataset into train and eval (10% eval)
split_dataset = dataset["train"].train_test_split(test_size=0.1)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]


# ------------------------------
# Preprocess dataset
# ------------------------------
def preprocess_function(examples):
    """
    Convert conversations into instruction-response format for causal LM training.
    """
    prompts = []
    for conversation in examples["conversations"]:
        human_input = next((msg["value"] for msg in conversation if msg["from"] == "human"), "")
        gpt_output = next((msg["value"] for msg in conversation if msg["from"] == "gpt"), "")
        prompt = f"Instruction: {human_input}\nAnswer: {gpt_output}"
        prompts.append(prompt)

    inputs = tokenizer(prompts, truncation=True, padding='max_length', max_length=1024)
    inputs["labels"] = inputs["input_ids"].copy()  # Labels for loss calculation
    return inputs


# Apply preprocessing to datasets
tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
tokenized_eval_dataset = eval_dataset.map(preprocess_function, batched=True)

# ------------------------------
# DeepSpeed configuration
# ------------------------------
deepspeed_config = {
    "fp16": {
        "enabled": True,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "optimizer": {
        "type": "AdamW",
        "params": {"lr": 1e-5, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.01}
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {"warmup_min_lr": 0, "warmup_max_lr": 1e-5, "warmup_num_steps": 500}
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {"device": "cpu", "pin_memory": True},
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": True
    },
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "gradient_accumulation_steps": 8,
    "steps_per_print": 10
}

# ------------------------------
# Training arguments
# ------------------------------
training_args = TrainingArguments(
    output_dir=f"./checkpoints/qwen2_5_7b_bowel_cancer_{unique_id}",
    run_name=f'my_run_{unique_id}',
    num_train_epochs=10,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=deepspeed_config['gradient_accumulation_steps'],
    learning_rate=1e-5,
    save_steps=500,
    save_total_limit=20,
    logging_steps=10,
    deepspeed=deepspeed_config,
    local_rank=int(os.getenv('LOCAL_RANK', -1)),
    ddp_find_unused_parameters=False,
    weight_decay=0.01,
    fp16=True,
    warmup_steps=500,
    evaluation_strategy="steps",
    eval_steps=50,
    report_to="wandb"
)

# ------------------------------
# Create Trainer
# ------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    tokenizer=tokenizer
)

# ------------------------------
# Initialize WandB (offline by default)
# ------------------------------
wandb.init(project=os.getenv("WANDB_PROJECT", "qwen2_5_7b_bowel_cancer"), name=unique_id)

# ------------------------------
# Start training
# ------------------------------
if __name__ == "__main__":
    start_time = time.time()
    trainer.train()
    end_time = time.time()

    # Print elapsed time
    elapsed_time = end_time - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds.")

    # Save fine-tuned model
    trainer.save_model(f"./results/qwen2_5_7b_bowel_cancer_{unique_id}")
