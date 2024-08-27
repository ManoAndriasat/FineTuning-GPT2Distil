import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling

# Load the DistilGPT-2 model and tokenizer
model_name = "distilgpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Prepare the dataset
def load_dataset(file_path, tokenizer, block_size=128):
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size,
    )
    return dataset

# Data collator for batching
def data_collator(tokenizer):
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

# Load and tokenize your training data
train_file = "data.txt"  # Add your training data here
train_dataset = load_dataset(train_file, tokenizer)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./distilgpt2-finetuned-bio",
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=2,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
    learning_rate=5e-5,  # Adjust this to a small value
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator(tokenizer),
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
trainer.save_model("./distilgpt2-finetuned-bio")
tokenizer.save_pretrained("./distilgpt2-finetuned-bio")

print("Model fine-tuned and saved successfully!")
