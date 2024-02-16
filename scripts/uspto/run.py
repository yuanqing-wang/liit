import token
import torch
import pandas as pd
from dataset import USPTO
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset, Dataset

def run():
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model = AutoModelForCausalLM.from_pretrained(model_name)

    df = pd.read_csv("uspto.csv")[:1000]
    dataset = Dataset.from_pandas(df)

    def tokenize_function(examples):
        return tokenizer(
            examples["reaction"], 
            examples["paragraph"],
            padding="max_length", 
            truncation=True
        )

    tokenized_datasets = dataset.map(tokenize_function, batched=True, batch_size=4)

    # split the dataset into train and validation
    tokenized_datasets = tokenized_datasets.train_test_split(test_size=0.2)

    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=1e-4,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=10,
        save_total_limit=2,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
    )

    trainer.train()

    
    





if __name__ == "__main__":
    run()