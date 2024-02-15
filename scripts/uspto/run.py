import token
import torch
from dataset import USPTO
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

def run():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    dataset = USPTO(
        path="uspto.csv",
        tokenizer=tokenizer,
    )

    reaction, smiles, paragraph = dataset.get(0)
    question, text = reaction, smiles
    
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    # question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
    inputs = tokenizer(question, text, return_tensors="pt")
    outputs = model.generate(**inputs)
    outputs = tokenizer.decode(outputs[0], skip_special_tokens=False)
    print(outputs)

    
    





if __name__ == "__main__":
    run()