import torch
from transformers import GPT2Config, GPT2Model, GPT2Tokenizer

class SlingModel(torch.nn.Module):
    def __init__(self):
        super(SlingModel, self).__init__()
        self.gpt = GPT2Model.from_pretrained("gpt2")

    
    