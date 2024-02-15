import pandas as pd
from torch.utils.data import Dataset


class USPTO(Dataset):
    def __init__(
            self,
            path,
            tokenizer,
    ):
        self.data = pd.read_csv(path)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)
    
    def tokenize(self, x):
        return self.tokenizer.encode(x, add_special_tokens=False, return_tensors="pt")
    
    def get(self, idx):
        reaction, smiles, paragraph = self.data.iloc[idx]
        return reaction, smiles, paragraph


    def __getitem__(self, idx):
        reaction, smiles, paragraph = self.get(idx)
        reaction = self.tokenize(reaction)
        smiles = self.tokenize(smiles)
        paragraph = self.tokenize(paragraph)
        return reaction, smiles, paragraph
    

    


    

