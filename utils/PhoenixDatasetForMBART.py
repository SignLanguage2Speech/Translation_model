from torch.utils.data import Dataset
import pandas as pd

class PhoenixDatasetForMBART(Dataset):
    def __init__(self, data_path, tokenizer):
        self.tokenizer = tokenizer
        self.glosses, self.labels = self.get_data(data_path)

    def get_data(self, data_path):
        df = pd.read_csv(data_path, delimiter = '|')
        glosses = [gloss_caps.lower() for gloss_caps in list(df["orth"])]
        labels = list(df["translation"])
        return glosses, labels

    def __getitem__(self,idx):
        tokens = self.tokenizer(
                    self.glosses[idx], 
                    text_target = self.labels[idx], 
                    return_tensors="pt")
        return {"input_ids": tokens["input_ids"][0], "attention_mask": tokens["attention_mask"][0], "labels": tokens["labels"][0]}

    def __len__(self):
        return len(self.glosses)

    def get_label(self, idx):
        return self.labels[idx]
    
    def get_gloss(self, idx):
        return self.glosses[idx]