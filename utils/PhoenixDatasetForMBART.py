from torch.utils.data import Dataset
import pandas as pd
import torch

class PhoenixDatasetForMBART(Dataset):
    def __init__(self, data_path):
        self.glosses, self.labels = self.get_data(data_path)

    def get_data(self, data_path):
        df = pd.read_csv(data_path, delimiter = '|')
        glosses = [gloss_caps.lower() for gloss_caps in list(df["orth"])]
        labels = list(translation.lower() for translation in df["translation"])
        return glosses, labels

    def __getitem__(self,idx):
        return {"glosses": self.glosses[idx], "labels": self.labels[idx]}

    def __len__(self):
        return len(self.glosses)

    def get_label(self, idx):
        return self.labels[idx]
    
    def get_gloss(self, idx):
        return self.glosses[idx]

    def get_labels(self):
        return self.labels
    
    def get_glosses(self):
        return self.glosses


def collator(data, tokenizer):

    glosses = [datapoint["glosses"] for datapoint in data]
    labels = [datapoint["labels"] for datapoint in data]

    inputs = tokenizer(
                    glosses,
                    text_target = labels, 
                    padding=True,
                    return_tensors="pt")
    
    return inputs, labels

