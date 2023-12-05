import torch
from torch.utils.data import Dataset


class FraudBertDataset(Dataset):
    def __init__(self, data, max_item_len=64):
        self.data = data
        self.item_max_len = max_item_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        [_, item, review, label] = self.data[index]
        return item, review, label