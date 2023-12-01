import torch
from torch.utils.data import Dataset


class FraudBertDataset(Dataset):
    def __init__(self, data, tokenizer, max_item_len=64):
        self.data = data
        self.tokenizer = tokenizer
        self.item_max_len = max_item_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        [_, item, review, label] = self.data[index]
        item = self.tokenizer(item, padding="max_length", truncation=True, max_length=self.item_max_len)
        review = self.tokenizer(review, padding="max_length", truncation=True)
        input_ids = [self.tokenizer.cls_token_id] + item["input_ids"] + [self.tokenizer.sep_token_id] + review[
            "input_ids"] + [self.tokenizer.sep_token_id]
        input_ids = input_ids[:self.tokenizer.model_max_length]
        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "label": torch.tensor(label)
        }


def collate_fn(batch):
    input_ids = []
    attention_masks = []
    labels = []

    for data in batch:
        input_ids.append(data["input_ids"])
        attention_masks.append(data["attention_mask"])
        labels.append(data["label"])

    # Pad input_ids and attention_masks to the same length
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_masks = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "label": labels
    }
