import warnings
from argparse import Namespace
import time

import pytorch_lightning as pl
import pytorch_lightning.callbacks as callbacks
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding
from pytorch_lightning.loggers import WandbLogger

from bert_datamodule.dataset import FraudBertDataset
from data.load_data_bert import load_amazon_data, load_steam_data
from module.fraudbert import FraudBert
from utils import print_confusion_matrix
from utils.arg_parser import parse_bert_args


class Collator:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, batch):
        text = [(item, review) for item, review, _ in batch]
        labels = [label for _, _, label in batch]

        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        inputs["labels"] = torch.tensor(labels)

        return inputs


def main(args: Namespace):
    warnings.filterwarnings("ignore")
    torch.set_float32_matmul_precision("medium")

    pl.seed_everything(args.seed)

    # Load data
    if args.dataset == "amazon":
        train_data, val_data, test_data = load_amazon_data(args.dataset)
    elif args.dataset == "steam":
        train_data, val_data, test_data = load_steam_data(args.dataset)
    else:
        raise ValueError("Invalid dataset")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_dataset = FraudBertDataset(train_data)
    val_dataset = FraudBertDataset(val_data)
    test_dataset = FraudBertDataset(test_data)

    collator = Collator(tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    # Initialize and set up the model, trainer, logger
    module = FraudBert(args.model_name, pos_weight=args.pos_weight, lr=args.module_lr,
                       weight_decay=args.module_weight_decay)
    unix_time = int(time.time())
    logger = WandbLogger(project="fraudster", name=f"FraudBert_{unix_time}", log_model=True)
    trainer = pl.Trainer(
        max_epochs=3000,
        devices=1,
        callbacks=[
            callbacks.EarlyStopping(monitor="val/loss", patience=5, mode="min"),
            callbacks.ModelCheckpoint(monitor="val/loss", mode="min"),
            callbacks.ModelSummary(max_depth=2),
        ],
        logger=logger,
        log_every_n_steps=1,
        enable_model_summary=True,
        enable_progress_bar=True,
    )

    # Train
    print("Training...")
    trainer.fit(module, train_dataloader, val_dataloader)
    print(f"Trained {trainer.current_epoch + 1} epochs")

    # Test
    test_metrics = trainer.test(module, test_dataloader)

    # Save output and metrics
    output = trainer.predict(module, test_dataloader)
    y, pred = [], []

    for out in output:
        y.append(out[0].cpu())
        pred.append(out[1].cpu())
    y = torch.cat(y)
    pred = torch.cat(pred)
    pred = torch.round(torch.sigmoid(pred))
    print_confusion_matrix(y, pred)

    logger.log_hyperparams(params=args, metrics=test_metrics[0])
    logger.save()


if __name__ == "__main__":
    main(parse_bert_args())
