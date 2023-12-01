import warnings
from argparse import Namespace

import pytorch_lightning as pl
import pytorch_lightning.callbacks as callbacks
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from pytorch_lightning.loggers import TensorBoardLogger

from bert_datamodule.dataset import FraudBertDataset
from data.load_data_bert import load_text_data
from module.fraudbert import FraudBert
from utils import print_confusion_matrix
from utils.arg_parser import parse_bert_args


def main(args: Namespace):
    warnings.filterwarnings("ignore")
    torch.set_float32_matmul_precision("medium")

    pl.seed_everything(args.seed)

    # Load data
    train_data, val_data, test_data = load_text_data()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_dataset = FraudBertDataset(train_data, tokenizer)
    val_dataset = FraudBertDataset(val_data, tokenizer)
    test_dataset = FraudBertDataset(test_data, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize and set up the model, trainer, logger
    module = FraudBert(args.model_name, lr=args.module_lr, weight_decay=args.module_weight_decay)
    logger = pl.loggers.TensorBoardLogger("logs", name="FraudBert", default_hp_metric=False)
    trainer = pl.Trainer(
        max_epochs=3000,
        devices=1,
        callbacks=[
            callbacks.EarlyStopping(monitor="val_loss", patience=50, mode="min"),
            callbacks.ModelCheckpoint(monitor="val_loss", mode="min"),
            callbacks.ModelSummary(max_depth=2),
        ],
        logger=logger,
        log_every_n_steps=1,
        enable_model_summary=False,
        enable_progress_bar=False,
    )

    # Train
    print("Training...")
    trainer.fit(module, train_dataloader, val_dataloader)
    print(f"Trained {trainer.current_epoch + 1} epochs")

    # Test
    test_metrics = trainer.test(module, test_dataloader)

    # # Save output and metrics
    # output = trainer.predict(module, test_dataloader)
    # print(f"output {output}")
    # y, pred = [], []
    #
    # for out in output:
    #     y.append(out[0].cpu())
    #     pred.append(out[1].cpu())
    # y = torch.cat(y)
    # pred = torch.cat(pred)
    # pred = torch.round(torch.sigmoid(pred))
    # print_confusion_matrix(y, pred)

    logger.log_hyperparams(params=args, metrics=test_metrics[0])
    logger.save()


if __name__ == "__main__":
    main(parse_bert_args())
