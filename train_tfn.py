import pickle
import warnings
from argparse import ArgumentParser, Namespace

import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.callbacks as callbacks
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from data.tfn_data import EmbeddingDataset
from tfn import LitTFN
from utils import parse_args, print_confusion_matrix


def parse_args() -> Namespace:
    parser = ArgumentParser()

    trainer = parser.add_argument_group("Trainer")
    trainer.add_argument("--max_epochs", type=int, default=1000)
    trainer.add_argument("--batch_size", type=int, default=16)
    trainer.add_argument("--seed", type=int, default=0)
    trainer.add_argument("--run_name", type=str, default=None)

    paths = parser.add_argument_group("Paths")
    paths.add_argument("--embedding_paths", type=str, nargs="+", required=True)
    paths.add_argument("--labels_path", type=str, required=True)
    paths.add_argument("--split_mask_path", type=str, required=True)

    return parser.parse_args()


def main(args: Namespace):
    warnings.filterwarnings("ignore")

    torch.set_float32_matmul_precision("medium")

    print("Loading data...")
    embeddings = []
    for embedding_path in args.embedding_paths:
        try:
            embeddings.append(torch.load(embedding_path, map_location="cpu"))
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Embedding file {embedding_path} not found") from e
    print(f"Loaded embeddings: {', '.join(args.embedding_paths)}")

    labels = pickle.load(open(args.labels_path, "rb"))
    split_mask = pickle.load(open(args.split_mask_path, "rb"))
    train_dataset = EmbeddingDataset(embeddings, labels, split_mask, mode="train")
    val_dataset = EmbeddingDataset(embeddings, labels, split_mask, mode="val")
    test_dataset = EmbeddingDataset(embeddings, labels, split_mask, mode="test")

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=EmbeddingDataset.collate_fn
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=EmbeddingDataset.collate_fn
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=EmbeddingDataset.collate_fn
    )

    # Initialize and set up the model, optimizer, and data loader
    pl.seed_everything(args.seed)
    # TODO: Replace hardcoded hyperparameters with argparse
    module = LitTFN(train_dataset.embedding_dims, latent_dim=16, lr=0.001, weight_decay=5e-4, dropout=0.2)

    if args.run_name is None:
        run_name = "TFN"
        args.run_name = run_name
    else:
        run_name = args.run_name
    print(f"Run name: {run_name}")

    logger = TensorBoardLogger("logs_tfn", name=run_name, default_hp_metric=False)
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        devices=1,
        callbacks=[
            # callbacks.EarlyStopping(monitor="val_auc", patience=50, mode="max"),
            # callbacks.ModelCheckpoint(monitor="val_auc", mode="max"),
            callbacks.EarlyStopping(monitor="val_loss", patience=50, mode="min"),
            callbacks.ModelCheckpoint(monitor="val_loss", mode="min"),
            callbacks.ModelSummary(max_depth=2),
        ],
        logger=logger,
        log_every_n_steps=1,
        enable_model_summary=False,
    )

    if not isinstance(logger, TensorBoardLogger):
        logger.log_hyperparams(params=args)

    print("Training...")
    trainer.fit(module, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    print(f"Trained {trainer.current_epoch + 1} epochs")

    test_metrics = trainer.test(module, dataloaders=test_dataloader)

    # Save output and metrics
    output: list[tuple[torch.Tensor, torch.Tensor]] = trainer.predict(module, dataloaders=train_dataloader)

    y = []
    pred = []

    for out in output:
        y.append(out[0])
        pred.append(out[1])

    y = torch.cat(y)
    pred = torch.cat(pred)

    pred = torch.round(torch.sigmoid(pred))
    print_confusion_matrix(y, pred)

    if isinstance(logger, TensorBoardLogger):
        logger.log_hyperparams(params=args, metrics=test_metrics[0])

    logger.save()


if __name__ == "__main__":
    main(parse_args())
