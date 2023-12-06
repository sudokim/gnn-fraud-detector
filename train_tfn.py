import pickle
import warnings
from argparse import ArgumentParser, Namespace

import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.callbacks as callbacks
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
import torch_geometric.nn as gnn

from data import load_data
from tfn import LitTFN
from utils import parse_args, print_confusion_matrix


def parse_args() -> Namespace:
    parser = ArgumentParser()

    trainer = parser.add_argument_group("Trainer")
    trainer.add_argument("--max_epochs", type=int, default=1000)
    trainer.add_argument("--seed", type=int, default=0)
    trainer.add_argument("--run_name", type=str, default=None)

    return parser.parse_args()


def main(args: Namespace):
    warnings.filterwarnings("ignore")

    torch.set_float32_matmul_precision("medium")

    print("Loading data...")
    bert_embedding = torch.load("dataset/node_embeddings/bert.pt", map_location="cpu")
    data_upu = load_data(
        edge_index_path="dataset/amazon_instruments_user_product_user.npz",
        features_path="dataset/amazon_instruments_features.npy",
        labels_path="dataset/amazon_instruments_labels.pkl",
        split_file="dataset/amazon_instruments_split_masks.pkl",
    )
    data_usu = load_data(
        edge_index_path="dataset/amazon_instruments_user_star_time_user.npz",
        features_path="dataset/amazon_instruments_features.npy",
        labels_path="dataset/amazon_instruments_labels.pkl",
        split_file="dataset/amazon_instruments_split_masks.pkl",
    )
    data_uvu = load_data(
        edge_index_path="dataset/amazon_instruments_user_tfidf_user.npz",
        features_path="dataset/amazon_instruments_features.npy",
        labels_path="dataset/amazon_instruments_labels.pkl",
        split_file="dataset/amazon_instruments_split_masks.pkl",
    )

    dataloader = DataLoader([(data_upu, data_usu, data_uvu)], shuffle=False, collate_fn=lambda x: x[0])

    # Initialize and set up the model, optimizer, and data loader
    pl.seed_everything(args.seed)
    # TODO: Replace hardcoded hyperparameters with argparse
    module = LitTFN(
        num_gnns=3,
        tfn_latent_dim=16,
        gnn_input_dim=data_upu.num_features,
        gnn_latent_dim=32,
        bert_embedding=bert_embedding,
    )

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
    trainer.fit(module, train_dataloaders=dataloader, val_dataloaders=dataloader)
    print(f"Trained {trainer.current_epoch + 1} epochs")

    test_metrics = trainer.test(module, dataloaders=dataloader)

    # Save output and metrics
    output: list[tuple[torch.Tensor, torch.Tensor]] = trainer.predict(module, dataloaders=dataloader)

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
