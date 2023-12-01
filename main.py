import pickle
import warnings
from argparse import Namespace

import pytorch_lightning as pl
import pytorch_lightning.callbacks as callbacks
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from data import load_data, load_data_from_matlab
from module import LitGCN
from utils import parse_args, print_confusion_matrix


def main(args: Namespace):
    warnings.filterwarnings("ignore")

    torch.set_float32_matmul_precision("medium")

    print("Loading data...")
    data = load_data(
        edge_index_path="dataset/amazon_instruments_user_all.npz",
        features_path="dataset/amazon_instruments_features.npy",
        labels_path="dataset/amazon_instruments_labels.pkl",
        split_file="dataset/amazon_instruments_split_masks.pkl",
    )
    data.validate()
    dataloader = DataLoader([data], shuffle=False, collate_fn=lambda x: x[0])

    # Initialize and set up the model, optimizer, and data loader
    pl.seed_everything(args.seed)
    module = LitGCN(
        input_dim=data.num_features,
        hidden_dim=args.gcn_hidden_dim,
        output_dim=1,
        lr=args.module_lr,
        weight_decay=args.module_weight_decay,
        pos_weight=args.gcn_pos_weight,
        dropout=args.gcn_dropout,
        num_layers=args.gcn_num_layers,
    )

    logger = TensorBoardLogger("logs", name="GCN-Fraud-Detection", default_hp_metric=False)
    trainer = pl.Trainer(
        max_epochs=3000,
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
        enable_progress_bar=False,
    )

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

    logger.log_hyperparams(params=args, metrics=test_metrics[0])
    logger.save()


if __name__ == "__main__":
    main(parse_args())
