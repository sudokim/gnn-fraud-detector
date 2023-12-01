from argparse import ArgumentParser, Namespace
import pytorch_lightning as pl
import pytorch_lightning.callbacks as callbacks
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from data import load_data_from_matlab
from module import LitGCN
from utils import print_confusion_matrix


def parse_args():
    parser = ArgumentParser()

    data = parser.add_argument_group("Data")
    data.add_argument(
        "--dataset_mat_path", type=str, default="dataset/Amazon.mat", help="Path to the dataset in .mat format"
    )
    data.add_argument("--dataset_split_seed", type=int, default=42, help="Seed for splitting the dataset")

    module = parser.add_argument_group("Module")
    module.add_argument("--module_type", type=str, default="GCN", help="Type of the module to use")
    module.add_argument("--module_weight_decay", type=float, default=5e-4, help="Weight decay of the module")
    module.add_argument("--module_lr", type=float, default=0.01, help="Learning rate of the module")

    module_gcn = module.add_argument_group("GCN")
    module_gcn.add_argument("--gcn_hidden_dim", type=int, default=16, help="Hidden dimension of the GCN module")
    module_gcn.add_argument("--gcn_dropout", type=float, default=0.5, help="Dropout rate of the GCN module")
    module_gcn.add_argument("--gcn_num_layers", type=int, default=2, help="Number of layers of the GCN module")
    module_gcn.add_argument("--gcn_pos_weight", type=float, default=10, help="Positive class weight of the GCN module")

    trainer = parser.add_argument_group("Trainer")
    trainer.add_argument("--trainer_max_epochs", type=int, default=2000, help="Maximum number of epochs to train")
    trainer.add_argument("--seed", type=int, default=42, help="Seed for the trainer")

    return parser.parse_args()


def main(args: Namespace):
    torch.set_float32_matmul_precision("medium")

    data = load_data_from_matlab(args.dataset_mat_path, split_ratio=(0.8, 0.1, 0.1), seed=args.dataset_split_seed)
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

    trainer = pl.Trainer(
        max_epochs=2000,
        devices=1,
        callbacks=[
            # callbacks.EarlyStopping(monitor="val_auc", patience=50, mode="max"),
            # callbacks.ModelCheckpoint(monitor="val_auc", mode="max"),
            callbacks.EarlyStopping(monitor="val_loss", patience=50, mode="min"),
            callbacks.ModelCheckpoint(monitor="val_loss", mode="min"),
        ],
        logger=TensorBoardLogger("logs", name="GCN-Fraud-Detection"),
        log_every_n_steps=1,
    )
    trainer.logger.log_hyperparams(args)

    trainer.fit(module, train_dataloaders=dataloader, val_dataloaders=dataloader)
    trainer.test(module, dataloaders=dataloader)

    output: list = trainer.predict(module, dataloaders=dataloader)

    y = []
    pred = []
    for out in output:
        y.append(out[0])
        pred.append(out[1])

    y = torch.cat(y)
    pred = torch.cat(pred)

    pred = torch.round(torch.sigmoid(pred))
    print_confusion_matrix(y, pred)


if __name__ == "__main__":
    main()
