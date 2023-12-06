import warnings
from argparse import ArgumentParser, Namespace

import pytorch_lightning as pl
import pytorch_lightning.callbacks as callbacks
import torch
import torch_geometric.nn as gnn
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from data import load_data
from tfn import LitTFN
from utils import parse_args, print_confusion_matrix


def parse_args() -> Namespace:
    parser = ArgumentParser()

    trainer = parser.add_argument_group("Trainer")
    trainer.add_argument("--max_epochs", type=int, default=1000, help="Maximum number of epochs to train for")
    trainer.add_argument("--seed", type=int, default=0, help="Random seed to use for training")
    trainer.add_argument("--run_name", type=str, default=None, help="Name of the run for logging purposes")

    module = parser.add_argument_group("Module")
    module.add_argument("--num_gnns", type=int, default=3, help="Number of GNNs to use")
    module.add_argument("--tfn_latent_dim", type=int, default=16, help="Latent dimension of the model")
    module.add_argument("--gnn_latent_dim", type=int, default=32, help="Latent dimension of the GNN")
    module.add_argument("--gnn_cls", type=str, default="GCNConv", help="GNN class to use to construct the GNN")
    module.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    module.add_argument("--weight_decay", type=float, default=5e-4, help="L2 normalization")
    module.add_argument("--dropout", type=float, default=0.1, help="Dropout within the model")
    module.add_argument("--pos_weight", type=int, default=1, help="Loss weight for the positive class")

    data = parser.add_argument_group("Data")
    data.add_argument("--edge_index_paths", type=str, nargs="+", help="Paths to edge index files")
    data.add_argument("--features_path", type=str, help="Path to features file")
    data.add_argument("--labels_path", type=str, help="Path to labels file")
    data.add_argument("--split_file", type=str, help="Path to split file")
    data.add_argument("--bert_embedding_path", type=str, help="Path to BERT embeddings file")

    args = parser.parse_args()

    args.gnn_cls = getattr(gnn, args.gnn_cls)

    return args


def main(args: Namespace):
    warnings.filterwarnings("ignore")

    torch.set_float32_matmul_precision("medium")

    print("Loading data...")
    bert_embedding = torch.load(args.bert_embedding_path, map_location="cpu")
    data = tuple(
        load_data(
            edge_index_path=edge_index_path,
            features_path=args.features_path,
            labels_path=args.labels_path,
            split_file=args.split_file,
        )
        for edge_index_path in args.edge_index_paths
    )

    dataloader = DataLoader([data], shuffle=False, collate_fn=lambda x: x[0])

    # Initialize and set up the model, optimizer, and data loader
    pl.seed_everything(args.seed)
    # TODO: Replace hardcoded hyperparameters with argparse
    module = LitTFN(
        num_gnns=len(data),
        tfn_latent_dim=args.tfn_latent_dim,
        gnn_input_dim=data[0].num_features,
        gnn_latent_dim=args.gnn_latent_dim,
        bert_embedding=bert_embedding,
        gnn_cls=args.gnn_cls,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        pos_weight=args.pos_weight,
    )
    module.compile(dynamic=False, mode="reduce-overhead")

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
