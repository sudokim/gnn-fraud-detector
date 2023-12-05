import pickle
import warnings
from argparse import Namespace

import pytorch_lightning as pl
import pytorch_lightning.callbacks as callbacks
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from data import load_data
from module import LitGCN, ModelOutput
from utils import get_bert_embedding, parse_args, print_confusion_matrix


def main(args: Namespace):
    warnings.filterwarnings("ignore")

    torch.set_float32_matmul_precision("medium")

    print("Loading data...")
    data = load_data(
        edge_index_path=args.edge_index_path,
        features_path=args.features_path,
        labels_path=args.labels_path,
        split_file=args.split_file,
    )
    data.validate()
    dataloader = DataLoader([data], shuffle=False, collate_fn=lambda x: x[0])

    if args.bert:
        user_text = pickle.load(open(args.user_text_path, "rb"))
        bert_embedding = get_bert_embedding(user_text, args.bert_embedding_path)
    else:
        bert_embedding = None

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
        bert_embedding=bert_embedding,
        bert_reduced_dim=args.bert_reduced_dim,
        autoencoder=args.gcn_autoencoder,
    )

    if args.run_name is None:
        run_name = "GCN-Fraud-Detection"
        if args.bert:
            if args.gcn_autoencoder:
                run_name += "-BERT-AE"
            else:
                run_name += "-BERT"
        args.run_name = run_name
    else:
        run_name = args.run_name
    print(f"Run name: {run_name}")

    logger = TensorBoardLogger("logs", name=run_name, default_hp_metric=False)
    trainer = pl.Trainer(
        max_epochs=args.trainer_max_epochs,
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

    if not isinstance(logger, TensorBoardLogger):
        logger.log_hyperparams(params=args)

    print("Training...")
    trainer.fit(module, train_dataloaders=dataloader, val_dataloaders=dataloader)
    print(f"Trained {trainer.current_epoch + 1} epochs")

    test_metrics = trainer.test(module, dataloaders=dataloader)

    # Save output and metrics
    output: list[ModelOutput] = trainer.predict(module, dataloaders=dataloader)

    node_embeddings = []
    y = []
    pred = []

    for out in output:
        node_embeddings.append(out.node_embedding)
        y.append(out.y)
        pred.append(out.output)

    node_embeddings = torch.cat(node_embeddings)
    y = torch.cat(y)
    pred = torch.cat(pred)

    pred = torch.round(torch.sigmoid(pred))
    print_confusion_matrix(y, pred)

    if isinstance(logger, TensorBoardLogger):
        logger.log_hyperparams(params=args, metrics=test_metrics[0])

    # Save node_embeddings to logger directory
    torch.save(node_embeddings, logger.log_dir + "/node_embeddings.pt")
    print(f"Saved node embeddings to {logger.log_dir}/node_embeddings.pt")
    print(f" Shape: {node_embeddings.shape}")

    logger.save()


if __name__ == "__main__":
    main(parse_args())
