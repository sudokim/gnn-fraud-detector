import pytorch_lightning as pl
import pytorch_lightning.callbacks as callbacks
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from module import *
from torch.utils.data import DataLoader

SEED = 42


def main():
    pl.seed_everything(SEED)
    torch.set_float32_matmul_precision("medium")

    data = load_data_from_matlab("dataset/Amazon.mat", split_ratio=(0.8, 0.1, 0.1), seed=SEED)
    data.validate()
    dataloader = DataLoader([data], shuffle=False, collate_fn=lambda x: x[0])

    # Initialize and set up the model, optimizer, and data loader
    module = LitGCN(input_dim=data.num_features, hidden_dim=16, output_dim=1, pos_weight=10)

    trainer = pl.Trainer(
        max_epochs=1000,
        devices=1,
        callbacks=[
            # callbacks.EarlyStopping(monitor="val_auc", patience=50, mode="max"),
            # callbacks.ModelCheckpoint(monitor="val_auc", mode="max"),
            callbacks.EarlyStopping(monitor="val_loss", patience=50, mode="min"),
            callbacks.ModelCheckpoint(monitor="val_loss", mode="min"),
        ],
        logger=TensorBoardLogger("lightning_logs", name="GCN-Fraud-Detection"),
        log_every_n_steps=1,
    )

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
