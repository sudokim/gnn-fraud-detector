import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from torch.nn.functional import dropout

__all__ = ["FraudBert"]

from transformers import AutoModelForSequenceClassification


class FraudBert(pl.LightningModule):
    """
    LightningModule for LM (Bert) with fraud detection.
    """

    def __init__(
            self,
            model_name,
            pos_weight: int,
            lr:float,
            weight_decay=1e-2, ):
        super().__init__()

        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))

        self.lr = lr
        self.weight_decay = weight_decay

        self._labels = []
        self._predictions = []

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)["logits"].squeeze(-1)

    def training_step(self, batch, batch_idx):
        labels = batch["labels"]
        del batch["labels"]

        output = self.forward(**batch)
        loss = self.loss_fn(output, labels.float())
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        self._labels.clear()
        self._predictions.clear()

    def validation_step(self, batch, batch_idx):
        labels = batch["labels"]
        del batch["labels"]

        output = self.forward(**batch)
        loss = self.loss_fn(output, labels.float())
        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self._labels.append(labels.cpu())
        self._predictions.append(output.cpu())
        
        return loss

    def on_validation_epoch_end(self):
        labels = torch.cat(self._labels, dim=0)
        predictions = torch.cat(self._predictions, dim=0)

        self.log_metrics(predictions, labels, prefix="val/")
        
    def on_test_epoch_start(self):
        self._labels.clear()
        self._predictions.clear()
    
    def test_step(self, batch, batch_idx):
        labels = batch["labels"]
        del batch["labels"]

        output = self.forward(**batch)
        loss = self.loss_fn(output, labels.float())
        self.log("test/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        self._labels.append(labels.cpu())
        self._predictions.append(output.cpu())
        
        return loss

    def on_test_epoch_end(self):
        labels = torch.cat(self._labels, dim=0)
        predictions = torch.cat(self._predictions, dim=0)

        self.log_metrics(predictions, labels, prefix="test/")

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        labels = batch["labels"]
        del batch["labels"]
        
        return labels, self.forward(**batch)

    @torch.no_grad()
    def log_metrics(self, output, target, prefix=""):
        predicted_labels = torch.round(torch.sigmoid(output))
        target_labels = target.float()

        # Calculate and log recall
        recall = recall_score(target_labels.cpu().numpy(), predicted_labels.cpu().numpy(), average="macro")
        self.log(prefix + "recall", recall, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Calculate and log AUC
        auc = roc_auc_score(
            target_labels.cpu().numpy(), output.cpu().detach().numpy(), average="macro", multi_class="ovr"
        )
        self.log(prefix + "auc", auc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Calculate and log macro F1-score
        f1_macro = f1_score(target_labels.cpu().numpy(), predicted_labels.cpu().numpy(), average="macro")
        self.log(prefix + "f1_macro", f1_macro, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
