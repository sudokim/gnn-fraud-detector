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
            lr=0.01,
            weight_decay=5e-4, ):
        super().__init__()

        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
        self.loss_fn = nn.BCEWithLogitsLoss()

        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"].reshape(-1, 1)

        output = self.forward(input_ids, attention_mask)
        loss = self.loss_fn(output.logits, labels.float())
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"].reshape(-1, 1)

        output = self.forward(input_ids, attention_mask)
        loss = self.loss_fn(output.logits, labels.float())
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"].reshape(-1, 1)

        output = self.forward(input_ids, attention_mask)
        loss = self.loss_fn(output.logits, labels.float())
        self.log("test_loss", loss)
        self.log_metrics(output, labels, prefix="test_")
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"].reshape(-1, 1)

        output = self.forward(input_ids, attention_mask)
        return labels, output.logits

    @torch.no_grad()
    def log_metrics(self, output, target, prefix=""):

        output = output.logits
        predicted_labels = torch.round(torch.sigmoid(output))
        target_labels = target.float()

        # Calculate and log recall
        recall = recall_score(target_labels.cpu().numpy(), predicted_labels.cpu().numpy(), average="macro")
        self.log(prefix + "recall", recall)

        # Calculate and log AUC
        auc = roc_auc_score(
            target_labels.cpu().numpy(), output.cpu().detach().numpy(), average="macro", multi_class="ovr"
        )
        self.log(prefix + "auc", auc)

        # Calculate and log macro F1-score
        f1_macro = f1_score(target_labels.cpu().numpy(), predicted_labels.cpu().numpy(), average="macro")
        self.log(prefix + "f1_macro", f1_macro)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
