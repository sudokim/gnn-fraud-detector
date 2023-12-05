"""
Tensor Fusion Network (TFN) model from "Tensor Fusion Network for Multimodal Sentiment Analysis"
"""
from sklearn.metrics import f1_score, recall_score, roc_auc_score
import torch
import pytorch_lightning as pl
import torch.nn as nn


class LitTFN(pl.LightningModule):
    """
    TFN with frozen embeddings.
    """

    def __init__(self, embedding_dims, latent_dim, lr=0.01, weight_decay=5e-4, dropout=0.1, pos_weight=2):
        assert isinstance(embedding_dims, list), "Embedding dimensions must be a list"
        assert all(
            map(lambda x: isinstance(x, int) and x > 0, embedding_dims)
        ), "All embedding dimensions must be positive integers"
        assert isinstance(latent_dim, int) and latent_dim > 0, "Latent dimension must be a positive integer"
        super(LitTFN, self).__init__()

        self.embedding_dims = embedding_dims
        self.num_embeddings = len(embedding_dims)
        self.latent_dim = latent_dim
        self.lr = lr
        self.weight_decay = weight_decay

        self.post_fusion_layers = nn.ModuleList(
            [nn.Linear(embedding_dim, latent_dim) for embedding_dim in embedding_dims]
        )
        self.decision_layer = nn.Sequential(
            nn.Dropout(dropout),
            nn.LazyLinear(128),
            nn.ReLU(),
            nn.LazyLinear(128),
            nn.ReLU(),
            nn.LazyLinear(1),
        )

        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))

        self._predictions = []
        self._labels = []

    def forward(self, *embeddings):
        # Concatenate one vector column to each embedding
        post_fusion_embeddings = [
            post_fusion_layer.forward(embedding)
            for post_fusion_layer, embedding in zip(self.post_fusion_layers, embeddings)
        ]

        post_fusion_embeddings_cat = [
            torch.cat(
                [embedding, torch.ones(embedding.shape[0], 1, device=embedding.device, requires_grad=False)], dim=1
            )
            for embedding in post_fusion_embeddings
        ]

        # Kronecker product
        product = post_fusion_embeddings_cat[0]  # (batch_size, (latent_dim + 1))
        for post_fusion_embedding in post_fusion_embeddings_cat[1:]:
            product = product.unsqueeze(-1)  # (batch_size, latent_dim ** n, 1)
            post_fusion_embedding = post_fusion_embedding.unsqueeze(-2)  # (batch_size, 1, latent_dim)

            product = torch.bmm(product, post_fusion_embedding)  # (batch_size, latent_dim ** n, latent_dim)
            product = torch.flatten(product, start_dim=1)  # (batch_size, latent_dim ** (n + 1))

        predictions = self.decision_layer.forward(product)

        return predictions

    def training_step(self, batch, batch_idx):
        embeddings, labels = batch
        predictions = self.forward(*embeddings).squeeze(-1)

        y_mask = torch.ne(labels, -100)
        predictions = predictions[y_mask]
        labels = labels[y_mask]

        loss = self.loss_fn(predictions, labels.float())

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_validation_epoch_start(self):
        self._predictions.clear()
        self._labels.clear()

    def validation_step(self, batch, batch_idx):
        embeddings, labels = batch
        predictions = self.forward(*embeddings).squeeze(-1)

        y_mask = torch.ne(labels, -100)
        predictions = predictions[y_mask]
        labels = labels[y_mask]

        loss = self.loss_fn(predictions, labels.float())

        self._predictions.append(predictions)
        self._labels.append(labels)

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_validation_epoch_end(self):
        predictions = torch.cat(self._predictions)
        labels = torch.cat(self._labels)
        self.log_metrics(predictions, labels, prefix="val_")

    def on_test_epoch_start(self):
        self._predictions.clear()
        self._labels.clear()

    def test_step(self, batch, batch_idx):
        embeddings, labels = batch
        predictions = self.forward(*embeddings).squeeze(-1)

        y_mask = torch.ne(labels, -100)
        predictions = predictions[y_mask]
        labels = labels[y_mask]

        loss = self.loss_fn(predictions, labels.float())

        self._predictions.append(predictions)
        self._labels.append(labels)

        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_test_epoch_end(self):
        predictions = torch.cat(self._predictions)
        labels = torch.cat(self._labels)
        self.log_metrics(predictions, labels, prefix="test_")

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        embeddings, labels = batch
        predictions = self.forward(*embeddings).squeeze(-1)

        return labels, predictions

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    @torch.no_grad()
    def log_metrics(self, output, target, prefix=""):
        """
        Log additional metrics during validation and test.

        Args:
            output (torch.Tensor): Model predictions.
            target (torch.Tensor): Ground truth labels.
            prefix (str): Metric name prefix for logging.
        """
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
