from dataclasses import dataclass

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from torch.nn.functional import dropout
from torch_geometric.nn import GCNConv

from ._autoencoder import AutoEncoder

__all__ = ["LitGCN", "ModelOutput"]


@dataclass
class ModelOutput:
    """
    Dataclass for model output.

    Attributes:
        output (torch.Tensor): Model predictions.
        auto_dec (torch.Tensor): Autoencoder output.
        node_embedding (torch.Tensor): Node embeddings.
        y (torch.Tensor): Ground truth labels.
    """

    output: torch.Tensor
    auto_dec: torch.Tensor | None = None
    node_embedding: torch.Tensor | None = None
    y: torch.Tensor | None = None


class LitGCN(pl.LightningModule):
    """
    LightningModule for Graph Convolutional Network (GCN) with node classification.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_layers,
        lr=0.01,
        weight_decay=5e-4,
        pos_weight: int = 10,
        dropout: float = 0.2,
        bert_embedding: torch.Tensor | None = None,
        bert_reduced_dim: int | None = None,
        autoencoder: bool = False,
    ):
        """
        Initializes a GCN model for node classification.

        Args:
            input_dim (int): Dimensionality of node features.
            hidden_dim (int): Dimensionality of hidden layers.
            output_dim (int): Number of output classes.
            num_layers (int): Number of hidden layers (excluding the output layer
            lr (float): Learning rate (default: 0.01).
            weight_decay (float): Weight decay (default: 5e-4).
            pos_weight (int): Weight of positive class in the loss function.
            dropout (float): Dropout probability (default: 0.2).
            bert_embedding (torch.Tensor): Pre-computed BERT embedding. If None, BERT embedding will not be used.
            bert_reduced_dim (int): Dimensionality of the reduced BERT embedding.
            autoencoder (bool): Whether to use autoencoder for BERT embedding.
        """
        super(LitGCN, self).__init__()

        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self._autoencoder = autoencoder

        if bert_embedding is None:
            self.bert_embedding = None
            self.bert_adapter = None
            self.bert_layer_norm = None
        else:
            self.bert_embedding = nn.Embedding.from_pretrained(bert_embedding, freeze=True)
            if autoencoder:
                self.bert_adapter = AutoEncoder([bert_embedding.shape[1], bert_reduced_dim])
            else:
                self.bert_adapter = nn.Linear(bert_embedding.shape[1], bert_reduced_dim)
            self.bert_layer_norm = nn.LayerNorm(bert_reduced_dim)

            input_dim += bert_reduced_dim

        # Define the GCN layers
        _modules = [GCNConv(input_dim, hidden_dim)]
        for _ in range(num_layers - 1):
            _modules.append(GCNConv(hidden_dim, hidden_dim))
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.layers = nn.ModuleList(_modules)

        assert isinstance(pos_weight, float), f"pos_weight should be a float, got {type(pos_weight)}"
        self.loss_fn_clf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
        self.loss_fn_ae = nn.MSELoss()

        self.save_hyperparameters(ignore="bert_embedding")

        self._is_binary_classification = output_dim == 1

    def forward(self, x, edge_index):
        """
        Forward pass of the GCN model.

        Args:
            x (torch.Tensor): Node features.
            edge_index (torch.Tensor): Edge indices.

        Returns:
            ModelOutput: Model output.
        """
        if self.bert_embedding is not None and self.bert_adapter is not None and self.bert_layer_norm is not None:
            if self._autoencoder:
                auto_enc, auto_dec = self.bert_adapter.forward(self.bert_embedding.weight)
                bert_embedding_forward = self.bert_layer_norm.forward(auto_enc)
            else:
                bert_embedding_forward = self.bert_adapter.forward(self.bert_embedding.weight)
                auto_enc = None
                auto_dec = None
            x = torch.cat([x, bert_embedding_forward], dim=1)

        else:
            auto_enc = None
            auto_dec = None

        node_embedding = None
        for layer in self.layers:
            node_embedding = layer(x, edge_index)
            x = torch.relu(node_embedding)
            x = dropout(x, p=self.dropout, training=self.training)
        
        output = self.linear(x)

        return ModelOutput(
            output=output,
            auto_dec=auto_dec,
            node_embedding=node_embedding,
        )

    def training_step(self, batch, batch_idx):
        """
        Training step for GCN.

        Args:
            batch: Batch of data.
            batch_idx: Batch index.

        Returns:
            torch.Tensor: Training loss.
        """
        x, edge_index, y = batch.x, batch.edge_index, batch.y

        model_output = self.forward(x, edge_index)
        output = model_output.output.squeeze(-1)

        y_masked = y[batch.train_mask]
        output_masked = output[batch.train_mask][y_masked != -100]
        y_masked = y_masked[y_masked != -100]

        loss_clf = self.loss_fn_clf.forward(output_masked, y_masked)
        if self._autoencoder:
            loss_ae = self.loss_fn_ae.forward(model_output.auto_dec, self.bert_embedding.weight)
            loss = loss_clf + loss_ae
        else:
            loss_ae = None
            loss = loss_clf

        self.log_dict(
            {
                "train_loss": loss,
                "train_loss_clf": loss_clf,
            }
            | ({"train_loss_ae": loss_ae} if loss_ae is not None else {})
        )
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for GCN.

        Args:
            batch: Batch of data.
            batch_idx: Batch index.

        Returns:
            torch.Tensor: Validation loss.
        """
        x, edge_index, y = batch.x, batch.edge_index, batch.y

        model_output = self.forward(x, edge_index)
        output = model_output.output.squeeze(-1)

        y_masked = y[batch.val_mask]
        output_masked = output[batch.val_mask][y_masked != -100]
        y_masked = y_masked[y_masked != -100]

        loss_clf = self.loss_fn_clf.forward(output_masked, y_masked)
        if self._autoencoder:
            loss_ae = self.loss_fn_ae.forward(model_output.auto_dec, self.bert_embedding.weight)
            self.log("val_loss_ae", loss_ae)

        self.log("val_loss", loss_clf)
        self.log_metrics(output_masked, y_masked, prefix="val_")
        return loss_clf

    def test_step(self, batch, batch_idx):
        """
        Test step for GCN.

        Args:
            batch: Batch of data.
            batch_idx: Batch index.

        Returns:
            torch.Tensor: Test loss.
        """
        x, edge_index, y = batch.x, batch.edge_index, batch.y

        model_output = self.forward(x, edge_index)
        output = model_output.output.squeeze(-1)

        y_masked = y[batch.test_mask]
        output_masked = output[batch.test_mask][y_masked != -100]
        y_masked = y_masked[y_masked != -100]

        loss = self.loss_fn_clf.forward(output_masked, y_masked)
        if self._autoencoder:
            loss_ae = self.loss_fn_ae.forward(model_output.auto_dec, self.bert_embedding.weight)
            self.log("test_loss_ae", loss_ae)

        self.log("test_loss", loss)
        self.log_metrics(output_masked, y_masked, prefix="test_")
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        """
        Predict step for GCN.

        Args:
            batch: Batch of data.
            batch_idx: Batch index.
            dataloader_idx: Dataloader index.

        Returns:
            torch.Tensor: Model predictions.
        """
        x, edge_index, y = batch.x, batch.edge_index, batch.y

        model_output = self.forward(x, edge_index)
        output = model_output.output.squeeze(-1)

        y_masked = y[batch.test_mask]
        output_masked = output[batch.test_mask][y_masked != -100]
        y_masked = y_masked[y_masked != -100]

        return ModelOutput(
            output=output_masked,
            node_embedding=model_output.node_embedding,
            y=y_masked,
        )

    @torch.no_grad()
    def log_metrics(self, output, target, prefix=""):
        """
        Log additional metrics during validation and test.

        Args:
            output (torch.Tensor): Model predictions.
            target (torch.Tensor): Ground truth labels.
            prefix (str): Metric name prefix for logging.
        """
        if self._is_binary_classification:
            predicted_labels = torch.round(torch.sigmoid(output))
        else:
            predicted_labels = torch.argmax(torch.softmax(output), dim=1)
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
        """
        Configure the optimizer for training.

        Returns:
            torch.optim.Optimizer: Optimizer object.
        """
        return optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
