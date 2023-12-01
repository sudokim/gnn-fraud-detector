import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from torch.nn.functional import dropout
from torch_geometric.nn import GCNConv

__all__ = ["LitGCN"]


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
        """
        super(LitGCN, self).__init__()

        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout

        # Define the GCN layers
        _modules = [GCNConv(input_dim, hidden_dim)]
        for _ in range(num_layers - 2):
            _modules.append(GCNConv(hidden_dim, hidden_dim))
        _modules.append(GCNConv(hidden_dim, output_dim))
        self.layers = nn.ModuleList(_modules)

        assert isinstance(pos_weight, float), f"pos_weight should be a float, got {type(pos_weight)}"
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
        self.save_hyperparameters()

        self._is_binary_classification = output_dim == 1

    def forward(self, x, edge_index):
        """
        Forward pass of the GCN model.

        Args:
            x (torch.Tensor): Node features.
            edge_index (torch.Tensor): Edge indices.

        Returns:
            torch.Tensor: Output logits.
        """
        for layer in self.layers[:-1]:
            x = layer(x, edge_index)
            x = torch.relu(x)
            x = dropout(x, p=self.dropout, training=self.training)

        return self.layers[-1](x, edge_index)

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

        output = self.forward(x, edge_index)
        output = output.squeeze(-1)

        y_masked = y[batch.train_mask]
        output_masked = output[batch.train_mask][y_masked != -100]
        y_masked = y_masked[y_masked != -100]

        loss = self.loss_fn.forward(output_masked, y_masked)

        self.log("train_loss", loss)
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

        output = self.forward(x, edge_index)
        output = output.squeeze(-1)

        y_masked = y[batch.val_mask]
        output_masked = output[batch.val_mask][y_masked != -100]
        y_masked = y_masked[y_masked != -100]

        loss = self.loss_fn.forward(output_masked, y_masked)

        self.log("val_loss", loss)
        self.log_metrics(output_masked, y_masked, prefix="val_")
        return loss

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

        output = self.forward(x, edge_index)
        output = output.squeeze(-1)

        y_masked = y[batch.test_mask]
        output_masked = output[batch.test_mask][y_masked != -100]
        y_masked = y_masked[y_masked != -100]

        loss = self.loss_fn.forward(output_masked, y_masked)

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

        output = self.forward(x, edge_index)
        output = output.squeeze(-1)

        y_masked = y[batch.test_mask]
        output_masked = output[batch.test_mask][y_masked != -100]
        y_masked = y_masked[y_masked != -100]

        return y_masked, output_masked

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
