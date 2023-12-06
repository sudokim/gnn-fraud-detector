"""
Tensor Fusion Network (TFN) model from "Tensor Fusion Network for Multimodal Sentiment Analysis"
"""
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from sklearn.metrics import f1_score, recall_score, roc_auc_score


class LitTFN(pl.LightningModule):
    """
    TFN with frozen embeddings.
    """

    def __init__(
        self,
        num_gnns,
        tfn_latent_dim,
        gnn_input_dim,
        gnn_latent_dim,
        bert_embedding,
        gnn_cls=gnn.GCNConv,
        lr=0.01,
        weight_decay=5e-4,
        dropout=0.1,
        pos_weight=1,
    ):
        """
        Initialize TFN model.

        Args:
            num_gnns (int): Number of GNNs to use.
            tfn_latent_dim (int): Latent dimension of the model.
            gnn_input_dim (int): Number of input features to the GNN.
            gnn_latent_dim (int): Latent dimension of the GNN.
            bert_embedding (torch.Tensor): BERT embeddings.
            gnn_cls (Type[nn.Module], optional): GNN class to use to construct the GNN. Defaults to gnn.GCNConv.
            lr (float, optional): Learning rate. Defaults to 0.01.
            weight_decay (float, optional): L2 normalization. Defaults to 5e-4.
            dropout (float, optional): Dropout within the model. Defaults to 0.1.
            pos_weight (int, optional): Loss weight for the positive class. Defaults to 1.
        """
        super(LitTFN, self).__init__()

        self.tfn_latent_dim = tfn_latent_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.bert_embedding = nn.Parameter(bert_embedding, requires_grad=False)

        self.bert_subnetwork = nn.Sequential(
            nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(128), nn.ReLU(), nn.LazyLinear(tfn_latent_dim)
        )

        def _gnn_builder():
            return gnn.Sequential(
                "x, edge_index",
                [
                    (gnn_cls(gnn_input_dim, gnn_latent_dim), "x, edge_index -> x"),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    (gnn_cls(gnn_latent_dim, gnn_latent_dim), "x, edge_index -> x"),
                    nn.Linear(gnn_latent_dim, tfn_latent_dim),
                ],
            )

        self.gnns = nn.ModuleList([_gnn_builder() for _ in range(num_gnns)])

        self.decision_layer = nn.Sequential(
            nn.Dropout(dropout),
            nn.LazyLinear(128),
            nn.ReLU(),
            nn.LazyLinear(128),
            nn.ReLU(),
            nn.LazyLinear(1),
        )

        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))

    def forward(self, *relations):
        # Concatenate one vector column to each embedding
        bert_output = self.bert_subnetwork.forward(self.bert_embedding)

        gcn_outputs = [gnn.forward(relation.x, relation.edge_index) for gnn, relation in zip(self.gnns, relations)]

        post_fusion_embeddings_cat = [
            torch.cat(
                [embedding, torch.ones(embedding.shape[0], 1, device=embedding.device, requires_grad=False)], dim=1
            )
            for embedding in ([bert_output] + gcn_outputs)
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
        # Here, batch is a tuple of relations
        # We assume all relations have the same number of nodes and same labels
        label = batch[0].y 
        mask = batch[0].train_mask

        predictions = self.forward(*batch)

        # Mask train
        predictions = predictions[mask]
        label = label[mask]

        # Mask out -100 in label
        mask = label != -100
        predictions = predictions[mask]
        label = label[mask]

        loss = self.loss_fn(predictions.squeeze(), label.float())

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        label = batch[0].y
        mask = batch[0].val_mask

        predictions = self.forward(*batch)

        # Mask val
        predictions = predictions[mask]
        label = label[mask]

        # Mask out -100 in label
        mask = label != -100
        predictions = predictions[mask]
        label = label[mask]

        loss = self.loss_fn(predictions.squeeze(), label.float())

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log_metrics(predictions, label, prefix="val_")
        return loss

    def test_step(self, batch, batch_idx):
        label = batch[0].y
        mask = batch[0].test_mask

        predictions = self.forward(*batch)

        # Mask test
        predictions = predictions[mask]
        label = label[mask]

        # Mask out -100 in label
        mask = label != -100
        predictions = predictions[mask]
        label = label[mask]

        loss = self.loss_fn(predictions.squeeze(), label.float())

        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log_metrics(predictions, label, prefix="test_")
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        label = batch[0].y
        mask = batch[0].test_mask

        predictions = self.forward(*batch)

        # Mask test
        predictions = predictions[mask]
        label = label[mask]

        # Mask out -100 in label
        mask = label != -100
        predictions = predictions[mask]
        label = label[mask]

        return label, predictions.squeeze()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

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

    def configure_optimizers(self):
        """
        Configure the optimizer for training.

        Returns:
            torch.optim.Optimizer: Optimizer object.
        """
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
