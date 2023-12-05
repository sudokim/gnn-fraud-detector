"""
Tensor Fusion Network (TFN) model from "Tensor Fusion Network for Multimodal Sentiment Analysis"
"""
from sklearn.metrics import f1_score, recall_score, roc_auc_score
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch_geometric.nn as gnn


class LitTFN(pl.LightningModule):
    """
    TFN with frozen embeddings.
    """

    def __init__(
        self,
        latent_dim,
        gcn_input_dim,
        gcn_latent_dim,
        bert_embedding,
        lr=0.01,
        weight_decay=5e-4,
        dropout=0.1,
        pos_weight=1,
    ):
        super(LitTFN, self).__init__()

        self.latent_dim = latent_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.bert_embedding = nn.Parameter(bert_embedding, requires_grad=False)

        self.bert_layer = nn.Sequential(
            nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(128), nn.ReLU(), nn.LazyLinear(latent_dim)
        )
        self.gcn_upu = gnn.Sequential(
            "x, edge_index",
            [
                (gnn.GCNConv(gcn_input_dim, gcn_latent_dim), "x, edge_index -> x"),
                nn.ReLU(),
                nn.Dropout(dropout),
                (gnn.GCNConv(gcn_latent_dim, gcn_latent_dim), "x, edge_index -> x"),
                nn.Linear(gcn_latent_dim, latent_dim),
            ],
        )
        self.gcn_usu = gnn.Sequential(
            "x, edge_index",
            [
                (gnn.GCNConv(gcn_input_dim, gcn_latent_dim), "x, edge_index -> x"),
                nn.ReLU(),
                nn.Dropout(dropout),
                (gnn.GCNConv(gcn_latent_dim, gcn_latent_dim), "x, edge_index -> x"),
                nn.Linear(gcn_latent_dim, latent_dim),
            ],
        )
        self.gcn_uvu = gnn.Sequential(
            "x, edge_index",
            [
                (gnn.GCNConv(gcn_input_dim, gcn_latent_dim), "x, edge_index -> x"),
                nn.ReLU(),
                nn.Dropout(dropout),
                (gnn.GCNConv(gcn_latent_dim, gcn_latent_dim), "x, edge_index -> x"),
                nn.Linear(gcn_latent_dim, latent_dim),
            ],
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

    def forward(self, upu, usu, uvu):
        # Concatenate one vector column to each embedding
        bert_output = self.bert_layer.forward(self.bert_embedding)

        gcn_upu_output = self.gcn_upu.forward(upu.x, upu.edge_index)
        gcn_usu_output = self.gcn_usu.forward(usu.x, usu.edge_index)
        gcn_uvu_output = self.gcn_uvu.forward(uvu.x, uvu.edge_index)

        post_fusion_embeddings_cat = [
            torch.cat(
                [embedding, torch.ones(embedding.shape[0], 1, device=embedding.device, requires_grad=False)], dim=1
            )
            for embedding in [bert_output, gcn_upu_output, gcn_usu_output, gcn_uvu_output]
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
        upu, usu, uvu = batch
        label = upu.y
        
        predictions = self.forward(upu, usu, uvu)

        # Mask train
        mask = upu.train_mask
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
        upu, usu, uvu = batch
        label = upu.y
        
        predictions = self.forward(upu, usu, uvu)

        # Mask val
        mask = upu.val_mask
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
        upu, usu, uvu = batch
        label = upu.y
        
        predictions = self.forward(upu, usu, uvu)

        # Mask test
        mask = upu.test_mask
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
        upu, usu, uvu = batch
        label = upu.y
        
        predictions = self.forward(upu, usu, uvu)

        # Mask test
        mask = upu.test_mask
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
