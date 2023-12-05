import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from torch.nn.functional import dropout
from torch_geometric.nn import SAGEConv  # GraphSAGE 레이어 사용
from torch_geometric.nn import global_mean_pool

class LitGraphSAGE(pl.LightningModule):
    """
    LightningModule for GraphSAGE with node classification.
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
        Initializes a GraphSAGE model for node classification.

        Args:
            input_dim (int): Dimensionality of node features.
            hidden_dim (int): Dimensionality of hidden layers.
            output_dim (int): Number of output classes.
            num_layers (int): Number of hidden layers (excluding the output layer).
            lr (float): Learning rate (default: 0.01).
            weight_decay (float): Weight decay (default: 5e-4).
            pos_weight (int): Weight of positive class in the loss function.
            dropout (float): Dropout probability (default: 0.2).
        """
        super(LitGraphSAGE, self).__init__()

        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout

        # Define the GraphSAGE layers
        _modules = [SAGEConv(input_dim, hidden_dim)]
        for _ in range(num_layers - 2):
            _modules.append(SAGEConv(hidden_dim, hidden_dim))
        _modules.append(SAGEConv(hidden_dim, output_dim))
        self.layers = nn.ModuleList(_modules)

        assert isinstance(pos_weight, float), f"pos_weight should be a float, got {type(pos_weight)}"
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
        self.save_hyperparameters()

        self._is_binary_classification = output_dim == 1

    def forward(self, x, edge_index):
        """
        Forward pass of the GraphSAGE model.

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

        x = self.layers[-1](x, edge_index)

        # Global pooling layer (you can customize this based on your problem)
        x = global_mean_pool(x, torch.zeros(x.size(0), dtype=torch.long))

        return x

    # ... (이전 코드와 동일한 training, validation, test, predict_step 함수 등을 포함)

    def configure_optimizers(self):
        """
        Configure the optimizer for training.

        Returns:
            torch.optim.Optimizer: Optimizer object.
        """
        return optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
