import numpy as np
import torch
import scipy.sparse as sp
from typing import Literal

from torch.utils.data import Dataset


def load_adj_matrix(adj_matrix_path):
    """
    Load sparse adjacency matrix into tensor of (2, num_edges)

    Args:
        adj_matrix_path (str): path to adjacency matrix (in .npz format)

    Returns:
        torch.tensor: edge index tensor of shape (2, num_edges)
    """
    adj = sp.load_npz(adj_matrix_path)
    assert isinstance(adj, sp.csr_matrix), f"Adjacency matrix must be a CSR matrix, got {type(adj)}"

    edge_index = torch.tensor(adj.nonzero(), dtype=torch.long)
    return edge_index


class EmbeddingDataset(Dataset):
    def __init__(
        self,
        embeddings: list[torch.Tensor],
        labels: np.ndarray,
        split_mask: dict[str, list[bool]],
        mode: Literal["train", "val", "test"],
    ):
        super(EmbeddingDataset, self).__init__()

        assert isinstance(labels, np.ndarray), f"Labels must be a numpy array, got {type(labels)}"
        self.labels = torch.from_numpy(labels)

        assert mode in ["train", "val", "test"], f"Mode must be one of 'train', 'val', or 'test', got {mode}"
        split_mask_mode = split_mask[mode]

        assert all(
            [len(embedding) == len(embeddings[0]) for embedding in embeddings]
        ), "Embeddings must have same number of nodes"
        self.embeddings = [embedding[split_mask_mode] for embedding in embeddings]

        self.num_nodes = len(self.embeddings[0])
        self.embedding_dims = [embedding.shape[1] for embedding in embeddings]

    def __len__(self):
        return self.num_nodes

    def __getitem__(self, idx):
        return tuple(embedding[idx] for embedding in self.embeddings) + (self.labels[idx],)

    @staticmethod
    def collate_fn(batch):
        labels = torch.stack([item[-1] for item in batch])

        embeddings = [torch.stack([item[i] for item in batch]) for i in range(len(batch[0]) - 1)]

        return embeddings, labels
