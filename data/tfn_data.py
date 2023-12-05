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

