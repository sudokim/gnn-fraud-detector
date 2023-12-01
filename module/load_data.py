import pickle

import numpy as np
import scipy.sparse as sp
from scipy.io import loadmat
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import NormalizeFeatures


def _post_process_data(data: Data, split_ratio: tuple[float], seed: int) -> Data:
    # Normalize node features
    data = NormalizeFeatures()(data)

    # Split data
    assert isinstance(split_ratio, tuple), f"Split ratio should be a tuple, got {type(split_ratio)}"
    assert len(split_ratio) == 3, "Split ratio should be a tuple of length 3"
    assert sum(split_ratio) == 1, "Split ratio should sum to 1"
    train_ratio, val_ratio, test_ratio = split_ratio

    np.random.seed(seed)

    num_nodes = data.num_nodes

    indices = np.arange(num_nodes)
    np.random.shuffle(indices)

    train_end = int(train_ratio * num_nodes)
    val_end = int(val_ratio * num_nodes) + train_end

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    data.train_mask[train_indices] = True
    data.val_mask[val_indices] = True
    data.test_mask[test_indices] = True

    return data


def load_data(edge_index_path: str, features_path: str, labels_path: str, split_ratio: tuple[float], seed: int) -> Data:
    # Sample adjacency matrix
    edge_index = sp.load_npz(edge_index_path)
    assert isinstance(edge_index, sp.csr_matrix), f"Adjacency matrix should be a CSR matrix, got {type(edge_index)}"
    # Convert to COO tensor
    edge_index = sp.coo_matrix(edge_index)
    edge_index = torch.tensor(edge_index.nonzero(), dtype=torch.long)

    # Sample node features
    x = np.load(features_path)
    assert isinstance(x, np.ndarray), f"Node features should be a NumPy array, got {type(x)}"
    assert len(x.shape) == 2, "Node features should be a 2D array"
    x = torch.tensor(x, dtype=torch.float)

    # Sample node labels
    y = pickle.load(open(labels_path, "rb"))
    if isinstance(y, list):
        y = torch.tensor(y)
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y)
    elif not isinstance(y, torch.Tensor):
        raise AssertionError(f"Node labels should be either a list, NumPy array, or PyTorch tensor, got {type(y)}")
    y = y.squeeze()
    if not len(y.shape) == 1:
        raise AssertionError(f"Node labels should be a 1D array, got {len(y.shape)}D array")
    y = y.type(torch.float)

    # Create a PyG Data object
    data = Data(x=x, edge_index=edge_index, y=y)

    return _post_process_data(data, split_ratio, seed)


def load_data_from_matlab(mat_path: str, split_ratio: tuple[float], seed: int) -> Data:
    """
    Load data from a MATLAB file.

    Args:
        mat_path (str): Path to the MATLAB file.

    Returns:
        Data: PyG Data object.
    """
    mat = loadmat(mat_path)

    edge_index = mat["homo"]
    x = mat["features"]
    y = mat["label"]

    edge_index = sp.coo_matrix(edge_index)
    edge_index = torch.tensor(edge_index.nonzero(), dtype=torch.long)

    y = torch.from_numpy(y).squeeze()
    y = y.type(torch.float)

    x = torch.from_numpy(x.todense()).type(torch.float)

    data = Data(x=x, edge_index=edge_index, y=y)

    return _post_process_data(data, split_ratio, seed)
