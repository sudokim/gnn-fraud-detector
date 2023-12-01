import pickle

import numpy as np
import scipy.sparse as sp
from scipy.io import loadmat
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import NormalizeFeatures


__all__ = ["load_data", "load_data_from_matlab"]


def _split_data_by_file(data: Data, split_file: str) -> Data:
    split_masks = pickle.load(open(split_file, "rb"))

    train_mask = split_masks["train"]
    val_mask = split_masks["val"]
    test_mask = split_masks["test"]

    train_mask = torch.Tensor(train_mask).type(torch.bool)
    val_mask = torch.Tensor(val_mask).type(torch.bool)
    test_mask = torch.Tensor(test_mask).type(torch.bool)

    data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
    data.val_mask = torch.tensor(val_mask, dtype=torch.bool)
    data.test_mask = torch.tensor(test_mask, dtype=torch.bool)

    return data


def _split_data_by_ratio(data: Data, split_ratio: tuple[float], seed: int) -> Data:
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


def _post_process_data(data: Data) -> Data:
    # Normalize node features
    data = NormalizeFeatures()(data)

    return data


def load_data(
    edge_index_path: str,
    features_path: str,
    labels_path: str,
    split_ratio: tuple[float] | None = None,
    seed: int | None = None,
    split_file: str | None = None,
) -> Data:
    # One of (split_ratio, seed) or split_file must be provided
    if split_file is None:
        assert (
            split_ratio is not None and seed is not None
        ), "Only one of (split_ratio, seed) or split_file must be provided"
    else:
        assert split_ratio is None and seed is None, "Only one of (split_ratio, seed) or split_file must be provided"

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

    data = _post_process_data(data)

    if split_file is None:
        data = _split_data_by_ratio(data, split_ratio, seed)
    else:
        data = _split_data_by_file(data, split_file)

    return data


def load_data_from_matlab(
    mat_path: str, split_ratio: tuple[float] | None = None, seed: int | None = None, split_file: str | None = None
) -> Data:
    """
    Load data from a MATLAB file.

    Args:
        mat_path (str): Path to the MATLAB file.

    Returns:
        Data: PyG Data object.
    """
    # One of (split_ratio, seed) or split_file must be provided
    if split_file is None:
        assert (
            split_ratio is not None and seed is not None
        ), "Only one of (split_ratio, seed) or split_file must be provided"
    else:
        assert split_ratio is None and seed is None, "Only one of (split_ratio, seed) or split_file must be provided"

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

    data = _post_process_data(data)

    if split_file is None:
        data = _split_data_by_ratio(data, split_ratio, seed)
    else:
        data = _split_data_by_file(data, split_file)

    return data
