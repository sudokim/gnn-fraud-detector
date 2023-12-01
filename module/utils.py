import torch
from sklearn.metrics import confusion_matrix


def print_confusion_matrix(y_true, y_pred):
    assert isinstance(y_true, torch.Tensor), f"y_true should be a PyTorch tensor, got {type(y_true)}"
    assert isinstance(y_pred, torch.Tensor), f"y_pred should be a PyTorch tensor, got {type(y_pred)}"

    assert (
        y_true.shape == y_pred.shape
    ), f"y_true and y_pred should have the same shape, got {y_true.shape} and {y_pred.shape}"
    assert (
        len(y_true.shape) == 1
    ), f"y_true and y_pred should be 1-dimensional, got {len(y_true.shape)} and {len(y_pred.shape)}"

    TN, FP, FN, TP = confusion_matrix(y_true, y_pred).ravel()

    output = "Confusion Matrix:\n"
    output += f"{'':<13}{'Predicted Benign':^17}{'Predicted Fraud':^17}\n"
    output += f"{'Actual Benign':<13}{TN:^17}{FP:^17}\n"
    output += f"{'Actual Fraud ':<13}{FN:^17}{TP:^17}\n"

    print(output)

    return output
