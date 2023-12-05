import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self, layers: list[int]):
        """
        Initializes an autoencoder.

        Args:
            layers (list[int]): List of layer sizes.
        """
        super(AutoEncoder, self).__init__()

        self._layers = layers

        _encoder = []
        for i in range(len(layers) - 1):
            _encoder.append(nn.Linear(layers[i], layers[i + 1]))
            _encoder.append(nn.ReLU())

        self.encoder = nn.Sequential(*_encoder[:-1])

        _decoder = []
        for i in range(len(layers) - 1, 0, -1):
            _decoder.append(nn.Linear(layers[i], layers[i - 1]))
            _decoder.append(nn.ReLU())

        self.decoder = nn.Sequential(*_decoder[:-1])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the autoencoder.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Encoded and decoded tensors.
        """
        enc = self.encoder(x)
        dec = self.decoder(enc)

        return enc, dec
