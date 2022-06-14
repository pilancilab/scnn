"""Wrapper for PyTorch layers allowing for (optional) layer-wise
regularizers."""
from typing import Optional

import torch

from scnn.private.models.regularizers import Regularizer


class LayerWrapper(torch.nn.Module):
    """Wrapper for Pytorch module which provides the opportunity for built-in
    regularization.

    Useful in the case of layer-wise regularization rather than over the whole
    network
    """

    def __init__(self, layer, regularizer: Optional[Regularizer] = None):

        super().__init__()
        self.layer = layer
        self.regularizer = regularizer

    def forward(self, X: torch.Tensor):
        return self.layer.forward(X)

    def get_regularizer(self):
        return self.regularizer

    def parameters(self):
        return torch.cat([param for param in self.layer.parameters()])
