"""Implementation of a Gated ReLU layer in PyTorch."""

import torch
import torch.nn.functional as F


class GatedReLULayer(torch.nn.Module):

    """A gated ReLU layer with activation.

        Z = X W * max(X U, 0),

    where 'U' is a fixed, pre-specified set of gate vectors.
    """

    def __init__(self, U: torch.Tensor):
        """
        :param U: the gates vectors for the gated ReLU activation.
        """
        super().__init__()
        self.U = U

        # infer input shape and layer width from gate vectors
        self.in_features, self.out_features = self.U.shape

        # linear layer
        self.linear = torch.nn.Linear(self.in_features, self.out_features, bias=False)

    def forward(self, x: torch.Tensor, batch_size=None):
        """
        :param x: the input data.
        :returns: the activations.
        """
        # return linear combinations gated by the U vectors.
        return torch.multiply(torch.sign(F.relu(x @ self.U)), self.linear(x))
