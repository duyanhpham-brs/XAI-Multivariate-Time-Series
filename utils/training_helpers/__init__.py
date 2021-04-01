# pylint: disable=no-self-use
import torch
import torch.nn as nn


class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def __repr__(self):
        return f"View{self.shape}"

    def forward(self, x):
        """
        Reshapes the input according to the shape saved in the view data structure.
        """
        batch_size = x.size(0)
        shape = (batch_size, *self.shape)
        out = x.view(shape)
        return out


class Squeeze(nn.Module):
    def __repr__(self):
        return "Squeeze()"

    def forward(self, x):
        """
        Squeeze unnecessary dim.
        """
        return torch.squeeze(x)


class SwapLastDims(nn.Module):
    def __repr__(self):
        return "SwapLastDim()"

    def forward(self, x):
        """
        Swap two last dims.
        """

        batch_size = 1
        if len(x.size()) == 3:
            batch_size = x.size(0)

        shape = (batch_size, x.size(-1), x.size(-2))
        out = x.view(shape)

        return out
