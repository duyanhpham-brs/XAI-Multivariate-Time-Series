# pylint: disable=no-self-use
import torch
import torch.nn as nn


class View(nn.Module):
    def __init__(self, feature_length):
        super().__init__()
        self.feature_length = feature_length

    def __repr__(self):
        return f"View{self.feature_length}"

    def forward(self, x):
        """
        Reshapes the input according to the shape saved in the view data structure.
        """
        batch_size = x.size(0)
        shape = (batch_size, self.feature_length, -1)
        out = x.reshape(shape)
        return out


class Squeeze(nn.Module):
    def __repr__(self):
        return "Squeeze()"

    def forward(self, x):
        """
        Squeeze unnecessary dim.
        """
        batch_size = x.size(0)
        x = torch.squeeze(x, 1)
        if len(x.size()) == 2:
            x = x.view((batch_size, -1))
        elif len(x.size()) == 1:
            x = x.view((batch_size, 1))
        return x


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


class ExtractLastCell(nn.Module):
    def __repr__(self):
        return "ExtractLastCell()"

    def forward(self, x):
        """
        Extract last cell for RNN
        """
        out, _ = x
        return out.detach()[:, -1:, :]
