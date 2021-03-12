import torch.nn as nn

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = list(args)

    def forward(self, x):
        print(self.shape)
        return x.view(*self.shape)
