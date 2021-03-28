import torch
import torch.nn as nn
from collections import OrderedDict
from utils.training_helpers import View

class MTEX(nn.Module):
    def __init__(self, time_length, feature_length, n_classes):
        super(MTEX, self).__init__()
        self.cnn_layers = nn.Sequential(OrderedDict([
            ('conv_1', nn.Conv2d(1, 16, (time_length//2 + 1, 1))),
            ('relu_1', nn.ReLU(inplace=True)),
            ('conv_2', nn.Conv2d(16, 32, (time_length//4 + 1, 1))),
            ('relu_2', nn.ReLU(inplace=True)),
            ('conv_3', nn.Conv2d(32, 1, 1)),
            ('relu_3', nn.ReLU(inplace=True)),
            ('view', View((feature_length,time_length//4 + 2))),
            ('conv_4', nn.Conv1d(feature_length, 64, 3)),
            ('relu_4', nn.ReLU(inplace=True))
        ]))

        self.linear_layers = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(64 * (time_length//4), 32)),
            ('fc2', nn.Linear(32, n_classes))
        ]))
        
        
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0),-1)
        x = self.linear_layers(x)

        return x
