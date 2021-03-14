import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from utils.training_helpers import View, Squeeze, SwapLastDims

class MTEXCNN(nn.Module):
    def __init__(self, time_length, feature_length, n_classes):
        super(MTEXCNN, self).__init__()
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

class XCM(nn.Module):
    def __init__(self, window_size, time_length, feature_length, n_classes):
        super(XCM, self).__init__()
        self.cnn_layers_1 = nn.Sequential(OrderedDict([
            ('conv_11', nn.Conv2d(1, 16, window_size, padding=window_size//2)),
            ('batchnorm_11', nn.BatchNorm2d(16)),
            ('relu_11', nn.ReLU(inplace=True)),
            ('conv_12', nn.Conv2d(16, 1, 1)),
            ('relu_12', nn.ReLU(inplace=True)),
            ('squeeze_12', Squeeze()),
            ('swap_12', SwapLastDims())
        ]))

        self.cnn_layers_2 = nn.Sequential(OrderedDict([
            ('view_21', View((feature_length,time_length))),
            ('conv_21', nn.Conv1d(3, 16, window_size, padding=window_size//2)),
            ('batchnorm_21', nn.BatchNorm1d(16)),
            ('relu_21', nn.ReLU(inplace=True)),
            ('conv_22', nn.Conv1d(16, 1, 1)),
            ('relu_22', nn.ReLU(inplace=True))
        ]))

        self.cnn_layers_3 = nn.Sequential(OrderedDict([
            ('swap_31', SwapLastDims()),
            ('conv_31', nn.Conv1d(time_length, 32, window_size, padding=window_size//2)),
            ('batchnorm_31', nn.BatchNorm1d(32)),
            ('relu_3', nn.ReLU(inplace=True)),
            ('avgpool', nn.AvgPool1d(feature_length + 1)),
            ('flatten', nn.Flatten()),
            ('fc1', nn.Linear(32, n_classes)),
            ('softmax', nn.Softmax(dim=1))
        ]))
        
    def forward(self, x):
        # 2d (spatial) branch
        first_branch = self.cnn_layers_1(x)
        # 1d (temporal) branch
        second_branch = self.cnn_layers_2(x)
        # Concatenation
        main_branch = torch.cat((first_branch, second_branch), 1)
        main_branch = self.cnn_layers_3(main_branch)

        return main_branch
