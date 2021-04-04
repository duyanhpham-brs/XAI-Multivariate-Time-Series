# pylint: disable=unused-import
from collections import OrderedDict
import torch
import torch.nn as nn
from context import utils, models, feature_extraction
from utils.training_helpers import View, Squeeze, SwapLastDims


class MockSingularBranchModel(nn.Module):
    def __init__(self, time_length, n_classes):
        super().__init__()
        self.cnn_layers = nn.Sequential(
            OrderedDict(
                [
                    ("conv_1", nn.Conv2d(1, 16, (3, 1))),
                    ("relu_1", nn.ReLU(inplace=True)),
                    ("conv_2", nn.Conv2d(16, 32, (3, 1))),
                    ("relu_2", nn.ReLU(inplace=True)),
                ]
            )
        )

        self.linear_layers = nn.Sequential(
            OrderedDict(
                [
                    ("fc1", nn.Linear(32 * (time_length - 4) * 3, 32)),
                    ("fc2", nn.Linear(32, n_classes)),
                ]
            )
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)

        return x

class MockDualBranchModel(nn.Module):
    def __init__(self, time_length, feature_length, n_classes):
        super().__init__()
        self.cnn_layers1_b1 = nn.Sequential(
            OrderedDict(
                [
                    ("conv_11", nn.Conv2d(1, 16, 3, padding=1)),
                    ("relu_11", nn.ReLU(inplace=True)),
                    ("conv_12", nn.Conv2d(16, 1, 1)),
                    ("relu_12", nn.ReLU(inplace=True)),
                    ("squeeze_12", Squeeze()),
                    ("swap_12", SwapLastDims()),
                ]
            )
        )

        self.cnn_layers2_b1 = nn.Sequential(
            OrderedDict(
                [
                    ("view_21", View((feature_length, time_length))),
                    ("conv_21", nn.Conv1d(3, 16, 3, padding=1)),
                    ("relu_21", nn.ReLU(inplace=True)),
                    ("conv_22", nn.Conv1d(16, 1, 1)),
                    ("relu_22", nn.ReLU(inplace=True)),
                ]
            )
        )

        self.cnn_layers3 = nn.Sequential(
            OrderedDict(
                [
                    ("swap_31", SwapLastDims()),
                    ("conv_31", nn.Conv1d(time_length, 16, 3, padding=1)),
                    ("relu_31", nn.ReLU(inplace=True))
                ]
            )
        )

        self.avgpool_layer = nn.Sequential(
            OrderedDict([("avgpool", nn.AvgPool1d(feature_length + 1))])
        )

        self.linear_layers = nn.Sequential(
            OrderedDict(
                [
                    ("flatten", nn.Flatten()),
                    ("fc1", nn.Linear(16, n_classes)),
                    ("softmax", nn.Softmax(dim=1)),
                ]
            )
        )

    def forward(self, x):
        # 2d (spatial) branch
        first_branch = self.cnn_layers1_b1(x)
        # 1d (temporal) branch
        second_branch = self.cnn_layers2_b1(x)
        # Concatenation
        main_branch = torch.cat((first_branch, second_branch), 1)
        main_branch = self.cnn_layers3(main_branch)
        main_branch = self.avgpool_layer(main_branch)
        main_branch = self.linear_layers(main_branch)

        return main_branch
