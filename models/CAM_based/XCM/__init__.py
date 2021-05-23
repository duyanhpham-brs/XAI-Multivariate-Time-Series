from collections import OrderedDict
import torch
import torch.nn as nn
from utils.training_helpers import View, Squeeze, SwapLastDims


class XCM(nn.Module):
    """The implementation of XCM serial two-phase CNN-based models

    Based on the paper:

        Fauvel, K., Lin, T., Masson, V., Fromont, É., & Termier, A. (2020).
        XCM: An Explainable Convolutional Neural Network for Multivariate Time Series
        Classification. arXiv preprint arXiv:2009.04796.

    """

    def __init__(self, window_size, time_length, feature_length, n_classes):
        super().__init__()
        self.cnn_layers1_b1 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv_11",
                        nn.Conv2d(1, 16, window_size, padding=window_size // 2),
                    ),
                    ("batchnorm_11", nn.BatchNorm2d(16)),
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
                    (
                        "conv_21",
                        nn.Conv1d(
                            feature_length, 16, window_size, padding=window_size // 2
                        ),
                    ),
                    ("batchnorm_21", nn.BatchNorm1d(16)),
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
                    (
                        "conv_31",
                        nn.Conv1d(
                            time_length, 32, window_size, padding=window_size // 2
                        ),
                    ),
                    ("batchnorm_31", nn.BatchNorm1d(32)),
                    ("relu_3", nn.ReLU(inplace=True)),
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
                    ("fc1", nn.Linear(32, n_classes)),
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
