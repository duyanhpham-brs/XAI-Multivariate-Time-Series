from collections import OrderedDict
import torch
import torch.nn as nn
import numpy as np
from utils.training_helpers import View


class MTEX(nn.Module):
    """The implementation of MTEX serial two-phase CNN-based models

    Based on the paper:

        Assaf, R., Giurgiu, I., Bagehorn, F., & Schumann, A. (2019, November).
        MTEX-CNN: Multivariate Time Series EXplanations for Predictions with
        Convolutional Neural Networks. In 2019 IEEE International Conference
        on Data Mining (ICDM) (pp. 952-957). IEEE.

    This implementation is modified to only support Multivariate Time Series
    Classification data and the corresponding CNN-based models

    """

    def __init__(self, time_length, feature_length, n_classes):
        super().__init__()
        self.cnn_layers = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv_1",
                        nn.Conv2d(1, 16, (time_length // 2 + 1, 1)),
                    ),
                    ("relu_1", nn.ReLU(inplace=True)),
                    (
                        "conv_2",
                        nn.Conv2d(16, 32, (time_length // 4 + 1, 1)),
                    ),
                    ("relu_2", nn.ReLU(inplace=True)),
                    ("conv_3", nn.Conv2d(32, 1, 1)),
                    ("relu_3", nn.ReLU(inplace=True)),
                    (
                        "view",
                        View((feature_length)),
                    ),
                    ("conv_4", nn.Conv1d(feature_length, 64, 3)),
                    ("relu_4", nn.ReLU(inplace=True)),
                ]
            )
        )

        self.linear_layers = nn.Sequential(
            OrderedDict(
                [
                    (
                        "fc1",
                        nn.Linear(
                            64
                            * int(
                                np.around(time_length / 4, decimals=0)
                                - 1 * (feature_length % 2 or time_length % 2)
                                - 1
                                * (
                                    feature_length % 2 == 0
                                    and time_length % 2 == 0
                                    and feature_length < 3
                                )
                                - 0 ** (time_length % 2)
                            ),
                            32,
                        ),
                    ),
                    ("fc2", nn.Linear(32, n_classes)),
                ]
            )
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)

        return x
