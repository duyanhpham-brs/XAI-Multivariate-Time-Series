from collections import OrderedDict
import torch
import torch.nn as nn
from utils.training_helpers import View, Squeeze, SwapLastDims, ExtractLastCell


class TSEM(nn.Module):
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
                    ("squeeze_12", Squeeze(True)),
                    ("swap_12", SwapLastDims()),
                ]
            )
        )

        self.rnn_layers2_b1 = nn.Sequential(
            OrderedDict(
                [
                    ("view_21", View((feature_length))),
                    (
                        "rnn_21",
                        nn.LSTM(
                            input_size=time_length,
                            hidden_size=window_size,
                            batch_first=True,
                        ),
                    ),
                    ("select_21", ExtractLastCell()),
                    ("relu_22", nn.ReLU(inplace=True)),
                    (
                        "upsample",
                        nn.Upsample(
                            scale_factor=time_length / window_size,
                            mode="linear",
                            align_corners=True,
                        ),
                    ),
                    ("softmax", nn.Softmax(dim=1)),
                ]
            )
        )

        self.cnn_layers3_tsem = nn.Sequential(
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
            OrderedDict([("avgpool", nn.AvgPool1d(feature_length))])
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
        second_branch = self.rnn_layers2_b1(x)
        # Multiplication
        main_branch = first_branch * second_branch
        main_branch = self.cnn_layers3_tsem(main_branch)
        main_branch = self.avgpool_layer(main_branch)
        main_branch = self.linear_layers(main_branch)

        return main_branch
