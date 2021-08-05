from collections import OrderedDict
import torch
import numpy as np
from feature_extraction.UnitCAM import UnitCAM
from utils.training import train_model
from utils.datasets import DatasetLoader
from utils.training_helpers import SwapLastDims, Squeeze


class CAM(UnitCAM):
    """The implementation of CAM for multivariate time series classification
    CNN-based deep learning models

    Attributes:
    -------
        model: The wanna-be explained deep learning model for
            multivariate time series classification
        feature_module: The wanna-be explained module group (e.g. linear_layers)
        target_layer_names: The wanna-be explained module
        use_cuda: Whether to use cuda
        has_gap: True if the model has GAP layer right after
            the being explained CNN layer

    :NOTE:
    -------
    CAM can only applied with models that have Global Average Pooling
    layer. If no Global Average Pooling layer exists, one has to be added
    and the model has to be retrained over. Please state whether your model
    has a Global Average Pooling layer right after the being explained CNN
    layer by setting "has_gap = True" at class initiation.

    Based on the paper:
    -------

        Zhou, B., Khosla, A., Lapedriza,
        A., Oliva, A., & Torralba, A. (2016).
        Learning deep features for discriminative localization.
        In Proceedings of the IEEE conference on computer vision
        and pattern recognition (pp. 2921-2929).

    Implementation adapted from:
    -------

        https://github.com/zhoubolei/CAM/blob/master/pytorch_CAM.py


    This implementation is modified to only support Multivariate Time Series
    Classification data and the corresponding CNN-based models.

    """

    def __init__(self, model, feature_module, target_layer_names, use_cuda, **kwargs):
        super().__init__(model, feature_module, target_layer_names, use_cuda)
        self.has_gap = kwargs["has_gap"]

    def __call__(self, input_features, print_out, index=None, dataset_path=None):
        """Implemented method when CAM is called on a given input and its targeted
        index

        Attributes:
        -------
            input_features: A multivariate data input to the model
            print_out: Whether to print the class with maximum likelihood when index is None
            index: Targeted output class
            dataset_path: Path of the dataset (the same one that has been used to train)
                to retrain the new model (if it does not have GAP right after the explaining conv)

        Returns:
        -------
            cam: The resulting weighted feature maps
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if index is not None and print_out == True:
            print_out = False

        if not self.has_gap:
            if dataset_path is None:
                raise AttributeError(
                    "Dataset path is not defined for retraining the new model"
                )

            for param in self.model.parameters():
                param.requires_grad = False

            if (
                "fc"
                not in list(
                    dict(self.model._modules["linear_layers"].named_children()).keys()
                )[-1]
            ):
                n_classes = self.model._modules["linear_layers"][-2].out_features
            else:
                n_classes = self.model._modules["linear_layers"][-1].out_features

            new_cnn_layer_list = []
            for idx, layer in enumerate(self.feature_module):
                new_cnn_layer_list.append(
                    (
                        list(dict(self.feature_module.named_children()).keys())[idx],
                        layer,
                    )
                )
                if (
                    list(dict(self.feature_module.named_children()).keys())[idx]
                    == self.target_layer_names[0]
                ):
                    out_channels = layer.out_channels
                    break

            new_cnn_layers = OrderedDict(new_cnn_layer_list)

            class TargetedModel(torch.nn.Module):
                def __init__(self, n_classes, out_channels):
                    super().__init__()
                    self.cnn_layers = torch.nn.Sequential(new_cnn_layers)

                    self.linear_layers_1d = torch.nn.Sequential(
                        OrderedDict(
                            [
                                ("avg_pool", torch.nn.AdaptiveAvgPool1d(1)),
                                ("view", SwapLastDims()),
                                ("fc1", torch.nn.Linear(out_channels, n_classes)),
                            ]
                        )
                    )

                    self.linear_layers_2d = torch.nn.Sequential(
                        OrderedDict(
                            [
                                ("avg_pool", torch.nn.AdaptiveAvgPool2d(1)),
                                ("squeeze", Squeeze()),
                                ("fc1", torch.nn.Linear(out_channels, n_classes)),
                            ]
                        )
                    )

                def forward(self, x):
                    x = self.cnn_layers(x)
                    if len(x.size()) == 4:
                        x = self.linear_layers_2d(x)
                    else:
                        x = self.linear_layers_1d(x)
                    x = torch.squeeze(x)

                    return x

            new_model = TargetedModel(n_classes, out_channels).to(device)

            for param in new_model._modules["linear_layers_1d"].parameters():
                param.requires_grad = True

            for param in new_model._modules["linear_layers_2d"].parameters():
                param.requires_grad = True

            dataset = DatasetLoader(dataset_path)
            dataloaders, datasets_size = dataset.get_torch_dataset_loader_auto(4, 4)

            criterion = torch.nn.CrossEntropyLoss()
            optimizer_ft = torch.optim.Adam(new_model.parameters(), lr=1.5e-4)
            exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer_ft, step_size=10, gamma=0.1
            )

            train_model(
                new_model,
                criterion,
                optimizer_ft,
                exp_lr_scheduler,
                dataloaders,
                datasets_size,
                10,
            )

            features, output, index = self.extract_features(
                input_features, print_out, index
            )
            print(index, output[0, index].data.cpu().numpy())

            target = features[-1]
            target = target.cpu().data.numpy()[0, :]

            try:
                print(
                    new_model._modules["linear_layers_1d"][-1]
                    .weight.detach()
                    .cpu()
                    .numpy()
                    .shape
                )
                weights = (
                    new_model._modules["linear_layers_1d"][-1]
                    .weight.detach()
                    .cpu()
                    .numpy()[index, :]
                )
            except AttributeError:
                print(
                    new_model._modules["linear_layers_1d"][-2]
                    .weight.detach()
                    .cpu()
                    .numpy()
                    .shape
                )
                weights = (
                    new_model._modules["linear_layers_1d"][-2]
                    .weight.detach()
                    .cpu()
                    .numpy()[index, :]
                )
            except KeyError:
                try:
                    print(
                        new_model._modules["linear_layers_2d"][-1]
                        .weight.detach()
                        .cpu()
                        .numpy()
                        .shape
                    )
                    weights = (
                        new_model._modules["linear_layers_2d"][-1]
                        .weight.detach()
                        .cpu()
                        .numpy()[index, :]
                    )
                except AttributeError:
                    print(
                        new_model._modules["linear_layers_2d"][-2]
                        .weight.detach()
                        .cpu()
                        .numpy()
                        .shape
                    )
                    weights = (
                        new_model._modules["linear_layers_2d"][-2]
                        .weight.detach()
                        .numpy()[index, :]
                    )

            cam = np.zeros(target.shape[1:], dtype=np.float32)
            target = np.squeeze(target)
            weights = np.squeeze(weights).T

            # assert (
            #     weights.shape[0] == target.shape[0]
            # ), "Weights and targets layer shapes are not compatible."
            cam = self.cam_weighted_sum(cam, weights, target, ReLU=False)

            return cam

        features, _, index = self.extract_features(input_features, print_out, index)

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        try:
            weights = (
                new_model._modules["linear_layers"][-1]
                .weight.detach()
                .cpu()
                .numpy()[:, index]
            )
        except AttributeError:
            weights = (
                new_model._modules["linear_layers"][-2]
                .weight.detach()
                .cpu()
                .numpy()[:, index]
            )

        cam = np.zeros(target.shape[1:], dtype=np.float32)
        target = np.squeeze(target)
        weights = np.squeeze(weights).T

        assert (
            weights.shape[0] == target.shape[0]
        ), "Weights and targets layer shapes are not compatible."
        cam = self.cam_weighted_sum(cam, weights, target, ReLU=False)

        return cam, output[0, index].data.cpu().numpy()
