import numpy as np
import torch
from torch.nn import functional as F
from feature_extraction.UnitCAM import UnitCAM
from utils.gradient_extraction import upsample
from models.attention_based.helpers.train_darnn.constants import device


class ScoreCAM(UnitCAM):
    """The implementation of Score-CAM for multivariate time series classification
    CNN-based deep learning models

    Based on the paper:

        Wang, H., Wang, Z., Du, M., Yang, F., Zhang, Z., Ding, S., ... & Hu, X. (2020).
        Score-CAM: Score-weighted visual explanations for convolutional neural networks.
        In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
        Recognition Workshops (pp. 24-25).

    Implementation adapted from:

        https://github.com/haofanwang/Score-CAM/blob/master/cam/scorecam.py

    This implementation is modified to only support Multivariate Time Series
    Classification data and the corresponding CNN-based models

    """

    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        super().__init__(model, feature_module, target_layer_names, use_cuda)
        self.target = None

    def forward_saliency_map(self, input_features, print_out, index):
        """Do forward pass through the network

        Attributes:
        -------
            input_features: A multivariate data input to the model
            print_out: Whether to print the class with maximum likelihood when index is None
            index: Targeted output class

        Returns:
        -------
            activation: The feature maps
            score_saliency_map: The placeholder for the resulting saliency map
            k: The number of channels in the feature maps
            index: The targeted index
            output: The network forward pass output
        """
        _, _, h, w = input_features.size()

        features, output, index = self.extract_features(
            input_features, print_out, index
        )

        self.feature_module.zero_grad()
        self.model.zero_grad()

        activations = features[-1]
        if len(activations.size()) == 4:
            _, k, _, _ = activations.size()
            score_saliency_map = torch.zeros((1, 1, h, w))
        elif len(activations.size()) == 3:
            _, k, _ = activations.size()
            score_saliency_map = torch.zeros((1, 1, h, 1))

        if torch.cuda.is_available():
            activations = activations.cuda()
            score_saliency_map = score_saliency_map.cuda()

        return activations, score_saliency_map, k, index, output

    def compute_score_saliency_map(self, input_features, print_out, index):
        """Compute the score saliency map

        Attributes:
        -------
            input_features: A multivariate data input to the model
            print_out: Whether to print the class with maximum likelihood when index is None
            index: Targeted output class

        Returns:
        -------
            cam: The resulting weighted feature maps
            scores: Corresponding scores to the feature maps
        """
        activations, score_saliency_map, k, index, output = self.forward_saliency_map(
            input_features, print_out, index
        )
        self.target = activations[-1]
        scores = 0

        with torch.no_grad():
            score_saliency_maps = []
            for i in range(k):
                # upsampling
                if len(self.target.size()) == 3:
                    saliency_map = torch.unsqueeze(self.target[i : i + 1, :, :], 0)
                elif len(self.target.size()) == 2:
                    saliency_map = torch.unsqueeze(
                        torch.unsqueeze(self.target[i : i + 1, :], 2), 0
                    )
                if saliency_map.max() != saliency_map.min():
                    # normalize to 0-1
                    norm_saliency_map = (saliency_map - saliency_map.min()) / (
                        saliency_map.max() - saliency_map.min()
                    )
                else:
                    norm_saliency_map = saliency_map
                if input_features.shape[:-1] == norm_saliency_map.shape[:-1]:
                    score_saliency_maps.append(input_features * norm_saliency_map)
                else:
                    norm_saliency_map = (
                        torch.from_numpy(
                            upsample(
                                norm_saliency_map.squeeze().cpu().numpy(),
                                input_features.squeeze().cpu().numpy().T,
                            )
                        )
                        .unsqueeze(0)
                        .unsqueeze(0)
                    ).to(device)
                    assert input_features.shape[:-1] == norm_saliency_map.shape[:-1]
                    score_saliency_maps.append(input_features * norm_saliency_map)

            # how much increase if keeping the highlighted region
            # predication on masked input
            masked_input_features = torch.squeeze(
                torch.stack(score_saliency_maps, dim=1), 0
            )
            output_ = self.model(masked_input_features)

            scores = output_[:, index] - output[0, index]
            cam = np.zeros(self.target.shape[1:], dtype=np.float32)

        return cam, scores, output

    def __call__(self, input_features, print_out, index=None):
        """Implemented method when CAM is called on a given input and its targeted
        index

        Attributes:
        -------
            input_features: A multivariate data input to the model
            print_out: Whether to print the class with maximum likelihood when index is None
            index: Targeted output class

        Returns:
        -------
            cam: The resulting weighted feature maps
        """
        if index is not None and print_out == True:
            print_out = False

        cam, scores, output = self.compute_score_saliency_map(
            input_features, print_out, index
        )

        assert (
            scores.shape[0] == self.target.shape[0]
        ), "Weights and targets layer shapes are not compatible."
        cam = self.cam_weighted_sum(
            cam, scores.detach().cpu().numpy(), self.target.detach().cpu().numpy()
        )

        return cam, output
