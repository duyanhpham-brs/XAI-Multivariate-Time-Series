import numpy as np
import torch
from torch.nn import functional as F
from feature_extraction.UnitCAM import UnitCAM


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

    def forward_saliency_map(self, input_features, index, print_out=True):
        _, _, h, w = input_features.size()

        features, output, index = self.extract_features(
            input_features, index, print_out
        )

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

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

        return activations, score_saliency_map, k, index

    def reforward_saliency_map(self, input_features, norm_saliency_map):
        output_ = self.model(input_features * norm_saliency_map)

        return output_

    def compute_score_saliency_map(self, input_features, index):
        activations, score_saliency_map, k, index = self.forward_saliency_map(
            input_features, index
        )
        with torch.no_grad():
            for i in range(k):
                # upsampling
                if len(activations.size()) == 4:
                    saliency_map = torch.unsqueeze(activations[:, i, :, :], 1)
                elif len(activations.size()) == 3:
                    saliency_map = torch.unsqueeze(
                        torch.unsqueeze(activations[:, i, :], 2), 0
                    )

                if saliency_map.max() == saliency_map.min():
                    continue

                # normalize to 0-1
                norm_saliency_map = (saliency_map - saliency_map.min()) / (
                    saliency_map.max() - saliency_map.min()
                )

                # how much increase if keeping the highlighted region
                # predication on masked input
                output_ = self.reforward_saliency_map(input_features, norm_saliency_map)
                output_ = F.softmax(output_, dim=1)
                score = output_[0][index]

                score_saliency_map_temp = score * saliency_map
                score_saliency_map += score_saliency_map_temp

        score_saliency_map = F.relu(score_saliency_map)
        score_saliency_map_min, score_saliency_map_max = (
            score_saliency_map.min(),
            score_saliency_map.max(),
        )

        if score_saliency_map_min == score_saliency_map_max:
            return None

        score_saliency_map = (
            (score_saliency_map - score_saliency_map_min)
            .div(score_saliency_map_max - score_saliency_map_min)
            .data
        )

        return score_saliency_map

    def __call__(self, input_features, index=None):
        score_saliency_map = self.compute_score_saliency_map(input_features, index)

        return score_saliency_map
