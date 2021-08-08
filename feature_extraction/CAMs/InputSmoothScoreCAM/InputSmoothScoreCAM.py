import torch
from torch.nn import functional as F
import numpy as np
from feature_extraction.CAMs.ScoreCAM import ScoreCAM
from utils.gradient_extraction import upsample
from models.attention_based.helpers.train_darnn.constants import device


class InputSmoothScoreCAM(ScoreCAM):
    """The implementation of Input Smooth Score-CAM for multivariate time series classification
    CNN-based deep learning models

    Based on the paper:

        Naidu, R., & Michael, J. (2020). SS-CAM: Smoothed Score-CAM for
        sharper visual feature localization. arXiv preprint arXiv:2006.14255.

    Implementation adapted from:

        https://github.com/frgfm/torch-cam/blob/master/torchcam/cams/cam.py#L179

    This implementation is modified to only support Multivariate Time Series
    Classification data and the corresponding CNN-based models

    """

    def __init__(self, model, feature_module, target_layer_names, use_cuda, **kwargs):
        super().__init__(model, feature_module, target_layer_names, use_cuda)
        self.smooth_factor = kwargs["smooth_factor"]
        self.std = kwargs["std"]
        self._distrib = torch.distributions.normal.Normal(0, self.std)

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

        with torch.no_grad():
            scores = 0
            for _ in range(self.smooth_factor):
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
                        score_saliency_maps.append(
                            (
                                input_features
                                + self._distrib.sample(input_features.size())
                            )
                            * norm_saliency_map
                        )
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
                    score_saliency_maps.append(
                        (
                            input_features
                            + self._distrib.sample(input_features.size()).to(device)
                        )
                        * norm_saliency_map
                    )

                # how much increase if keeping the highlighted region
                # predication on masked input
                masked_input_features = torch.squeeze(
                    torch.stack(score_saliency_maps, dim=1), 0
                )
                output_ = self.model(masked_input_features)

                scores += output_[:, index] - output[0, index]

            scores.div_(self.smooth_factor)
            cam = np.zeros(self.target.shape[1:], dtype=np.float32)

        return cam, scores, output
