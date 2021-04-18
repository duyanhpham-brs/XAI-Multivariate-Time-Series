import torch
from torch.nn import functional as F
import numpy as np
from feature_extraction.CAMs.ScoreCAM import ScoreCAM


class IntegratedScoreCAM(ScoreCAM):
    """The implementation of Integrated Score-CAM for multivariate time series classification
    CNN-based deep learning models

    Based on the paper:

        Naidu, R., Ghosh, A., Maurya, Y., & Kundu, S. S. (2020).
        IS-CAM: Integrated Score-CAM for axiomatic-based explanations.
        arXiv preprint arXiv:2010.03023.

    Implementation adapted from:

        https://github.com/frgfm/torch-cam/blob/master/torchcam/cams/cam.py#L291

    This implementation is modified to only support Multivariate Time Series
    Classification data and the corresponding CNN-based models

    """

    def __init__(self, model, feature_module, target_layer_names, use_cuda, **kwargs):
        super().__init__(model, feature_module, target_layer_names, use_cuda)
        self.smooth_factor = kwargs["smooth_factor"]

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
            for idx in range(self.smooth_factor):
                score_saliency_maps = []
                for i in range(k):
                    # upsampling
                    if len(self.target.size()) == 3:
                        saliency_map = torch.unsqueeze(self.target[i : i + 1, :, :], 0)
                    elif len(self.target.size()) == 2:
                        saliency_map = torch.unsqueeze(
                            torch.unsqueeze(self.target[i : i + 1, :], 2), 0
                        )

                    if saliency_map.max() == saliency_map.min():
                        continue

                    # normalize to 0-1
                    norm_saliency_map = (saliency_map - saliency_map.min()) / (
                        saliency_map.max() - saliency_map.min()
                    )

                    assert input_features.shape[:-1] == norm_saliency_map.size()[:-1]
                    score_saliency_maps.append(
                        ((idx + 1) / self.smooth_factor)
                        * input_features
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

        return cam, scores
