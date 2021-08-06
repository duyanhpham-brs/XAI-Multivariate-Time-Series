import numpy as np
import torch
from feature_extraction.UnitCAM import UnitCAM


class AblationCAM(UnitCAM):
    """The implementation of Ablation-CAM for multivariate time series classification
    CNN-based deep learning models

    Based on the paper:

        Ramaswamy, H. G. (2020). Ablation-cam: Visual explanations for deep
        convolutional network via gradient-free localization.
        In Proceedings of the IEEE/CVF Winter Conference on
        Applications of Computer Vision (pp. 983-991).


    This implementation is modified to only support Multivariate Time Series
    Classification data and the corresponding CNN-based models

    """

    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        super().__init__(model, feature_module, target_layer_names, use_cuda)
        self.target_layer_names = target_layer_names
        self.slope = []
        self.target = None

    def calculate_slope(self, input_features, print_out, index):
        """Implemented method when CAM is called on a given input and its targeted
        index to calculate the slope between the feature maps and their ablation counter part

        Attributes:
        -------
            input_features: A multivariate data input to the model
            print_out: Whether to print the class with maximum likelihood when index is None
            index: Targeted output class

        """
        if index is None:
            features, output, index = self.extract_features(
                input_features, index, print_out
            )
        else:
            features, output, _ = self.extract_features(
                input_features, index, print_out
            )

        self.feature_module.zero_grad()
        self.model.zero_grad()

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output[0])
        else:
            one_hot = torch.sum(one_hot * output[0])

        self.target = features[-1]
        if len(self.target.size()) == 4:
            _, k, _, _ = self.target.size()
        elif len(self.target.size()) == 3:
            _, k, _ = self.target.size()
        self.target = self.target.cpu().data.numpy()[0, :]

        for i in range(k):
            _, output_k, _ = self.extract_features(
                input_features, int(index), print_out, zero_out=i
            )
            one_hot_k = np.zeros((1, output_k.size()[-1]), dtype=np.float32)
            one_hot_k[0][int(index)] = 1
            one_hot_k = torch.from_numpy(one_hot_k).requires_grad_(True)
            if self.cuda:
                one_hot_k = torch.sum(one_hot_k.cuda() * output_k[0])
            else:
                one_hot_k = torch.sum(one_hot_k * output_k[0])

            self.slope.append((one_hot - one_hot_k) / (one_hot + 1e-9))

        return output

    def map_slopes(self):
        """Caculate weights based on the gradients corresponding to the extracting layer
        via global average pooling

        Returns:
        -------
            cam: The placeholder for resulting weighted feature maps
            weights: The weights corresponding to the extracting feature maps
        """
        weights = torch.stack(self.slope)
        self.slope = []

        cam = np.zeros(self.target.shape[1:], dtype=np.float32)

        return cam, weights.detach().numpy()

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

        output = self.calculate_slope(input_features, print_out, index)

        cam, weights = self.map_slopes()
        assert (
            weights.shape[0] == self.target.shape[0]
        ), "Weights and targets layer shapes are not compatible."
        cam = self.cam_weighted_sum(cam, weights, self.target)

        return cam, output
