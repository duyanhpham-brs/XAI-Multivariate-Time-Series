import numpy as np
import torch
from torch.nn import functional as F
from feature_extraction.CAMs.GradCAMPlusPlus import GradCAMPlusPlus


class SmoothGradCAMPlusPlus(GradCAMPlusPlus):
    """The implementation of Smooth Grad-CAM++ for multivariate time series classification
    CNN-based deep learning models

    Based on the paper:

        Omeiza, D., Speakman, S., Cintas, C., & Weldermariam, K. (2019).
        Smooth grad-cam++: An enhanced inference level visualization technique for
        deep convolutional neural network models. arXiv preprint arXiv:1908.01224.

    Implementation adapted from:

        https://github.com/frgfm/torch-cam/blob/master/torchcam/cams/gradcam.py#L164

    This implementation is modified to only support Multivariate Time Series
    Classification data and the corresponding CNN-based models

    """

    def __init__(self, model, feature_module, target_layer_names, use_cuda, **kwargs):
        super().__init__(model, feature_module, target_layer_names, use_cuda)
        self.smooth_factor = kwargs["smooth_factor"]
        self.std = kwargs["std"]
        self._distrib = torch.distributions.normal.Normal(0, self.std)

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

        grads_vals = None
        second_derivatives = None
        third_derivatives = None
        for _ in range(self.smooth_factor):
            output = elf.calculate_gradients(
                input_features + self._distrib.sample(input_features.size()),
                print_out,
                index,
            )
            second_derivative = self.compute_second_derivative(
                self.one_hot, self.target
            )
            third_derivative = self.compute_third_derivative(self.one_hot, self.target)
            if (
                grads_vals is None
                or second_derivatives is None
                or third_derivatives is None
            ):
                grads_vals = self.grads_val
                second_derivatives = second_derivative
                third_derivatives = third_derivative
            else:
                grads_vals += self.grads_val
                second_derivatives += second_derivative
                third_derivatives += third_derivative

            second_derivatives = F.relu(second_derivatives)
            second_derivatives_min, second_derivatives_max = (
                second_derivatives.min(),
                second_derivatives.max(),
            )
            if second_derivatives_min == second_derivatives_max:
                return None
            second_derivatives = (
                (second_derivatives - second_derivatives_min)
                .div(second_derivatives_min - second_derivatives_max)
                .data
            )

            third_derivatives = F.relu(third_derivatives)
            third_derivatives_min, third_derivatives_max = (
                third_derivatives.min(),
                third_derivatives.max(),
            )
            if third_derivatives_min == third_derivatives_max:
                return None
            third_derivatives = (
                (third_derivatives - third_derivatives_min)
                .div(third_derivatives_min - third_derivatives_max)
                .data
            )

        self.calculate_gradients(input_features, print_out, index)
        global_sum = self.compute_global_sum(self.one_hot)

        self.extract_higher_level_gradient(
            global_sum,
            second_derivatives.div_(self.smooth_factor),
            third_derivatives.div_(self.smooth_factor),
        )
        self.grads_val = grads_vals.div(self.smooth_factor)

        cam, weights = self.map_gradients()
        assert (
            weights.shape[0] == self.target.shape[0]
        ), "Weights and targets layer shapes are not compatible."
        cam = self.cam_weighted_sum(cam, weights, self.target)

        return cam, output
