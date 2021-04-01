import numpy as np
import torch
from torch.nn import functional as F
from feature_extraction.CAMs.GradCAMPlusPlus import GradCAMPlusPlus

# Adapt from https://github.com/frgfm/torch-cam/blob/master/torchcam/cams/gradcam.py#L164
class SmoothGradCAMPlusPlus(GradCAMPlusPlus):
    def __init__(self, model, feature_module, target_layer_names, use_cuda, **kwargs):
        super().__init__(model, feature_module, target_layer_names, use_cuda)
        self.smooth_factor = kwargs['smooth_factor']
        self.std = kwargs['std']
        self._distrib = torch.distributions.normal.Normal(0, self.std)

    @staticmethod
    def extract_higher_level_gradient(global_sum, second_derivative, third_derivative):
        alpha_num = second_derivative.numpy()
        alpha_denom = second_derivative.numpy()*2.0 + third_derivative.numpy()*global_sum
        alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, np.ones(alpha_denom.shape))
        alphas = alpha_num/alpha_denom

        return alphas

    def __call__(self, input_features, index=None):
        grads_vals = None
        second_derivatives = None
        third_derivatives = None
        for _ in range(self.smooth_factor):
            one_hot, grads_val, target = self.calculate_gradients(input_features + \
                self._distrib.sample(input_features.size()), index, False)
            second_derivative = self.compute_second_derivative(one_hot, target)
            third_derivative = self.compute_third_derivative(one_hot, target)
            if grads_vals is None or second_derivatives is None or third_derivatives is None:
                grads_vals = grads_val
                second_derivatives = second_derivative
                third_derivatives = third_derivative
            else:
                grads_vals += grads_val
                second_derivatives += second_derivative
                third_derivatives += third_derivative

            second_derivatives = F.relu(second_derivatives)
            second_derivatives_min, second_derivatives_max = second_derivatives.min(), \
                second_derivatives.max()
            if second_derivatives_min == second_derivatives_max:
                return None
            second_derivatives = (second_derivatives - second_derivatives_min).div( \
                second_derivatives_min - second_derivatives_max).data

            third_derivatives = F.relu(third_derivatives)
            third_derivatives_min, third_derivatives_max = third_derivatives.min(), \
                third_derivatives.max()
            if third_derivatives_min == third_derivatives_max:
                return None
            third_derivatives = (third_derivatives - third_derivatives_min).div( \
                third_derivatives_min - third_derivatives_max).data

        one_hot, _, self.target = self.calculate_gradients(input_features, index)
        global_sum = self.compute_global_sum(one_hot)

        self.alphas = self.extract_higher_level_gradient(global_sum, \
            second_derivatives.div_(self.smooth_factor), \
            third_derivatives.div_(self.smooth_factor))
        self.grads_val = grads_vals.div(self.smooth_factor)

        cam, weights = self.map_gradients()
        cam = self.cam_weighted_sum(cam, weights, self.target)

        return cam
