import numpy as np
import torch
from torch.nn import functional as F
from feature_extraction.CAMs.GradCAM import GradCAM

# Adapt from https://github.com/adityac94/Grad_CAM_plus_plus/blob/4a9faf6ac61ef0c56e19b88d8560b81cd62c5017/misc/utils.py#L51
class GradCAMPlusPlus(GradCAM):
    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        super().__init__(model, feature_module, target_layer_names, use_cuda)
        self.alphas = None

    @staticmethod
    def compute_second_derivative(one_hot, target):
        # second_derivative
        second_derivative = torch.exp(one_hot.detach()) * target

        return second_derivative

    @staticmethod
    def compute_third_derivative(one_hot, target):
        # third_derivative
        third_derivative = torch.exp(one_hot.detach()) * target * target

        return third_derivative

    @staticmethod
    def compute_global_sum(one_hot):
        # global sum
        global_sum = np.sum(one_hot.detach().numpy(), axis=0)

        return global_sum

    @staticmethod
    def extract_higher_level_gradient(global_sum, second_derivative, third_derivative):
        alpha_num = second_derivative.numpy()
        alpha_denom = (
            second_derivative.numpy() * 2.0 + third_derivative.numpy() * global_sum
        )
        alpha_denom = np.where(
            alpha_denom != 0.0, alpha_denom, np.ones(alpha_denom.shape)
        )
        alphas = alpha_num / alpha_denom

        return alphas

    def map_gradients(self):
        if len(self.grads_val.shape) == 4:
            weights = np.sum(F.relu(self.grads_val).numpy() * self.alphas, axis=(2, 3))[
                0, :
            ]
        elif len(self.grads_val.shape) == 3:
            weights = np.sum(F.relu(self.grads_val).numpy() * self.alphas, axis=(1, 2))
        cam = np.zeros(self.target.shape[1:], dtype=np.float32)

        return cam, weights

    def __call__(self, input_features, index=None):
        one_hot, self.grads_val, self.target = self.calculate_gradients(
            input_features, index
        )
        second_derivative = self.compute_second_derivative(one_hot, self.target)
        third_derivative = self.compute_third_derivative(one_hot, self.target)
        global_sum = self.compute_global_sum(one_hot)
        self.alphas = self.extract_higher_level_gradient(
            global_sum, second_derivative, third_derivative
        )
        cam, weights = self.map_gradients()
        cam = self.cam_weighted_sum(cam, weights, self.target)

        return cam
