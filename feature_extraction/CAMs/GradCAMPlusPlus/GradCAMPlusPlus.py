import numpy as np
import torch
from torch.nn import functional as F
from feature_extraction.CAMs.GradCAM import GradCAM


class GradCAMPlusPlus(GradCAM):
    """The implementation of Grad-CAM++ for multivariate time series classification
    CNN-based deep learning models

    Based on the paper:

        Chattopadhay, A., Sarkar, A., Howlader, P., & Balasubramanian, V. N.
        (2018, March). Grad-cam++: Generalized gradient-based visual explanations
        for deep convolutional networks.
        In 2018 IEEE Winter Conference on Applications of Computer Vision (WACV)
        (pp. 839-847). IEEE.

    Implementation adapted from:

        https://github.com/adityac94/Grad_CAM_plus_plus/blob/4a9faf6ac61ef0c56e19b88d8560b81cd62c5017/misc/utils.py#L51


    This implementation is modified to only support Multivariate Time Series
    Classification data and the corresponding CNN-based models

    """

    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        super().__init__(model, feature_module, target_layer_names, use_cuda)
        self.alphas = None
        self.one_hot = None

    @staticmethod
    def compute_second_derivative(one_hot, target):
        """Second Derivative

        Attributes:
        -------
            one_hot: Targeted index output
            target: Targeted module output

        Returns:
        -------
            second_derivative: The second derivative of the output

        """
        second_derivative = torch.exp(one_hot.detach()) * target

        return second_derivative

    @staticmethod
    def compute_third_derivative(one_hot, target):
        """Third Derivative

        Attributes:
        -------
            one_hot: Targeted index output
            target: Targeted module output

        Returns:
        -------
            third_derivative: The third derivative of the output

        """
        third_derivative = torch.exp(one_hot.detach()) * target * target

        return third_derivative

    @staticmethod
    def compute_global_sum(one_hot):
        """Global Sum

        Attributes:
        -------
            one_hot: Targeted index output

        Returns:
        -------
            global_sum: Collapsed sum from the input

        """

        global_sum = np.sum(one_hot.detach().numpy(), axis=0)

        return global_sum

    def extract_higher_level_gradient(
        self, global_sum, second_derivative, third_derivative
    ):
        """Alpha calculation

        Calculate alpha based on high derivatives and global sum

        Attributes:
        -------
            global_sum: Collapsed sum from the input
            second_derivative: The second derivative of the output
            third_derivative: The third derivative of the output

        """
        alpha_num = second_derivative.numpy()
        alpha_denom = (
            second_derivative.numpy() * 2.0 + third_derivative.numpy() * global_sum
        )
        alpha_denom = np.where(
            alpha_denom != 0.0, alpha_denom, np.ones(alpha_denom.shape)
        )
        self.alphas = alpha_num / alpha_denom

    def calculate_gradients(self, input_features, index):
        """Implemented method when CAM is called on a given input and its targeted
        index

        Attributes:
        -------
            input_features: A multivariate data input to the model
            index: Targeted output class
            print_out: Whether to print the class with maximum likelihood when index is None

        """
        features, output, index = self.extract_features(input_features, index)
        self.feature_module.zero_grad()
        self.model.zero_grad()

        self.one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        self.one_hot[0][index] = 1
        self.one_hot = torch.from_numpy(self.one_hot).requires_grad_(True)
        if self.cuda:
            self.one_hot = torch.sum(self.one_hot.cuda() * output)
        else:
            self.one_hot = torch.sum(self.one_hot * output)

        self.one_hot.backward(retain_graph=True)

        self.grads_val = self.extractor.get_gradients()[-1].cpu().data

        self.target = features[-1]
        self.target = self.target.cpu().data.numpy()[0, :]

    def map_gradients(self):
        """Caculate weights based on the gradients corresponding to the extracting layer
        via global average pooling

        Returns:
        -------
            cam: The placeholder for resulting weighted feature maps
            weights: The weights corresponding to the extracting feature maps
        """
        if len(self.grads_val.shape) == 4:
            weights = np.sum(F.relu(self.grads_val).numpy() * self.alphas, axis=(2, 3))[
                0, :
            ]
        elif len(self.grads_val.shape) == 3:
            weights = np.sum(
                F.relu(self.grads_val).numpy() * self.alphas, axis=2
            ).reshape(-1, self.grads_val.size(0))
        cam = np.zeros(self.target.shape[1:], dtype=np.float32)

        return cam, weights

    def __call__(self, input_features, index=None):
        """Implemented method when CAM is called on a given input and its targeted
        index

        Attributes:
        -------
            input_features: A multivariate data input to the model
            index: Targeted output class

        Returns:
        -------
            cam: The resulting weighted feature maps
        """
        self.calculate_gradients(input_features, index)
        second_derivative = self.compute_second_derivative(self.one_hot, self.target)
        third_derivative = self.compute_third_derivative(self.one_hot, self.target)
        global_sum = self.compute_global_sum(self.one_hot)
        self.extract_higher_level_gradient(
            global_sum, second_derivative, third_derivative
        )
        cam, weights = self.map_gradients()
        assert (
            weights.shape[0] == self.target.shape[0]
        ), "Weights and targets layer shapes are not compatible."
        cam = self.cam_weighted_sum(cam, weights, self.target)

        return cam
