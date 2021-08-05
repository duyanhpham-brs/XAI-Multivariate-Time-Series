import numpy as np
import torch
from feature_extraction.UnitCAM import UnitCAM


class GradCAM(UnitCAM):
    """The implementation of Grad-CAM for multivariate time series classification
    CNN-based deep learning models

    Based on the paper:

        Selvaraju, R. R., Cogswell, M.,
        Das, A., Vedantam, R., Parikh,
        D., & Batra, D. (2017). Grad-cam: Visual explanations from deep networks
        via gradient-based localization. In Proceedings of the
        IEEE international conference on computer vision (pp. 618-626).

    Implementation adapted from:

        https://github.com/jacobgil/pytorch-grad-cam/blob/bf27469f5b3accf9535e04e52106e3f77f5e9cf5/gradcam.py#L31


    This implementation is modified to only support Multivariate Time Series
    Classification data and the corresponding CNN-based models

    """

    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        super().__init__(model, feature_module, target_layer_names, use_cuda)
        self.grads_val = None
        self.target = None

    def calculate_gradients(self, input_features, print_out, index):
        """Implemented method when CAM is called on a given input and its targeted
        index

        Attributes:
        -------
            input_features: A multivariate data input to the model
            print_out: Whether to print the class with maximum likelihood when index is None
            index: Targeted output class

        """
        features, output, index = self.extract_features(
            input_features, print_out, index
        )
        self.feature_module.zero_grad()
        self.model.zero_grad()

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        one_hot.backward(retain_graph=True)

        self.grads_val = self.extractor.get_gradients()[-1].cpu().data

        self.target = features[-1]
        self.target = self.target.cpu().data.numpy()[0, :]

        return output

    def map_gradients(self):
        """Caculate weights based on the gradients corresponding to the extracting layer
        via global average pooling

        Returns:
        -------
            cam: The placeholder for resulting weighted feature maps
            weights: The weights corresponding to the extracting feature maps
        """
        if len(self.grads_val.shape) == 4:
            weights = np.mean(self.grads_val.numpy(), axis=(2, 3))[0, :]
        elif len(self.grads_val.shape) == 3:
            weights = np.mean(self.grads_val.numpy(), axis=2).reshape(
                -1, self.grads_val.size(0)
            )

        cam = np.zeros(self.target.shape[1:], dtype=np.float32)

        return cam, weights

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

        output = self.calculate_gradients(input_features, print_out, index)

        cam, weights = self.map_gradients()
        assert (
            weights.shape[0] == self.target.shape[0]
        ), "Weights and targets layer shapes are not compatible."
        cam = self.cam_weighted_sum(cam, weights, self.target)

        return cam, output[0, index].data.cpu().numpy()
