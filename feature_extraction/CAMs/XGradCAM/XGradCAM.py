import numpy as np
from feature_extraction.CAMs.GradCAM import GradCAM


class XGradCAM(GradCAM):
    """The implementation of XGrad-CAM for multivariate time series classification
    CNN-based deep learning models

    Based on the paper:

        Fu, R., Hu, Q., Dong, X., Guo, Y., Gao, Y., & Li, B. (2020). Axiom-based
        grad-cam: Towards accurate visualization and explanation of cnns.
        arXiv preprint arXiv:2008.02312.

    Implementation adapted from:

        https://github.com/Fu0511/XGrad-CAM/blob/main/XGrad-CAM.py


    This implementation is modified to only support Multivariate Time Series
    Classification data and the corresponding CNN-based models

    """

    def map_gradients(self):
        """Caculate weights based on the gradients corresponding to the extracting layer
        via global average pooling

        Returns:
        -------
            cam: The placeholder for resulting weighted feature maps
            weights: The weights corresponding to the extracting feature maps
        """
        if len(self.grads_val.shape) == 4:
            weights = np.sum(self.grads_val.numpy()[0, :] * self.target, axis=(1, 2))
            weights = weights / (np.sum(self.target, axis=(1, 2)) + 1e-6)
        elif len(self.grads_val.shape) == 3:
            weights = np.sum(self.grads_val.numpy()[0, :] * self.target, axis=1)
            weights = weights / (np.sum(self.target, axis=(0, 1)) + 1e-6)
        cam = np.zeros(self.target.shape[1:], dtype=np.float32)

        return cam, weights

    def __call__(self, input_features, print_out, index=None):
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
        output = self.calculate_gradients(input_features, print_out, index)

        cam, weights = self.map_gradients()
        assert (
            weights.shape[0] == self.target.shape[0]
        ), "Weights and targets layer shapes are not compatible."
        cam = self.cam_weighted_sum(cam, weights, self.target)

        return cam, output[0, index].data.cpu().numpy()
