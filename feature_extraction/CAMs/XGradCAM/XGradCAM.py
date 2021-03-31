import numpy as np
from feature_extraction.CAMs.GradCAM import GradCAM

# Adapt from https://github.com/Fu0511/XGrad-CAM/blob/main/XGrad-CAM.py
class XGradCAM(GradCAM):
    def map_gradients(self):
        if len(self.grads_val.shape) == 4:
            weights = np.sum(self.grads_val.numpy()[0, :] * self.target, axis=(1, 2))
            weights = weights / (np.sum(self.target, axis=(1, 2)) + 1e-6)
        elif len(self.grads_val.shape) == 3:
            weights = np.sum(self.grads_val.numpy()[0, :] * self.target, axis=(0, 1))
            weights = weights / (np.sum(self.target, axis=(0, 1)) + 1e-6)
        cam = np.zeros(self.target.shape[1:], dtype=np.float32)

        return cam, weights

    def __call__(self, input_features, index=None):
        _, self.grads_val, self.target = self.calculate_gradients(input_features, index)

        cam, weights = self.map_gradients()
        cam = self.cam_weighted_sum(cam, weights, self.target)

        return cam
