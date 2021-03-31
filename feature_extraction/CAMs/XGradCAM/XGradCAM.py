import numpy as np
import torch
from torch.nn import functional as F
from feature_extraction.CAMs.GradCAM import GradCAM

# Adapt from https://github.com/Fu0511/XGrad-CAM/blob/main/XGrad-CAM.py
class XGradCAM(GradCAM):
    def __init__(self, model, feature_module, target_layer_names, use_cuda, **kwargs):
        super().__init__(model, feature_module, target_layer_names, use_cuda, **kwargs)

    def map_gradients(self, grads_val, target):
        if len(grads_val.shape) == 4:
            weights = np.sum(grads_val.numpy()[0, :] * target, axis=(1, 2))
            weights = weights / (np.sum(target, axis=(1, 2)) + 1e-6)
        elif len(grads_val.shape) == 3:
            weights = np.sum(grads_val.numpy()[0, :] * target, axis=(0, 1))
            weights = weights / (np.sum(target, axis=(0, 1)) + 1e-6)
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        return cam, weights

    def __call__(self, input_features, index=None):
        _, grads_val, target = self.calculate_gradients(input_features, index)
        
        cam, weights = self.map_gradients(grads_val, target)
        cam = self.cam_weighted_sum(cam, weights, target)

        return cam
