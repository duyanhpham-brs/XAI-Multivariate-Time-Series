import numpy as np
import torch
from feature_extraction.UnitCAM import UnitCAM

# Adapt from https://github.com/jacobgil/pytorch-grad-cam/blob/bf27469f5b3accf9535e04e52106e3f77f5e9cf5/gradcam.py#L31
class GradCAM(UnitCAM):
    def __init__(self, model, feature_module, target_layer_names, use_cuda, **kwargs):
        super().__init__(model, feature_module, target_layer_names, use_cuda, **kwargs)
    
    def calculate_gradients(self, input_features, index, print_out=True):
        features, output, index = self.extract_features(input_features, index, print_out)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        return one_hot, grads_val, target

    def map_gradients(self, grads_val, target):
        if len(grads_val.shape) == 4:
            weights = np.mean(grads_val.numpy(), axis=(2, 3))[0, :]
        elif len(grads_val.shape) == 3:
            weights = np.mean(grads_val.numpy(), axis=(1, 2))
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        return cam, weights

    def __call__(self, input_features, index=None):
        _, grads_val, target = self.calculate_gradients(input_features, index)
        
        cam, weights = self.map_gradients(grads_val, target)
        cam = self.cam_weighted_sum(cam, weights, target)

        return cam
