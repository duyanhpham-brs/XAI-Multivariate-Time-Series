import numpy as np
import torch
from feature_extraction.UnitCAM import UnitCAM


class AblationCAM(UnitCAM):
    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        super().__init__(model, feature_module, target_layer_names, use_cuda)
        self.target_layer_names = target_layer_names
        self.slope = None
        self.target = None

    def calculate_slope(self, input_features, index, print_out=True):
        features, output, index = self.extract_features(
            input_features, index, print_out
        )

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        _, output_k, _ = self.extract_features(
            input_features, index, print_out, zero_out=True
        )
        one_hot_k = np.zeros((1, output_k.size()[-1]), dtype=np.float32)
        one_hot_k[0][index] = 1
        one_hot_k = torch.from_numpy(one_hot_k).requires_grad_(True)
        if self.cuda:
            one_hot_k = torch.sum(one_hot_k.cuda() * output)
        else:
            one_hot_k = torch.sum(one_hot_k * output)

        slope = (output - output_k) / output

        return one_hot, slope, target

    def __call__(self, input_features, index=None):
        _, slope, target = self.calculate_slope(input_features, index)

        cam = np.zeros(target.shape[1:], dtype=np.float32)
        cam = self.cam_weighted_sum(cam, np.squeeze(slope.detach().numpy()), target)

        return cam
