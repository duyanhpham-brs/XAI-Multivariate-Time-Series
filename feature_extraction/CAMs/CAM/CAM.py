import numpy as np
from feature_extraction.UnitCAM import UnitCAM

# Adapt from https://github.com/zhoubolei/CAM/blob/master/pytorch_CAM.py
class CAM(UnitCAM):
    def __init__(self, model, feature_module, target_layer_names, use_cuda, **kwargs):
        super().__init__(model, feature_module, target_layer_names, use_cuda, **kwargs)
        if 'avgpool_layer' not in list(self.model._modules.keys()):
            raise ValueError('CAM does not support model without global average pooling')

    def __call__(self, input_features, index=None):
        features, output, index = self.extract_features(input_features, index)

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.squeeze(output.data.numpy())

        cam = np.zeros(target.shape[1:], dtype=np.float32)
        target = np.squeeze(target)

        cam = self.cam_weighted_sum(cam, weights, target)

        return cam
