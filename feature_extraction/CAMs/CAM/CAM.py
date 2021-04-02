import numpy as np
from feature_extraction.UnitCAM import UnitCAM

# Adapt from https://github.com/zhoubolei/CAM/blob/master/pytorch_CAM.py
class CAM(UnitCAM):
    """The implementation of CAM for multivariate time series classification
    CNN-based deep learning models

    **NOTE**: CAM can only applied with models that have Global Average Pooling layer.
    If no Global Average Pooling layer exists, one has to be added and the model has
    to be retrained over.

    Based on the paper:

        Zhou, B., Khosla, A., Lapedriza,
        A., Oliva, A., & Torralba, A. (2016).
        Learning deep features for discriminative localization.
        In Proceedings of the IEEE conference on computer vision
        and pattern recognition (pp. 2921-2929).

    Implementation adapted from:

        https://github.com/zhoubolei/CAM/blob/master/pytorch_CAM.py


    This implementation is modified to only support Multivariate Time Series
    Classification data and the corresponding CNN-based models.

    """

    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        super().__init__(model, feature_module, target_layer_names, use_cuda)
        if "avgpool_layer" not in list(self.model._modules.keys()):
            raise ValueError(
                "CAM does not support model without global average pooling."
            )

    def __call__(self, input_features, index=None):
        """Implemented method when CAM is called on a given input and its targeted
        index

        Attributes:
            input_features: A multivariate data input to the model
            index: Targeted output class

        """
        features, output, index = self.extract_features(input_features, index)

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.squeeze(output.data.numpy())

        cam = np.zeros(target.shape[1:], dtype=np.float32)
        target = np.squeeze(target)

        cam = self.cam_weighted_sum(cam, weights, target)

        return cam
