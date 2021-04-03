import torch
import numpy as np
from feature_extraction.UnitCAM import UnitCAM


class CAM(UnitCAM):
    """The implementation of CAM for multivariate time series classification
    CNN-based deep learning models

    Attributes:
    -------
        model: The wanna-be explained deep learning model for \\
            multivariate time series classification
        feature_module: The wanna-be explained module group (e.g. linear_layers)
        target_layer_names: The wanna-be explained module
        use_cuda: Whether to use cuda
        has_gap: True if the model has GAP layer right after \\
            the being explained CNN layer

    :NOTE:
    -------
    CAM can only applied with models that have Global Average Pooling
    layer. If no Global Average Pooling layer exists, one has to be added 
    and the model has to be retrained over. Please state whether your model 
    has a Global Average Pooling layer right after the being explained CNN 
    layer by setting "has_gap = True" at class initiation.

    Based on the paper:
    -------

        Zhou, B., Khosla, A., Lapedriza,
        A., Oliva, A., & Torralba, A. (2016).
        Learning deep features for discriminative localization.
        In Proceedings of the IEEE conference on computer vision
        and pattern recognition (pp. 2921-2929).

    Implementation adapted from:
    -------

        https://github.com/zhoubolei/CAM/blob/master/pytorch_CAM.py


    This implementation is modified to only support Multivariate Time Series
    Classification data and the corresponding CNN-based models.

    """

    def __init__(self, model, feature_module, target_layer_names, use_cuda, **kwargs):
        super().__init__(model, feature_module, target_layer_names, use_cuda)
        self.has_gap = kwargs["has_gap"]

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
        features, output, index = self.extract_features(input_features, index)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)

        print(one_hot)

        if self.cuda:
            one_hot = one_hot.cuda() * output
        else:
            one_hot = one_hot * output

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        print(target.shape)

        weights = np.squeeze(one_hot.data.numpy())

        print(weights)

        cam = np.zeros(target.shape[1:], dtype=np.float32)
        target = np.squeeze(target)

        cam = self.cam_weighted_sum(cam, weights, target)

        return cam
