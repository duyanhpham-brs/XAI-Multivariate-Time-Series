# pylint: disable=redefined-outer-name
import pytest
import torch
import torch.nn.functional as F
import numpy as np
from utils.visualization import CAMFeatureMaps

class mockCAM:
    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        self.model = model
        self.feature_module = feature_module
        self.target_layer_names = target_layer_names
        self.cuda = use_cuda

    def __call__(self, input_features, index=None):
        if len(input_features.size()) == 4:
            with torch.no_grad():
                input_features.squeeze_(0)
        input_features = input_features[:,2:]
        output = F.avg_pool1d(input_features, 2, 2)

        return output.detach().numpy()

@pytest.fixture()
def set_up():
    CAM_model = mockCAM
    feature_maps = CAMFeatureMaps(CAM_model)

    return feature_maps

def test_load_function(set_up):
    feature_maps = set_up
    feature_maps.load(None, None, None, None)
    inputs = torch.zeros((1,16,16))
    expected = torch.zeros((1,14,8))

    assert feature_maps.cam(inputs).shape == expected.size()

def test_show_function_with_upsampling(set_up):
    inputs = np.zeros((16,16))
    feature_maps = set_up
    feature_maps.load(None, None, None, None)
    outputs = feature_maps.show(inputs, None)

    assert outputs.shape[0] == inputs.shape[0]

def test_show_function_without_upsampling(set_up):
    inputs = np.zeros((16,16))
    feature_maps = set_up
    feature_maps.load(None, None, None, None)
    outputs = feature_maps.show(inputs, None, False)

    assert outputs.shape[0] == inputs.shape[0] - 2
