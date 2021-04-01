import numpy as np
from utils.gradient_extraction import ModelOutputs

class UnitCAM:
    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input_features):
        return self.model(input_features)

    def extract_features(self, input_features, index, print_out=True):
        if self.cuda:
            features, output = self.extractor(input_features.cuda())
        else:
            features, output = self.extractor(input_features)

        if index is None:
            index = np.argmax(output.cpu().data.numpy())
            if print_out:
                print(f'The index has the largest maximum likelihood is {index}')

        return features, output, index

    @staticmethod
    def cam_weighted_sum(cam, weights, target):
        for i, w in enumerate(weights):
            if len(target.shape) == 3:
                cam += w * target[i, :, :]
            elif len(target.shape) == 2:
                cam += w * target[i, :]
            elif len(target.shape) == 1 or target.shape[0]==1 and len(target.shape) == 2:
                cam += w * target.reshape(-1)[i]

        cam = np.maximum(cam, 0)
        return cam


    def __call__(self, input_features, index=None):
        return NotImplementedError
