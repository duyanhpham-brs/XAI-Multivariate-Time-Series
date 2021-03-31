import torch
import numpy as np
import pandas as pd

# Adapt from https://github.com/jacobgil/pytorch-grad-cam/blob/bf27469f5b3accf9535e04e52106e3f77f5e9cf5/gradcam.py#L9
class FeatureExtractor():
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x

# Adapt from https://github.com/jacobgil/pytorch-grad-cam/blob/bf27469f5b3accf9535e04e52106e3f77f5e9cf5/gradcam.py#L31
class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        branches = {}
        for name, module in self.model._modules.items():
            if module == self.feature_module and not "_b" in name.lower():
                target_activations, x = self.feature_extractor(x)
            elif "_b" in name.lower():
                if module == self.feature_module:
                    target_activations, temp = self.feature_extractor(x)
                    _, _, num_branch = name.split('_')
                    if not num_branch in list(branches.keys()):
                        branches[num_branch] = []
                        branches[num_branch].append(temp)
                    else:
                        branches[num_branch].append(temp)
                else:
                    _, _, num_branch = name.split('_')
                    if not num_branch in list(branches.keys()):
                        branches[num_branch] = []
                        branches[num_branch].append(module(x))
                    else:
                        branches[num_branch].append(module(x))

            elif "avgpool" in name.lower():
                x = module(x)
                x = x.view(x.size(0),-1)
            elif not "linear" in name.lower():
                x = torch.cat(tuple(branches[num_branch]), 1)
                x = module(x)
            else:
                if len(x.size()) == 3:
                    x = x.view(x.size(0), -1)
                elif len(x.size()) == 4:
                    x = x.view(x.size(0),x.size(-1), -1)
                x = module(x)

        return target_activations, x

def upsample(mask, orig):
    index = np.linspace(0,len(mask)-1,len(mask))
    df = pd.DataFrame(mask)
    df.set_index(index)

    X_resampled = np.linspace(0,len(mask)-1,orig.shape[1])
    df_resampled = df.reindex(df.index.union(X_resampled)).interpolate('values').loc[X_resampled]
    mask = np.array(df_resampled)

    return mask
