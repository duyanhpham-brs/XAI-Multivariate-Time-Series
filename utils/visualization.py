import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.gradient_extraction import upsample

class CAMFeatureMaps():
    def __init__(self, CAM_model):
        self.CAM_model = CAM_model
    
    def load(self, model, module, target_layer_names, smooth_factor=0, std=1.0):
        self.cam = self.CAM_model(model=model, feature_module=module, \
                    target_layer_names=[target_layer_names], use_cuda=False, smooth_factor=smooth_factor, std=std)

    def show(self, data, index, upsampling = True):
        self.data = data
        target_index = index
        X_inp = torch.from_numpy(self.data.reshape(1,-1,self.data.shape[0]))
        X_inp.unsqueeze_(0)
        X_inp = X_inp.float().requires_grad_(True)
        mask = np.squeeze(self.cam(X_inp, target_index))
        if len(mask.shape)==2:
            plt.figure(figsize=(200,60))
            plt.imshow(mask.T, cmap="rainbow")
            plt.yticks([i for i in range(mask.shape[1])])
            plt.grid()
            plt.show(block=False)
        elif len(mask.shape)==1:
            plt.figure(figsize=(200,60))
            plt.imshow(mask.reshape(1,-1), cmap="rainbow")
            plt.grid()
            plt.show(block=False)

        if upsampling:
            mask = upsample(mask, self.data)

        return mask

    def map_activation_to_input(self, mask):
        plt.plot(self.data.T,c='black',alpha=0.2)

        if len(mask.shape) > 1:
            if mask.shape[1] > 1:
                for j in range(self.data.T.shape[1]):
                    plt.scatter(np.arange(0, self.data.T.shape[0],1), self.data.T[:,j], c=mask[:,j], cmap="rainbow", s=7)
            else:
                for j in range(self.data.T.shape[1]):
                    plt.scatter(np.arange(0, self.data.T.shape[0],1), self.data.T[:,j], c=mask[:,0], cmap="rainbow", s=7)
        else:
            for j in range(self.data.T.shape[1]):
                plt.scatter(np.arange(0, self.data.T.shape[0],1), self.data.T[:,j], c=mask, cmap="rainbow", s=7)

        plt.show()