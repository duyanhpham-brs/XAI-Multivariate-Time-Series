import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.gradient_extraction import upsample

class CAMFeatureMaps():
    def __init__(self, CAM_model):
        self.CAM_model = CAM_model
    
    def load(self, model, module, target_layer_names, has_globavgpool=True, final_depth=None):
        self.cam = self.CAM_model(model=model, feature_module=module, \
                    target_layer_names=[target_layer_names], use_cuda=False)

    def show(self, data, index, upsampling = True):
        target_index = index
        X_inp = torch.from_numpy(data.reshape(1,-1,data.shape[0]))
        X_inp.unsqueeze_(0)
        X_inp = X_inp.float().requires_grad_(True)
        mask = np.squeeze(self.cam(X_inp, target_index))
        if len(mask.shape)==2:
            plt.figure(figsize=(200,60))
            plt.imshow(mask.T)
            plt.yticks([i for i in range(mask.shape[1])])
            plt.grid()
            plt.show()
        elif len(mask.shape)==1:
            plt.figure(figsize=(200,60))
            plt.imshow(mask.reshape(1,-1))
            plt.grid()
            plt.show()

        if upsampling:
            mask = upsample(mask, data)

        return mask

def map_activation_to_input(data, mask):
    plt.plot(data.T,c='black',alpha=0.2)

    if mask.shape[1] > 1:
        for j in range(data.T.shape[1]):
            plt.scatter(np.arange(0,data.T.shape[0],1),data.T[:,j],c=mask[:,j],s=7)
    else:
        for j in range(data.T.shape[1]):
            plt.scatter(np.arange(0,data.T.shape[0],1),data.T[:,j],c=mask[:,0],s=7)
    plt.show()