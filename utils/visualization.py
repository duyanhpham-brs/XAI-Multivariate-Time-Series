import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class CAMFeatureMaps():
    def __init__(self, CAM_model):
        self.CAM_model = CAM_model
    
    def load(self, model, module, target_layer_names):
        self.cam = self.CAM_model(model=model, feature_module=module, \
                    target_layer_names=[target_layer_names], use_cuda=False)

    def show(self, data, index, upsampling = True):
        target_index = index
        X_inp = torch.from_numpy(data.reshape(1,-1,data.shape[0]))
        X_inp.unsqueeze_(0)
        X_inp = X_inp.float().requires_grad_(True)
        mask = self.cam(X_inp, target_index)
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
            index = np.linspace(0,len(mask)-1,len(mask))
            df = pd.DataFrame(mask)
            df.set_index(index)

            X_resampled = np.linspace(0,len(mask)-1,data.shape[1])
            df_resampled = df.reindex(df.index.union(X_resampled)).interpolate('values').loc[X_resampled]
            mask = np.array(df_resampled)

        return mask

def map_activation_to_input(data, mask, magnifying_coeff):
    plt.plot(data.T)
    for j in range(3):
        for i in range(len(data.T)):
            if j == 0:
                plt.scatter(i,data.T[i,j],c='red',s=magnifying_coeff*mask[i,j])
            elif j == 1:
                plt.scatter(i,data.T[i,j],c='lightgreen',s=magnifying_coeff*mask[i,j])
            else:
                plt.scatter(i,data.T[i,j],c='black',s=magnifying_coeff*mask[i,j])
    plt.show()