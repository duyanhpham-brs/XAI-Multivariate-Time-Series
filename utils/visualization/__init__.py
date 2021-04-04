import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.gradient_extraction import upsample


class CAMFeatureMaps:
    def __init__(self, CAM_model):
        self.CAM_model = CAM_model
        self.cam = None
        self.data = None

    def load(self, model, module, target_layer_names, smooth_factor=None, std=None, has_gap=None):
        if smooth_factor is not None and std is not None and has_gap is not None:
            self.cam = self.CAM_model(
                model=model,
                feature_module=module,
                target_layer_names=[target_layer_names],
                use_cuda=False,
                smooth_factor=smooth_factor,
                std=std,
                has_gap=has_gap
            )
        elif smooth_factor is not None:
            self.cam = self.CAM_model(
                model=model,
                feature_module=module,
                target_layer_names=[target_layer_names],
                use_cuda=False,
                smooth_factor=smooth_factor
            )
        elif std is not None:
            self.cam = self.CAM_model(
                model=model,
                feature_module=module,
                target_layer_names=[target_layer_names],
                use_cuda=False,
                std=std
            )
        elif has_gap is not None:
            self.cam = self.CAM_model(
                model=model,
                feature_module=module,
                target_layer_names=[target_layer_names],
                use_cuda=False,
                has_gap=has_gap
            )
        elif has_gap is not None and std is not None:
            self.cam = self.CAM_model(
                model=model,
                feature_module=module,
                target_layer_names=[target_layer_names],
                use_cuda=False,
                has_gap=has_gap,
                std=std
            )
        elif has_gap is not None and smooth_factor is not None:
            self.cam = self.CAM_model(
                model=model,
                feature_module=module,
                target_layer_names=[target_layer_names],
                use_cuda=False,
                has_gap=has_gap,
                smooth_factor=smooth_factor
            )
        elif std is not None and smooth_factor is not None:
            self.cam = self.CAM_model(
                model=model,
                feature_module=module,
                target_layer_names=[target_layer_names],
                use_cuda=False,
                std=std,
                smooth_factor=smooth_factor
            )
        else:
            self.cam = self.CAM_model(
                model=model,
                feature_module=module,
                target_layer_names=[target_layer_names],
                use_cuda=False
            )

    def show(self, data, index, dataset_path=None, upsampling=True):
        self.data = data
        target_index = index
        X_inp = torch.from_numpy(self.data.reshape(1, -1, self.data.shape[0]))
        X_inp.unsqueeze_(0)
        X_inp = X_inp.float().requires_grad_(True)
        if dataset_path is None:
            mask = np.squeeze(self.cam(X_inp, target_index))
        else:
            mask = np.squeeze(self.cam(X_inp, target_index, dataset_path))
        if len(mask.shape) == 2:
            plt.figure(figsize=(200, 60))
            plt.imshow(mask.T, cmap="jet")
            plt.yticks(range(mask.shape[1]))
            plt.grid()
            plt.show(block=False)
        elif len(mask.shape) == 1:
            plt.figure(figsize=(200, 60))
            plt.imshow(mask.reshape(1, -1), cmap="jet")
            plt.grid()
            plt.show(block=False)

        if upsampling:
            mask = upsample(mask, self.data)

        return mask

    def map_activation_to_input(self, mask):
        plt.plot(self.data.T, c="black", alpha=0.2)

        if len(mask.shape) > 1:
            if mask.shape[1] > 1:
                for j in range(self.data.T.shape[1]):
                    plt.scatter(
                        np.arange(0, self.data.T.shape[0], 1),
                        self.data.T[:, j],
                        c=mask[:, j],
                        cmap="jet",
                        s=7,
                    )
            else:
                for j in range(self.data.T.shape[1]):
                    plt.scatter(
                        np.arange(0, self.data.T.shape[0], 1),
                        self.data.T[:, j],
                        c=mask[:, 0],
                        cmap="jet",
                        s=7,
                    )
        else:
            for j in range(self.data.T.shape[1]):
                plt.scatter(
                    np.arange(0, self.data.T.shape[0], 1),
                    self.data.T[:, j],
                    c=mask,
                    cmap="jet",
                    s=7,
                )

        plt.show()
