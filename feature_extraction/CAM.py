import numpy as np
import torch
from torch.autograd import Function
from torchvision import models
from torch import nn
from utils.gradient_extraction import ModelOutputs

# Adapt from https://github.com/jacobgil/pytorch-grad-cam/blob/bf27469f5b3accf9535e04e52106e3f77f5e9cf5/gradcam.py#L74
class GradCAM:
    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        if len(grads_val.shape) == 4:
            weights = np.mean(grads_val, axis=(2, 3))[0, :]
        elif len(grads_val.shape) == 3:
            weights = np.mean(grads_val, axis=(1, 2))
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            if len(target.shape) == 3:
                cam += w * target[i, :, :]
            elif len(target.shape) == 2:
                cam += w * target[i, :]

        cam = np.maximum(cam, 0)
        return cam

# Adapt from https://github.com/adityac94/Grad_CAM_plus_plus/blob/4a9faf6ac61ef0c56e19b88d8560b81cd62c5017/misc/utils.py#L51
class GradCAMPlusPlus:
    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input):
        return self.model(input)

    @staticmethod
    def relu(x):
        cam = np.maximum(x, 0)
        cam = cam/np.max(cam)
        return cam

    def __call__(self, input, index=None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        #second_derivative
        second_derivative = torch.exp(one_hot.detach())*target 

        #triple_derivative
        triple_derivative = torch.exp(one_hot.detach())*target*target

        global_sum = np.sum(one_hot.detach().numpy(), axis=0)

        alpha_num = second_derivative.numpy()
        alpha_denom = second_derivative.numpy()*2.0 + triple_derivative.numpy()*global_sum
        alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, np.ones(alpha_denom.shape))
        alphas = alpha_num/alpha_denom

        if len(grads_val.shape) == 4:
            weights = np.sum(self.relu(grads_val)*alphas, axis=(2, 3))[0, :]
        elif len(grads_val.shape) == 3:
            weights = np.sum(self.relu(grads_val)*alphas, axis=(1, 2))
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            if len(target.shape) == 3:
                cam += w * target[i, :, :]
            elif len(target.shape) == 2:
                cam += w * target[i, :]

        cam = np.maximum(cam, 0)
        return cam