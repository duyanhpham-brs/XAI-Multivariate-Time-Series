import numpy as np
import torch
from torch.autograd import Function
from torchvision import models
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict
from utils.gradient_extraction import ModelOutputs, upsample

# Adapt from https://github.com/zhoubolei/CAM/blob/master/pytorch_CAM.py
class CAM:
    def __init__(self, model, feature_module, target_layer_names, use_cuda, **kwargs):
        self.model = model
        if 'avgpool_layer' not in list(self.model._modules.keys()):
            raise ValueError('CAM does not support model without global average pooling')
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
            print(f'The index has the largest maximum likelihood is {index}')

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.squeeze(output.data.numpy())

        cam = np.zeros(target.shape[1:], dtype=np.float32)
        target = np.squeeze(target)
        for i, w in enumerate(weights):
            if len(target.shape) == 3:
                cam += w * target[i, :, :]
            elif len(target.shape) == 2:
                cam += w * target[i, :]

        cam = np.maximum(cam, 0)
        return cam

# Adapt from https://github.com/jacobgil/pytorch-grad-cam/blob/bf27469f5b3accf9535e04e52106e3f77f5e9cf5/gradcam.py#L31
class GradCAM:
    def __init__(self, model, feature_module, target_layer_names, use_cuda, **kwargs):
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
            print(f'The index has the largest maximum likelihood is {index}')

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
    def __init__(self, model, feature_module, target_layer_names, use_cuda, **kwargs):
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
            print(f'The index has the largest maximum likelihood is {index}')

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
            elif len(target.shape) == 1 or target.shape[0]==1 and len(target.shape) == 2:
                cam += w * target.reshape(-1)[i]

        cam = np.maximum(cam, 0)
        return cam

class ScoreCAM:
    def __init__(self, model, feature_module, target_layer_names, use_cuda, **kwargs):
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
        b, c, h, w = input.size()

        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())
            print(f'The index has the largest maximum likelihood is {index}')

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()

        activations = features[-1]
        if len(activations.size()) == 4:
            b, k, u, v = activations.size()
            score_saliency_map = torch.zeros((1, 1, h, w))
        elif len(activations.size()) == 3:
            b, k, u = activations.size()
            score_saliency_map = torch.zeros((1, 1, h, 1))

        if torch.cuda.is_available():
          activations = activations.cuda()
          score_saliency_map = score_saliency_map.cuda()

        with torch.no_grad():
          for i in range(k):

              # upsampling
                if len(activations.size()) == 4:
                    saliency_map = torch.unsqueeze(activations[:, i, :, :], 1)
                elif len(activations.size()) == 3:
                    saliency_map = torch.unsqueeze(torch.unsqueeze(activations[:, i, :], 2),0)
                
                if saliency_map.max() == saliency_map.min():
                    continue
                
                # normalize to 0-1
                norm_saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())

                # how much increase if keeping the highlighted region
                # predication on masked input
                output_ = self.model(input * norm_saliency_map)
                output_ = F.softmax(output_, dim=1)
                score = output_[0][index]

                score_saliency_map_temp =  score * saliency_map
                score_saliency_map += score_saliency_map_temp
                
        score_saliency_map = F.relu(score_saliency_map)
        score_saliency_map_min, score_saliency_map_max = score_saliency_map.min(), score_saliency_map.max()

        if score_saliency_map_min == score_saliency_map_max:
            return None

        score_saliency_map = (score_saliency_map - score_saliency_map_min).div(score_saliency_map_max - score_saliency_map_min).data

        return score_saliency_map

class ActivationSmoothScoreCAM:
    def __init__(self, model, feature_module, target_layer_names, use_cuda, **kwargs):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.smooth_factor = kwargs['smooth_factor']
        self.std = kwargs['std']
        self._distrib = torch.distributions.normal.Normal(0, self.std)

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        b, c, h, w = input.size()

        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())
            print(f'The index has the largest maximum likelihood is {index}')

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()

        activations = features[-1]
        if len(activations.size()) == 4:
            b, k, u, v = activations.size()
            score_saliency_map = torch.zeros((1, 1, h, w))
        elif len(activations.size()) == 3:
            b, k, u = activations.size()
            score_saliency_map = torch.zeros((1, 1, h, 1))

        if torch.cuda.is_available():
          activations = activations.cuda()
          score_saliency_map = score_saliency_map.cuda()

        for idx in range(self.smooth_factor):
            with torch.no_grad():
                for i in range(k):
                    # upsampling
                        if len(activations.size()) == 4:
                            saliency_map = torch.unsqueeze(activations[:, i, :, :], 1)
                        elif len(activations.size()) == 3:
                            saliency_map = torch.unsqueeze(torch.unsqueeze(activations[:, i, :], 2),0)
                        
                        if saliency_map.max() == saliency_map.min():
                            continue
                        
                        # normalize to 0-1
                        norm_saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())

                        # how much increase if keeping the highlighted region
                        # predication on masked input
                        output_ = self.model(input * (norm_saliency_map + self._distrib.sample(input.size())))
                        output_ = F.softmax(output_, dim=1)
                        score = output_[0][index]

                        score_saliency_map_temp =  score * saliency_map
                        score_saliency_map += score_saliency_map_temp
                        
                score_saliency_map = F.relu(score_saliency_map)
                score_saliency_map_min, score_saliency_map_max = score_saliency_map.min(), score_saliency_map.max()

                if score_saliency_map_min == score_saliency_map_max:
                    return None

                score_saliency_map = (score_saliency_map - score_saliency_map_min).div(score_saliency_map_max - score_saliency_map_min).data
        score_saliency_map.div_(self.smooth_factor)

        return score_saliency_map

class InputSmoothScoreCAM:
    def __init__(self, model, feature_module, target_layer_names, use_cuda, **kwargs):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.smooth_factor = kwargs['smooth_factor']
        self.std = kwargs['std']
        self._distrib = torch.distributions.normal.Normal(0, self.std)

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        b, c, h, w = input.size()

        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())
            print(f'The index has the largest maximum likelihood is {index}')

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()

        activations = features[-1]
        if len(activations.size()) == 4:
            b, k, u, v = activations.size()
            score_saliency_map = torch.zeros((1, 1, h, w))
        elif len(activations.size()) == 3:
            b, k, u = activations.size()
            score_saliency_map = torch.zeros((1, 1, h, 1))

        if torch.cuda.is_available():
          activations = activations.cuda()
          score_saliency_map = score_saliency_map.cuda()

        for idx in range(self.smooth_factor):
            with torch.no_grad():
                for i in range(k):
                    # upsampling
                        if len(activations.size()) == 4:
                            saliency_map = torch.unsqueeze(activations[:, i, :, :], 1)
                        elif len(activations.size()) == 3:
                            saliency_map = torch.unsqueeze(torch.unsqueeze(activations[:, i, :], 2),0)
                        
                        if saliency_map.max() == saliency_map.min():
                            continue
                        
                        # normalize to 0-1
                        norm_saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())

                        # how much increase if keeping the highlighted region
                        # predication on masked input
                        output_ = self.model((input + self._distrib.sample(input.size())) * norm_saliency_map )
                        output_ = F.softmax(output_, dim=1)
                        score = output_[0][index]

                        score_saliency_map_temp =  score * saliency_map
                        score_saliency_map += score_saliency_map_temp
                        
                score_saliency_map = F.relu(score_saliency_map)
                score_saliency_map_min, score_saliency_map_max = score_saliency_map.min(), score_saliency_map.max()

                if score_saliency_map_min == score_saliency_map_max:
                    return None

                score_saliency_map = (score_saliency_map - score_saliency_map_min).div(score_saliency_map_max - score_saliency_map_min).data
        score_saliency_map.div_(self.smooth_factor)

        return score_saliency_map
