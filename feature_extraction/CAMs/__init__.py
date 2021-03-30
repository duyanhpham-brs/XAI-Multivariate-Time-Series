import numpy as np
import torch
from torch.autograd import Function
from torchvision import models
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict
from utils.gradient_extraction import ModelOutputs, upsample
from feature_extraction.UnitCAM import UnitCAM

# Adapt from https://github.com/zhoubolei/CAM/blob/master/pytorch_CAM.py
class CAM(UnitCAM):
    def __init__(self, model, feature_module, target_layer_names, use_cuda, **kwargs):
        super().__init__(model, feature_module, target_layer_names, use_cuda, **kwargs)
        if 'avgpool_layer' not in list(self.model._modules.keys()):
            raise ValueError('CAM does not support model without global average pooling')

    def __call__(self, input_features, index=None):
        features, output, index = self.extract_features(input_features, index)

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.squeeze(output.data.numpy())

        cam = np.zeros(target.shape[1:], dtype=np.float32)
        target = np.squeeze(target)

        cam = self.cam_weighted_sum(cam, weights, target)

        return cam

# Adapt from https://github.com/jacobgil/pytorch-grad-cam/blob/bf27469f5b3accf9535e04e52106e3f77f5e9cf5/gradcam.py#L31
class GradCAM(UnitCAM):
    def __init__(self, model, feature_module, target_layer_names, use_cuda, **kwargs):
        super().__init__(model, feature_module, target_layer_names, use_cuda, **kwargs)
    
    def calculate_gradients(self, input_features, index, print_out=True):
        features, output, index = self.extract_features(input_features, index, print_out)

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

        grads_val = self.extractor.get_gradients()[-1].cpu().data

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        return one_hot, grads_val, target

    def map_gradients(self, grads_val, target):
        if len(grads_val.shape) == 4:
            weights = np.mean(grads_val.numpy(), axis=(2, 3))[0, :]
        elif len(grads_val.shape) == 3:
            weights = np.mean(grads_val.numpy(), axis=(1, 2))
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        return cam, weights

    def __call__(self, input_features, index=None):
        _, grads_val, target = self.calculate_gradients(input_features, index)
        
        cam, weights = self.map_gradients(grads_val, target)
        cam = self.cam_weighted_sum(cam, weights, target)

        return cam

# Adapt from https://github.com/adityac94/Grad_CAM_plus_plus/blob/4a9faf6ac61ef0c56e19b88d8560b81cd62c5017/misc/utils.py#L51
class GradCAMPlusPlus(GradCAM):
    def __init__(self, model, feature_module, target_layer_names, use_cuda, **kwargs):
        super().__init__(model, feature_module, target_layer_names, use_cuda, **kwargs)

    @staticmethod
    def relu(x):
        cam = np.maximum(x, 0)
        cam = cam/(np.max(cam)+1e-9)
        return cam

    @staticmethod
    def compute_second_derivative(one_hot, target):
        #second_derivative
        second_derivative = torch.exp(one_hot.detach())*target

        return second_derivative

    @staticmethod
    def compute_third_derivative(one_hot, target):
        #third_derivative
        third_derivative = torch.exp(one_hot.detach())*target*target

        return third_derivative

    @staticmethod
    def compute_global_sum(one_hot):
        #global sum
        global_sum = np.sum(one_hot.detach().numpy(), axis=0)

        return global_sum

    def extract_higher_level_gradient(self, global_sum, second_derivative, third_derivative):      
        alpha_num = second_derivative.numpy()
        alpha_denom = second_derivative.numpy()*2.0 + third_derivative.numpy()*global_sum
        alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, np.ones(alpha_denom.shape))
        alphas = alpha_num/alpha_denom

        return alphas

    def map_gradients(self, grads_val, target, alphas):
        if len(grads_val.shape) == 4:
            weights = np.sum(F.relu(grads_val).numpy()*alphas, axis=(2, 3))[0, :]
        elif len(grads_val.shape) == 3:
            weights = np.sum(F.relu(grads_val).numpy()*alphas, axis=(1, 2))
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        return cam, weights

    def __call__(self, input_features, index=None):
        one_hot, grads_val, target = self.calculate_gradients(input_features, index)
        second_derivative = self.compute_second_derivative(one_hot, target)
        third_derivative = self.compute_third_derivative(one_hot, target)
        global_sum = self.compute_global_sum(one_hot)
        alphas = self.extract_higher_level_gradient(global_sum, second_derivative, third_derivative)
        cam, weights = self.map_gradients(grads_val, target, alphas)
        cam = self.cam_weighted_sum(cam, weights, target)

        return cam

# Adapt from https://github.com/frgfm/torch-cam/blob/master/torchcam/cams/gradcam.py#L164
class SmoothGradCAMPlusPlus(GradCAMPlusPlus):
    def __init__(self, model, feature_module, target_layer_names, use_cuda, **kwargs):
        super().__init__(model, feature_module, target_layer_names, use_cuda, **kwargs)
        self.smooth_factor = kwargs['smooth_factor']
        self.std = kwargs['std']
        self._distrib = torch.distributions.normal.Normal(0, self.std)

    def extract_higher_level_gradient(self, global_sum, second_derivative, third_derivative):      
        alpha_num = second_derivative.numpy()
        alpha_denom = second_derivative.numpy()*2.0 + third_derivative.numpy()*global_sum
        alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, np.ones(alpha_denom.shape))
        alphas = alpha_num/alpha_denom

        return alphas

    def __call__(self, input_features, index=None):
        grads_vals = None
        second_derivatives = None
        third_derivatives = None
        for i in range(self.smooth_factor):
            one_hot, grads_val, target = self.calculate_gradients(input_features + self._distrib.sample(input_features.size()), index, False)
            second_derivative = self.compute_second_derivative(one_hot, target)
            third_derivative = self.compute_third_derivative(one_hot, target)
            if grads_vals is None or second_derivatives is None or third_derivatives is None:
                grads_vals = grads_val
                second_derivatives = second_derivative
                third_derivatives = third_derivative
            else:
                grads_vals += grads_val
                second_derivatives += second_derivative
                third_derivatives += third_derivative
            
            second_derivatives = F.relu(second_derivatives)
            second_derivatives_min, second_derivatives_max = second_derivatives.min(), second_derivatives.max()
            if second_derivatives_min == second_derivatives_max:
                return None
            second_derivatives = (second_derivatives - second_derivatives_min).div(second_derivatives_min - second_derivatives_max).data

            third_derivatives = F.relu(third_derivatives)
            third_derivatives_min, third_derivatives_max = third_derivatives.min(), third_derivatives.max()
            if third_derivatives_min == third_derivatives_max:
                return None
            third_derivatives = (third_derivatives - third_derivatives_min).div(third_derivatives_min - third_derivatives_max).data
        
        one_hot, _, target = self.calculate_gradients(input_features, index)
        global_sum = self.compute_global_sum(one_hot)

        alphas = self.extract_higher_level_gradient(global_sum, second_derivatives.div_(self.smooth_factor),third_derivatives.div_(self.smooth_factor))
        cam, weights = self.map_gradients(grads_vals.div_(self.smooth_factor), target, alphas)
        cam = self.cam_weighted_sum(cam, weights, target)

        return cam

# Adapt from https://github.com/haofanwang/Score-CAM/blob/master/cam/scorecam.py
class ScoreCAM(UnitCAM):
    def __init__(self, model, feature_module, target_layer_names, use_cuda, **kwargs):
        super().__init__(model, feature_module, target_layer_names, use_cuda, **kwargs)
    
    def forward_saliency_map(self, input_features, index):
        b, c, h, w = input_features.size()

        features, output, index = self.extract_features(input_features, index)

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

        return activations, score_saliency_map, k, index

    def reforward_saliency_map(self, input_features, norm_saliency_map):
        output_ = self.model(input_features * norm_saliency_map)

        return output_
    
    def compute_score_saliency_map(self, input_features, index):
        activations, score_saliency_map, k, index = self.forward_saliency_map(input_features, index)
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
                output_ = self.reforward_saliency_map(input_features, norm_saliency_map)
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

    def __call__(self, input_features, index=None):
        score_saliency_map = self.compute_score_saliency_map(input_features, index)

        return score_saliency_map
        
        

# Adapt from https://github.com/frgfm/torch-cam/blob/master/torchcam/cams/cam.py#L179
class ActivationSmoothScoreCAM(ScoreCAM):
    def __init__(self, model, feature_module, target_layer_names, use_cuda, **kwargs):
        super().__init__(model, feature_module, target_layer_names, use_cuda, **kwargs)
        self.smooth_factor = kwargs['smooth_factor']
        self.std = kwargs['std']
        self._distrib = torch.distributions.normal.Normal(0, self.std)

    def reforward_saliency_map(self, input_features, norm_saliency_map):
        output_ = self.model(input_features * (norm_saliency_map + self._distrib.sample(input_features.size())))

        return output_
    
    def compute_score_saliency_map(self, input_features, index):
        activations, score_saliency_map, k, index = self.forward_saliency_map(input_features, index)
        with torch.no_grad():
            for idx in range(self.smooth_factor):
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
                    output_ = self.reforward_saliency_map(input_features, norm_saliency_map)
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

# Adapt from https://github.com/frgfm/torch-cam/blob/master/torchcam/cams/cam.py#L179
class InputSmoothScoreCAM(ScoreCAM):
    def __init__(self, model, feature_module, target_layer_names, use_cuda, **kwargs):
        super().__init__(model, feature_module, target_layer_names, use_cuda, **kwargs)
        self.smooth_factor = kwargs['smooth_factor']
        self.std = kwargs['std']
        self._distrib = torch.distributions.normal.Normal(0, self.std)

    def reforward_saliency_map(self, input_features, norm_saliency_map):
        output_ = self.model((input_features + self._distrib.sample(input_features.size())) * norm_saliency_map)

        return output_
    
    def compute_score_saliency_map(self, input_features, index):
        activations, score_saliency_map, k, index = self.forward_saliency_map(input_features, index)
        with torch.no_grad():
            for idx in range(self.smooth_factor):
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
                    output_ = self.reforward_saliency_map(input_features, norm_saliency_map)
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

class IntegratedScoreCAM(ScoreCAM):
    def __init__(self, model, feature_module, target_layer_names, use_cuda, **kwargs):
        super().__init__(model, feature_module, target_layer_names, use_cuda, **kwargs)
        self.smooth_factor = kwargs['smooth_factor']

    def reforward_saliency_map(self, int_feature_maps):
        output_ = self.model(int_feature_maps)

        return output_
    
    def compute_score_saliency_map(self, input_features, index):
        activations, score_saliency_map, k, index = self.forward_saliency_map(input_features, index)
        int_feature_maps = torch.zeros((score_saliency_map.shape[0], *input_features.shape[1:]),
                           dtype=score_saliency_map.dtype, device=score_saliency_map.device)
        with torch.no_grad():
            for idx in range(self.smooth_factor):
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
                    int_feature_maps += (idx + 1) / self.smooth_factor * input_features * norm_saliency_map
                    output_ = self.reforward_saliency_map(int_feature_maps)
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