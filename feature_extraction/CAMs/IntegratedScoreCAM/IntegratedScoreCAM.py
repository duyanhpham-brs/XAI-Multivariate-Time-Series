import numpy as np
import torch
from torch.nn import functional as F
from feature_extraction.CAMs.ScoreCAM import ScoreCAM

# Adapt from https://github.com/frgfm/torch-cam/blob/master/torchcam/cams/cam.py#L291
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
