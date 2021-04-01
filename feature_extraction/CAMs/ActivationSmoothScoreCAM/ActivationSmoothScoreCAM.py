import torch
from torch.nn import functional as F
from feature_extraction.CAMs.ScoreCAM.ScoreCAM import ScoreCAM

# Adapt from https://github.com/frgfm/torch-cam/blob/master/torchcam/cams/cam.py#L179
class ActivationSmoothScoreCAM(ScoreCAM):
    def __init__(self, model, feature_module, target_layer_names, use_cuda, **kwargs):
        super().__init__(model, feature_module, target_layer_names, use_cuda)
        self.smooth_factor = kwargs['smooth_factor']
        self.std = kwargs['std']
        self._distrib = torch.distributions.normal.Normal(0, self.std)

    def reforward_saliency_map(self, input_features, norm_saliency_map):
        output_ = self.model(input_features * (norm_saliency_map + self._distrib.sample(input_features.size())))

        return output_

    def compute_score_saliency_map(self, input_features, index):
        activations, score_saliency_map, k, index = self.forward_saliency_map(input_features, index)
        with torch.no_grad():
            for _ in range(self.smooth_factor):
                for i in range(k):
                    # upsampling
                    if len(activations.size()) == 4:
                        saliency_map = torch.unsqueeze(activations[:, i, :, :], 1)
                    elif len(activations.size()) == 3:
                        saliency_map = torch.unsqueeze(torch.unsqueeze(activations[:, i, :], 2),0)

                    if saliency_map.max() == saliency_map.min():
                        continue

                    # normalize to 0-1
                    norm_saliency_map = (saliency_map - saliency_map.min()) \
                        / (saliency_map.max() - saliency_map.min())

                    # how much increase if keeping the highlighted region
                    # predication on masked input
                    output_ = self.reforward_saliency_map(input_features, norm_saliency_map)
                    output_ = F.softmax(output_, dim=1)
                    score = output_[0][index]

                    score_saliency_map_temp =  score * saliency_map
                    score_saliency_map += score_saliency_map_temp

                score_saliency_map = F.relu(score_saliency_map)
                score_saliency_map_min, score_saliency_map_max = \
                    score_saliency_map.min(), score_saliency_map.max()

                if score_saliency_map_min == score_saliency_map_max:
                    return None

                score_saliency_map = (score_saliency_map - score_saliency_map_min) \
                    .div(score_saliency_map_max - score_saliency_map_min).data
        score_saliency_map.div_(self.smooth_factor)

        return score_saliency_map
