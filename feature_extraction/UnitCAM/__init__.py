import numpy as np
from utils.gradient_extraction import ModelOutputs


class UnitCAM:
    """Unit Class Activation Mapping (UnitCAM)

    UnitCAM is the foundation for implementing all the CAMs

    Attributes:
    -------
        model: The wanna-be explained deep learning model for multivariate time series classification
        feature_module: The wanna-be explained module group (e.g. linear_layers)
        target_layer_names: The wanna-be explained module
        use_cuda: Whether to use cuda

    """

    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        self.model = model
        self.feature_module = feature_module
        self.target_layer_names = target_layer_names
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(
            self.model, self.feature_module, self.target_layer_names
        )

    def forward(self, input_features):
        """Forward pass

        Attributes:
        -------
            input_features: A multivariate data input to the model

        Returns:
        -------
            Forward-pass result

        """
        return self.model(input_features)

    def extract_features(self, input_features, print_out, index, zero_out=None):
        """Extract the feature maps of the targeted layer

        Attributes:
        -------
            input_features: A multivariate data input to the model
            index: Targeted output class
            print_out: Whether to print the maximum likelihood class
                (if index is set to None)
            zero_out: Whether to set the targeted module weights to 0
                (used in Ablation-CAM)

        Returns:
        -------
            features: The feature maps of the targeted layer
            output: The forward-pass result
            index: The targeted class index

        """
        if self.cuda:
            if zero_out:
                features, output = self.extractor(input_features.cuda(), zero_out)
            else:
                features, output = self.extractor(input_features.cuda())
        else:
            if zero_out:
                features, output = self.extractor(input_features, zero_out)
            else:
                features, output = self.extractor(input_features)

        if index is None:
            index = np.argmax(output.cpu().data.numpy())
            if print_out:
                print(f"The index has the largest maximum likelihood is {index}")

        return features, output, index

    @staticmethod
    def cam_weighted_sum(cam, weights, target, ReLU=True):
        """Do linear combination between the defined weights and corresponding
        feature maps

        Attributes:
        -------
            cam: A placeholder for the final results
            weights: The weights computed based on the network output
            target: The targeted feature maps

        Returns:
        -------
            cam: The resulting weighted feature maps

        """
        try:
            for i, w in enumerate(weights):
                if len(target.shape) == 3:
                    cam += w * target[i, :, :]
                elif len(target.shape) == 2:
                    cam += w * target[i, :]
                elif (
                    len(target.shape) == 1
                    or target.shape[0] == 1
                    and len(target.shape) == 2
                ):
                    cam += w * target.reshape(-1)[i]
        except TypeError:
            if len(target.shape) == 3:
                cam += weights * target[0:1, :, :]
            elif len(target.shape) == 2:
                cam += weights * target[0:1, :]
            elif (
                len(target.shape) == 1
                or target.shape[0] == 1
                and len(target.shape) == 2
            ):
                cam += weights * target.reshape(-1)[0]

        if ReLU:
            cam = np.maximum(cam, 0)

        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-9)
        return cam

    def __call__(self, input_features, index=None):
        """Abstract methods for implementing in the sub classes

        Attributes:
        -------
            input_features: A multivariate data input to the model
            index: Targeted output class

        """
        return NotImplementedError
