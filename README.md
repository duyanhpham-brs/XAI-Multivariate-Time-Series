![develop lint status](https://github.com/duyanhpham-brs/XAI-Multivariate-Time-Series/workflows/develop_branch_lint/badge.svg)
![develop test status](https://github.com/duyanhpham-brs/XAI-Multivariate-Time-Series/workflows/develop_branch_test/badge.svg)

# XAI-Multivariate-Time-Series

## Installation

This project is using `pipenv` to manage and organize the libraries. Before being up and running, please make sure to have `pipenv` in your system and do the following:

```
pipenv shell
pipenv install
```

## Basic Usage

```python
from feature_extraction.CAMs import CAM
from utils.visualization import CAMFeatureMaps

feature_maps = CAMFeatureMaps(CAM)
feature_maps.load(extracting_model,extracting_module,targeting_layer)
mask = feature_maps.show(X_test[0], None)
feature_maps.map_activation_to_input(mask)
```

## Result of CAM explaining XCM ReLU 1 Conv2d Branch

### Mask

![Mask Image](https://github.com/duyanhpham-brs/XAI-Multivariate-Time-Series/blob/develop/static_images/XCM_relu11_CAM_UWave_test0.png.png?raw=true)

### Mapping mask onto input

![Mapped Image](https://github.com/duyanhpham-brs/XAI-Multivariate-Time-Series/blob/develop/static_images/XCM_relu11_CAM_UWave_test0_map.png?raw=true)
