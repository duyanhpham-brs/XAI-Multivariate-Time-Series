![develop lint status](https://github.com/duyanhpham-brs/XAI-Multivariate-Time-Series/workflows/develop_branch_lint/badge.svg)
![develop test status](https://github.com/duyanhpham-brs/XAI-Multivariate-Time-Series/workflows/develop_branch_test/badge.svg)

# XAI-Multivariate-Time-Series

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

![Mask Image](https://github.com/duyanhpham-brs/XAI-Multivariate-Time-Series/blob/develop/static_images/XCM_relu11_CAM_UWave_test0_map.png?raw=true)

### Mapping mask onto input

![Mapped Image](https://github.com/duyanhpham-brs/XAI-Multivariate-Time-Series/blob/develop/static_images/XCM_relu11_CAM_UWave_test0.png?raw=true)
