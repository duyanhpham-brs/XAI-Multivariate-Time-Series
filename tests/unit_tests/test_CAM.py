import pytest
import torch
import torch.nn.functional as F
import numpy as np
from context import utils, feature_extraction, tests
from feature_extraction.CAMs import CAM
from tests.fixtures.models import MockSingularBranchModel, MockDualBranchModel
from utils.visualization import CAMFeatureMaps


def test_