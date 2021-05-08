import torch
from torch.autograd import Variable

from models.attention_based.helpers.train_darnn.constants import device


def numpy_to_tvar(x: torch.Tensor):
    return Variable(torch.from_numpy(x).type(torch.FloatTensor).to(device))
