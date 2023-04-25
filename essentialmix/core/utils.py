import torch.nn as nn

from functools import reduce
from operator import mul


def get_number_of_parameters(model: nn.Module) -> int:
    n_params = 0
    for name, parameter in model.named_parameters():
        n_params += reduce(mul, parameter.shape, 1)
    return n_params