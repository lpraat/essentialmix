from functools import reduce
from operator import mul

import torch
import torch.nn as nn


def get_number_of_parameters(model: nn.Module) -> int:
    n_params = 0
    for parameter in model.parameters():
        n_params += reduce(mul, parameter.shape, 1)
    return n_params


def slerp_batched(alpha: float, point_a: torch.Tensor, point_b: torch.Tensor) -> torch.Tensor:
    """
    Batched spherical linear interpolation
    """
    assert point_a.shape == point_b.shape
    batch_size = point_a.shape[0]
    point_a_vec = point_a.view(batch_size, -1)
    point_b_vec = point_b.view(batch_size, -1)
    theta = torch.arccos(
        torch.sum(point_a_vec * point_b_vec, dim=1) / (torch.norm(point_a_vec, dim=1) * torch.norm(point_b_vec, dim=1))
    )
    interp = (torch.sin((1 - alpha) * theta) / torch.sin(theta)) * point_a_vec + (
        torch.sin(alpha * theta) / torch.sin(theta)
    ) * point_b_vec
    return interp.view(point_a.shape)


def slerp(alpha: float, point_a: torch.Tensor, point_b: torch.Tensor) -> torch.Tensor:
    """
    Spherical linear interpolation
    """
    return slerp_batched(alpha, point_a.unsqueeze(dim=0), point_b.unsqueeze(dim=0))
