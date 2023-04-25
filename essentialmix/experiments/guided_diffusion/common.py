from collections import ChainMap
from typing import Literal

import torch
import torch.nn as nn
import numpy as np

from essentialmix.models.externals.guided_diffusion import (
    create_model, noise_model_defaults, create_classifier, classifier_defaults
)


def prepare_model(model: nn.Module, weights_uri: str, device: Literal['mps', 'cuda'], use_fp16: bool) -> nn.Module:
    model.load_state_dict(torch.load(weights_uri, map_location=device))
    model.to(device)
    if use_fp16:
        model.convert_to_fp16()
    model.eval()
    return model


def build_diffusion_kwargs(config: dict, device: Literal['mps', 'cuda']) -> dict:
    kwargs = {}

    # Weights
    weights_uris = config['weights_uris']

    # Diffusion
    kwargs.update(config['diffusion_process'])

    # Model
    model_config = ChainMap(config['model'], noise_model_defaults())
    kwargs['model'] = prepare_model(
        model=create_model(**model_config),
        weights_uri=weights_uris['model'],
        device=device,
        use_fp16=model_config['use_fp16']
    )

    # Classifier
    if 'classifier' in config:
        classifier_config = ChainMap(config['classifier'], classifier_defaults())
        kwargs['classifier'] = prepare_model(
            model=create_classifier(**classifier_config),
            weights_uri=weights_uris['classifier'],
            device=device,
            use_fp16=classifier_config['classifier_use_fp16'],
        )

    return kwargs


def prepare_img_tensor_for_plot(img: torch.Tensor) -> np.ndarray:
    ch, h, w = img.shape
    return (
        ((img + 1) * 127.5).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).contiguous()
        .cpu().numpy().reshape(h, w, ch)
    )











