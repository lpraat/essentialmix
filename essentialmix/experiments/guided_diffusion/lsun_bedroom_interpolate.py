import os
import time
from collections import deque
from typing import Literal

import click
import matplotlib.pyplot as plt
import torch

from essentialmix.core.plot import image_grid
from essentialmix.core.utils import slerp
from essentialmix.experiments.guided_diffusion.common import build_diffusion_kwargs, prepare_img_tensor_for_plot
from essentialmix.experiments.guided_diffusion.diffusion import GaussianDiffusion

# class-unconditional lsun bedroom
config = {
    "model": {
        "attention_resolutions": "32,16,8",
        "class_cond": False,
        "dropout": 0.1,
        "num_channels": 256,
        "num_head_channels": 64,
        "num_res_blocks": 2,
        "resblock_updown": True,
        "use_fp16": True,
        "use_scale_shift_norm": True,
        "image_size": 256,
        "learn_sigma": True
    },
    "diffusion_process": {
        "num_diffusion_timesteps": 1000,
        "sigma_learned": True,
        "noise_schedule": "linear",
    },

    "weights_uris": {
        "model": "/Users/lpraat/develop/essentialmix/essentialmix/experiments/guided_diffusion/lsun_bedroom.pt",
    }
}


@click.command()
@click.option('--n_interp_rows', help='Number of interpolated rows to generate', required=True, type=int)
@click.option('--batch_size', help='Batch size', required=True, type=int)
@click.option('--device', help='Torch device', default='cuda', type=click.Choice(['cuda', 'mps'], case_sensitive=False))
@click.option('--output_folder', help='Folder to save the results to', default='.', type=str)
@click.option('--denoise_steps', help='The number of steps in the reverse process', required=True, type=int)
def generate(
    n_interp_rows: int,
    batch_size: int,
    device: Literal['mps', 'cuda'],
    output_folder: str,
    denoise_steps: int,
) -> None:
    t0 = time.perf_counter()
    diffusion_kwargs = build_diffusion_kwargs(config, device=device)
    with torch.device(device):
        diffusion_process = GaussianDiffusion(
            timestep_respacing=denoise_steps,
            **diffusion_kwargs,
        )

        noise_starts = deque()
        alphas = torch.linspace(1, 0, 6)

        for _ in range(n_interp_rows):
            a = torch.randn(3, 256, 256)
            b = torch.randn(3, 256, 256)
            for alpha in alphas:
                noise_starts.append(slerp(alpha, a, b).view(*a.shape))

        batch_size = batch_size
        full = len(noise_starts) // batch_size
        batch_sizes = [batch_size for _ in range(full)]
        if len(noise_starts) % batch_size != 0:
            batch_sizes += [len(noise_starts) - full * batch_size]

        images = []
        with torch.no_grad():
            for batch_iter, batch_size in enumerate(batch_sizes):
                print(
                    f"Generated {sum(batch_sizes[:batch_iter])}/{sum(batch_sizes[batch_iter:])} images. "
                    f"Current iter={batch_iter+1}/{len(batch_sizes)}"
                )
                x_0 = []
                for _ in range(batch_size):
                    x_0.append(noise_starts.popleft())
                x_0 = torch.stack(x_0)

                for output in diffusion_process.denoise(x_0=x_0, batch_size=batch_size, use_ddim=True):
                    denoised_x = output['denoised_x']

                for i in range(batch_size):
                    images.append(prepare_img_tensor_for_plot(denoised_x[i]))

        fig = image_grid(images, n_cols=len(alphas))
        grid_path = os.path.join(output_folder, f"lsun_bedroom_interp_grid.png")
        print(f"Saving interp grid {grid_path}")
        fig.savefig(grid_path)
        plt.clf()
        print(f"Done! Took {time.perf_counter() - t0:.2f} s")


if __name__ == '__main__':
    generate()
