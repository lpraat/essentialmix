import os
import time
from typing import Literal

import click
import matplotlib.pyplot as plt
import torch

from essentialmix.core.plot import image_grid
from essentialmix.experiments.guided_diffusion.common import (
    build_diffusion_kwargs,
    prepare_img_tensor_for_plot,
)
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
        "learn_sigma": True,
    },
    "diffusion_process": {
        "num_diffusion_timesteps": 1000,
        "sigma_learned": True,
        "noise_schedule": "linear",
    },
    "weights_uris": {
        "model": "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/lsun_bedroom.pt",
    },
}


@click.command()
@click.option(
    "--n_samples", help="The number of samples to generate", required=True, type=int
)
@click.option("--batch_size", help="Batch size", required=True, type=int)
@click.option(
    "--device",
    help="Torch device",
    default="cuda",
    type=click.Choice(["cuda", "mps"], case_sensitive=False),
)
@click.option(
    "--output_folder", help="Folder to save the results to", default=".", type=str
)
@click.option(
    "--denoise_steps",
    help="The number of steps in the reverse process",
    required=True,
    type=int,
)
@click.option(
    "--use_ddim", help="If true use DDIM else use DDPM", required=True, type=bool
)
@click.option(
    "--save_partial_every",
    help="To define the frequency at which partial denoised images are saved (by default every 10 steps)",
    default=10,
    type=int,
)
def generate(
    n_samples: int,
    batch_size: int,
    device: Literal["mps", "cuda"],
    output_folder: str,
    denoise_steps: int,
    use_ddim: bool,
    save_partial_every: int,
) -> None:
    t0 = time.perf_counter()
    diffusion_kwargs = build_diffusion_kwargs(config, device=device)
    with torch.device(device):
        diffusion_process = GaussianDiffusion(
            timestep_respacing=denoise_steps,
            **diffusion_kwargs,
        )

        n_samples = n_samples
        batch_size = batch_size
        full = n_samples // batch_size
        batch_sizes = [batch_size for _ in range(full)]
        if n_samples % batch_size != 0:
            batch_sizes += [n_samples - full * batch_size]

        images = []
        with torch.no_grad():
            for batch_iter, batch_size in enumerate(batch_sizes):
                print(
                    f"Generated {sum(batch_sizes[:batch_iter])}/{sum(batch_sizes)} images. "
                    f"Current iter={batch_iter + 1}/{len(batch_sizes)}"
                )
                for output in diffusion_process.denoise(
                    batch_size=batch_size, use_ddim=use_ddim
                ):
                    denoised_x = output["denoised_x"]
                    timestep = output["timestep"]
                    if save_partial_every and timestep % save_partial_every == 0:
                        for i in range(batch_size):
                            img = denoised_x[i]
                            out_img = prepare_img_tensor_for_plot(img)
                            out_img_path = os.path.join(
                                output_folder,
                                f"lsun_bedroom_batch_{batch_iter}_sample_{i}_step_{timestep}.png",
                            )
                            print(f"Saving partial denoised image {out_img_path}")
                            plt.imsave(out_img_path, out_img)

                for i in range(batch_size):
                    images.append(prepare_img_tensor_for_plot(denoised_x[i]))

        fig = image_grid(images)
        grid_path = os.path.join(output_folder, "lsun_bedroom_grid.png")
        print(f"Saving grid {grid_path}")
        fig.savefig(grid_path)
        plt.clf()
        print(f"Done! Took {time.perf_counter() - t0:.2f} s")


if __name__ == "__main__":
    generate()
