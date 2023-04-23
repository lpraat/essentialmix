import math

import matplotlib.pyplot as plt
import torch

from essentialmix.models.externals.guided_diffusion import UNetModel

from dataclasses import dataclass

T = 1000


def linear_betas(num_diffusion_timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02) -> torch.Tensor:
    scale = T / num_diffusion_timesteps
    return torch.linspace(
        beta_start * scale,
        beta_end * scale,
        num_diffusion_timesteps
    )


def cosine_betas(num_diffusion_timesteps: int, s: float = 0.008, max_beta: float = 0.999) -> torch.Tensor:
    alpha_bar = lambda t: math.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas)


class GaussianDiffusion:
    def __init__(self, model: UNetModel, num_diffusion_timesteps: int = T, noise_schedule: str = 'linear', sigma_learned=False):
        self.model = model
        self.num_diffusion_timesteps = num_diffusion_timesteps
        self.noise_schedule = noise_schedule
        self.betas: torch.Tensor = (
            linear_betas(num_diffusion_timesteps)
            if self.noise_schedule == 'linear'
            else cosine_betas(num_diffusion_timesteps)
        )
        self.alpha_t = 1 - self.betas
        self.alpha_bar = torch.cumprod(self.alpha_t, dim=0)
        self.alpha_bar_prev = torch.cat([torch.tensor([1.0]), self.alpha_bar[:-1]], dim=0)

        # beta tilde
        self.posterior_variance = ((1 - self.alpha_bar_prev) / (1 - self.alpha_bar)) * self.betas
        self.log_posterior_variance = torch.log(torch.cat([torch.tensor([self.posterior_variance[1]]), self.posterior_variance[1:]], dim=0))

        self.sigma_learned = sigma_learned

    def denoise_at_t(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        z = torch.randn_like(x_t)
        z[t == 0] = 0  # no noise at the last timestep

        t = torch.tensor([t] * x_t.shape[0])

        if self.sigma_learned:
            model_output = model(x_t, t.float())
            model_noise, v = torch.split(model_output, model_output.shape[1] // 2, dim=1)
            v = (v + 1) / 2  # v is between [-1, 1]
            log_variance_our = (
                v * torch.log(self.betas[t])
                + (1 - v) * self.log_posterior_variance[t]
            )
            x_prev = (1 / torch.sqrt(self.alpha_t[t])) * (
                x_t
                - ((1 - self.alpha_t[t]) / torch.sqrt(1 - self.alpha_bar[t])) * model_noise
            )
            x_prev += torch.exp(0.5*log_variance_our)*z
        else:
            raise NotImplementedError("Fixed sigma not implemented")
        return x_prev

    def denoise(self) -> torch.Tensor:
        x = torch.randn(size=(1, self.model.in_channels, self.model.image_size, self.model.image_size))
        x = x.clamp(-1.0, 1.0)
        for t in range(self.num_diffusion_timesteps)[::-1]:
            print(f"Denoise step: {t}")
            x = self.denoise_at_t(x, torch.tensor(t))
            if t % 20 == 0:
                yield (t, x)
        return x


def inspect_model(model: UNetModel) -> None:
    print(f"Image size: {model.image_size}")
    print(f"Input channels: {model.in_channels}")
    print(f"Model channels: {model.model_channels}")
    print(f"Output channels: {model.out_channels}")
    print(f"Num Res blocks: {model.num_res_blocks}")
    print(f"Attention resolutions: {model.attention_resolutions}")
    print(f"Channel multiplier: {model.channel_mult}")
    print(f"Use learned convolutions?: {model.conv_resample}")
    print(f"Num classes (if None it is not class-conditional): {model.num_classes}")


# class-unconditional lsun bedroom
lsun_bedroom_config = {
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
}


if __name__ == '__main__':
    DEVICE = 'mps'

    from essentialmix.models.externals.guided_diffusion import create_model, noise_model_defaults
    from collections import ChainMap

    diffusion_config = lsun_bedroom_config['diffusion_process']
    model_config = ChainMap(lsun_bedroom_config['model'], noise_model_defaults())

    with torch.device('mps'):
        model = create_model(**dict(model_config))
        diffusion_process = GaussianDiffusion(model=model, **lsun_bedroom_config['diffusion_process'])

        model.load_state_dict(
            torch.load(
                "/Users/lpraat/develop/essentialmix/essentialmix/experiments/guided_diffusion/lsun_bedroom.pt",
                map_location=DEVICE,
            )
        )
        model.eval()
        model = model.to('mps')
        if model_config['use_fp16']:
            model.convert_to_fp16()

        with torch.no_grad():
            for t, sample in diffusion_process.denoise():
                sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
                sample = sample.permute(0, 2, 3, 1)
                sample = sample.contiguous()
                out_img = sample.to('cpu').numpy().reshape(256, 256, 3)
                print("saving image: ", out_img.shape)
                plt.imsave(f"./prova_{t}.png", out_img)
