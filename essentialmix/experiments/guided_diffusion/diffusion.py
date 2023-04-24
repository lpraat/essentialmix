import math

import matplotlib.pyplot as plt
import torch

from essentialmix.models.externals.guided_diffusion import UNetModel

from dataclasses import dataclass


def linear_betas(num_diffusion_timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02) -> torch.Tensor:
    return torch.linspace(
        beta_start,
        beta_end,
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
    def __init__(self,
        model: UNetModel, num_diffusion_timesteps: int,
        noise_schedule: str = 'linear', sigma_learned: bool = False,
        timestep_respacing: int | None = None,
    ):
        self.model = model
        self.num_diffusion_timesteps = num_diffusion_timesteps
        self.timestep_respacing = timestep_respacing

        self.noise_schedule = noise_schedule
        self.betas: torch.Tensor = (
            linear_betas(num_diffusion_timesteps)
            if self.noise_schedule == 'linear'
            else cosine_betas(num_diffusion_timesteps)
        )

        if self.timestep_respacing is not None:
            self.steps = torch.linspace(0, self.num_diffusion_timesteps - 1, self.timestep_respacing, dtype=torch.int32)
            new_alpha_bar = torch.cumprod(1 - self.betas, dim=0)[self.steps]
            new_alpha_bar_prev = torch.cat([torch.tensor([1.0]), new_alpha_bar[:-1]])
            new_betas = 1 - new_alpha_bar / new_alpha_bar_prev
            self.betas = new_betas
        else:
            self.steps = None

        # The number of denoising steps (can be < num_diffusion_steps if we use timestep_respacing)
        self.denoise_timesteps = self.betas.shape[0]

        def _check_shape(tensor):
            assert tensor.shape == (self.denoise_timesteps,)

        self.alpha_t = 1 - self.betas
        self.alpha_bar = torch.cumprod(self.alpha_t, dim=0)
        self.alpha_bar_prev = torch.cat([torch.tensor([1.0]), self.alpha_bar[:-1]], dim=0)
        _check_shape(self.alpha_t)
        _check_shape(self.alpha_bar)
        _check_shape(self.alpha_bar_prev)

        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alpha_bar)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alpha_bar - 1)
        _check_shape(self.sqrt_recip_alphas_cumprod)
        _check_shape(self.sqrt_recipm1_alphas_cumprod)

        self.posterior_variance = ((1 - self.alpha_bar_prev) / (1 - self.alpha_bar)) * self.betas
        self.log_posterior_variance = torch.log(torch.cat([torch.tensor([self.posterior_variance[1]]), self.posterior_variance[1:]], dim=0))
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alpha_bar_prev) / (1.0 - self.alpha_bar)
        self.posterior_mean_coef2 = (1.0 - self.alpha_bar_prev) * torch.sqrt(self.alpha_t) / (1 - self.alpha_bar)
        _check_shape(self.posterior_variance)
        _check_shape(self.log_posterior_variance)
        _check_shape(self.posterior_mean_coef1)
        _check_shape(self.posterior_mean_coef2)

        self.sigma_learned = sigma_learned

    def call_model(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        assert x_t.shape[0] == t.shape[0]
        if self.steps is not None:
            # In case we use re-spacing, pick the correct corresponding t
            return self.model(x_t, self.steps[t])
        return self.model(x_t, t)

    def index_and_broadcast(self, tensor: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return tensor[t].view([t.shape[0], 1, 1, 1])

    def ddpm_denoise_at_t(self, x_t: torch.Tensor, t: int, pred_x_0: bool = True) -> torch.Tensor:
        # Assumption: t is equal for all the samples in the batch x_t
        if t == 0:
            z = 0  # No noise at the last timestep
        else:
            z = torch.randn_like(x_t)

        t = torch.tensor([t] * x_t.shape[0])
        if self.sigma_learned:
            num_channels = x_t.shape[1]
            model_output = self.call_model(x_t, t)
            assert model_output.shape[1] == num_channels * 2
            model_noise, v = torch.split(model_output, model_output.shape[1] // 2, dim=1)

            v = (v + 1) / 2  # v is between [-1, 1]
            log_variance = (
                v * torch.log(self.index_and_broadcast(self.betas, t))
                + (1 - v) * self.index_and_broadcast(self.log_posterior_variance, t)
            )

            # Posterior mean
            if pred_x_0:
                # From x_0
                x_0 = (
                    self.index_and_broadcast(self.sqrt_recip_alphas_cumprod, t) * x_t
                    - self.index_and_broadcast(self.sqrt_recipm1_alphas_cumprod, t) * model_noise
                )
                x_prev = (
                    self.index_and_broadcast(self.posterior_mean_coef1, t) * x_0
                    + self.index_and_broadcast(self.posterior_mean_coef2, t) * x_t
                )
            else:
                # Direct formula
                alpha_t = self.index_and_broadcast(self.alpha_t, t)
                x_prev = (1 / torch.sqrt(alpha_t)) * (
                    x_t
                    - ((1 - alpha_t) / torch.sqrt(1 - self.index_and_broadcast(self.alpha_bar, t))) * model_noise
                )

            x_prev += torch.exp(0.5 * log_variance) * z
        else:
            raise NotImplementedError("Fixed sigma not implemented")
        return x_prev

    def ddim_denoise_at_t(self, x_t: torch.Tensor, t: int, eta: float = 0.0) -> torch.Tensor:
        # Assumption: t is equal for all the samples in the batch x_t
        if t == 0:
            z = 0  # No noise at the last timestep
        else:
            z = torch.randn_like(x_t)

        t = torch.tensor([t] * x_t.shape[0])
        if self.sigma_learned:
            num_channels = x_t.shape[1]
            model_output = self.call_model(x_t, t)
            model_noise, _ = torch.split(model_output, model_output.shape[1] // 2, dim=1)
            assert model_noise.shape[1] == num_channels

            x_0 = (
                self.index_and_broadcast(self.sqrt_recip_alphas_cumprod, t) * x_t
                - self.index_and_broadcast(self.sqrt_recipm1_alphas_cumprod, t) * model_noise
            )

            # We can use this instead of model noise in the following
            # eps = (
            #     (self.index_and_broadcast(self.sqrt_recip_alphas_cumprod, t) * x_t - x_0)
            #     / self.index_and_broadcast(self.sqrt_recipm1_alphas_cumprod, t)
            # )

            # Eta interpolates between DDIM (eta=0) and DDPM (eta=1)
            alpha_bar = self.index_and_broadcast(self.alpha_bar, t)
            alpha_bar_prev = self.index_and_broadcast(self.alpha_bar_prev, t)
            sigma = (
                eta
                * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
                * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
            )

            x_prev = (
                torch.sqrt(alpha_bar_prev) * x_0
                + torch.sqrt(1 - alpha_bar_prev - sigma**2) * model_noise
            )
            x_prev += sigma * z
        else:
            raise NotImplementedError("Fixed sigma not implemented")
        return x_prev

    def denoise(self, batch_size: int = 1, use_ddim=False) -> dict:
        x = torch.randn(size=(batch_size, self.model.in_channels, self.model.image_size, self.model.image_size))
        denoise_fn = self.ddim_denoise_at_t if use_ddim else self.ddpm_denoise_at_t
        for t in range(self.denoise_timesteps)[::-1]:
            print(f"{'DDIM' if use_ddim else 'DDPM'} denoise timestep: {t}")
            x = denoise_fn(x, t)
            yield {
                'timestep': t,
                'denoised_x': x,
            }


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
    from essentialmix.models.externals.guided_diffusion import create_model, noise_model_defaults
    from collections import ChainMap

    DEVICE = 'mps'

    diffusion_config = lsun_bedroom_config['diffusion_process']
    model_config = ChainMap(lsun_bedroom_config['model'], noise_model_defaults())

    with torch.device(DEVICE):
        model = create_model(**dict(model_config))
        diffusion_process = GaussianDiffusion(
            timestep_respacing=25,
            model=model,
            **lsun_bedroom_config['diffusion_process']
        )

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

        batch_size = 1
        with torch.no_grad():
            for output in diffusion_process.denoise(batch_size=batch_size, use_ddim=True):
                denoised_x = output['denoised_x']
                timestep = output['timestep']
                if timestep % 10 == 0:
                    for i in range(batch_size):
                        img = denoised_x[i]
                        img = ((img + 1) * 127.5).clamp(0, 255).to(torch.uint8)
                        img = img.permute(1, 2, 0)
                        img = img.contiguous()
                        out_img = img.to('cpu').numpy().reshape(256, 256, 3)
                        print("saving image: ", out_img.shape)
                        plt.imsave(f"./batch_{i}_{timestep}.png", out_img)
