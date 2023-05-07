import math
import time

import torch
import torch.nn.functional as F

from typing import Generator

from essentialmix.models.externals.guided_diffusion import UNetModel, EncoderUNetModel


def linear_betas(
    num_diffusion_timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02
) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, num_diffusion_timesteps)


def cosine_betas(
    num_diffusion_timesteps: int, s: float = 0.008, max_beta: float = 0.999
) -> torch.Tensor:
    def alpha_bar(t):
        return math.cos((t + s) / (1 + s) * math.pi / 2) ** 2

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas)


class GaussianDiffusion:
    def __init__(
        self,
        model: UNetModel,
        num_diffusion_timesteps: int,
        classifier: EncoderUNetModel | None = None,
        class_guidance_scale: float = 1.0,
        noise_schedule: str = "linear",
        sigma_learned: bool = False,
        timestep_respacing: int | None = None,
    ):
        self.model = model
        self.classifier = classifier
        self.class_guidance_scale = class_guidance_scale
        self.num_diffusion_timesteps = num_diffusion_timesteps
        self.timestep_respacing = timestep_respacing

        self.noise_schedule = noise_schedule
        self.betas: torch.Tensor = (
            linear_betas(num_diffusion_timesteps)
            if self.noise_schedule == "linear"
            else cosine_betas(num_diffusion_timesteps)
        )

        if self.timestep_respacing is not None:
            # https://arxiv.org/pdf/2102.09672.pdf Section 4
            self.steps = torch.linspace(
                0,
                self.num_diffusion_timesteps - 1,
                self.timestep_respacing,
                dtype=torch.int32,
            )
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
        self.alpha_bar_prev = torch.cat(
            [torch.tensor([1.0]), self.alpha_bar[:-1]], dim=0
        )
        _check_shape(self.alpha_t)
        _check_shape(self.alpha_bar)
        _check_shape(self.alpha_bar_prev)

        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alpha_bar)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alpha_bar - 1)
        _check_shape(self.sqrt_recip_alphas_cumprod)
        _check_shape(self.sqrt_recipm1_alphas_cumprod)

        self.posterior_variance = (
            (1 - self.alpha_bar_prev) / (1 - self.alpha_bar)
        ) * self.betas
        self.log_posterior_variance = torch.log(
            torch.cat(
                [
                    torch.tensor([self.posterior_variance[1]]),
                    self.posterior_variance[1:],
                ],
                dim=0,
            )
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alpha_bar_prev) / (1.0 - self.alpha_bar)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alpha_bar_prev)
            * torch.sqrt(self.alpha_t)
            / (1 - self.alpha_bar)
        )
        _check_shape(self.posterior_variance)
        _check_shape(self.log_posterior_variance)
        _check_shape(self.posterior_mean_coef1)
        _check_shape(self.posterior_mean_coef2)

        self.sigma_learned = sigma_learned

    def q_sample(self, x_0: torch.Tensor, t: int):
        """Forward process q(x_t | x_0)"""
        t = torch.tensor([t] * x_0.shape[0])
        alpha_bar = self.index_and_broadcast(self.alpha_bar, t)
        x_t = torch.sqrt(alpha_bar) * x_0
        x_t += torch.sqrt(1 - alpha_bar) * torch.randn_like(x_0)
        return x_t

    def call_model(
        self, x_t: torch.Tensor, t: torch.Tensor, y: torch.Tensor | None
    ) -> torch.Tensor:
        assert x_t.shape[0] == t.shape[0]
        if self.steps is not None:
            # In case we use re-spacing, pick the correct corresponding t
            return self.model(x_t, self.steps[t], y)
        return self.model(x_t, t, y)

    def index_and_broadcast(
        self, tensor: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        return tensor[t].view([t.shape[0], 1, 1, 1])

    def class_gradient_guide(self, x, t, y):
        """
        grad(pϕ(y | x_t))
        """
        if self.classifier is None:
            raise RuntimeError(
                "Trying to use class gradient but the classifier model is not set"
            )

        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = self.classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            return (
                torch.autograd.grad(selected.sum(), x_in)[0] * self.class_guidance_scale
            )

    def ddpm_denoise_at_t(
        self,
        x_t: torch.Tensor,
        t: int,
        y: torch.Tensor | None = None,
        pred_x_0: bool = True,
    ) -> torch.Tensor:
        """
        DDPM reverse process pθ(x_{t-1} | x_t)
        """
        # Assumption: t is equal for all the samples in the batch x_t
        if t == 0:
            z = 0  # No noise at the last timestep
        else:
            z = torch.randn_like(x_t)

        t = torch.tensor([t] * x_t.shape[0])
        if self.sigma_learned:
            num_channels = x_t.shape[1]
            model_output = self.call_model(x_t, t, y)
            assert model_output.shape[1] == num_channels * 2
            model_noise, v = torch.split(
                model_output, model_output.shape[1] // 2, dim=1
            )

            v = (v + 1) / 2  # v is between [-1, 1]
            log_variance = v * torch.log(self.index_and_broadcast(self.betas, t)) + (
                1 - v
            ) * self.index_and_broadcast(self.log_posterior_variance, t)

            # Posterior mean
            if pred_x_0:
                # From x_0
                x_0 = (
                    self.index_and_broadcast(self.sqrt_recip_alphas_cumprod, t) * x_t
                    - self.index_and_broadcast(self.sqrt_recipm1_alphas_cumprod, t)
                    * model_noise
                )
                x_0.clamp_(-1.0, 1.0)

                x_prev = (
                    self.index_and_broadcast(self.posterior_mean_coef1, t) * x_0
                    + self.index_and_broadcast(self.posterior_mean_coef2, t) * x_t
                )
            else:
                # Direct formula
                alpha_t = self.index_and_broadcast(self.alpha_t, t)
                x_prev = (1 / torch.sqrt(alpha_t)) * (
                    x_t
                    - (
                        (1 - alpha_t)
                        / torch.sqrt(1 - self.index_and_broadcast(self.alpha_bar, t))
                    )
                    * model_noise
                )

            if y is not None:
                # Class guidance
                # This is probably wrong: x_prev.add_(self.class_gradient_guide(x_t, t, y) * torch.exp(log_variance))
                x_prev.add_(
                    self.class_gradient_guide(x_prev, t - 1, y)
                    * torch.exp(log_variance)
                )

            x_prev += torch.exp(0.5 * log_variance) * z
        else:
            raise NotImplementedError("Fixed sigma not implemented")
        return x_prev

    def ddim_denoise_at_t(
        self, x_t: torch.Tensor, t: int, y: torch.Tensor | None = None, eta: float = 0.0
    ) -> torch.Tensor:
        """
        DDDIM reverse process pθ(x_{t-1} | x_{t})
        """
        # Assumption: t is equal for all the samples in the batch x_t
        if t == 0:
            z = 0  # No noise at the last timestep
        else:
            z = torch.randn_like(x_t)

        t = torch.tensor([t] * x_t.shape[0])
        if self.sigma_learned:
            num_channels = x_t.shape[1]
            model_output = self.call_model(x_t, t, y)
            model_noise, _ = torch.split(
                model_output, model_output.shape[1] // 2, dim=1
            )
            assert model_noise.shape[1] == num_channels

            if y is not None:
                # Class guidance
                model_noise.add_(
                    -torch.sqrt(1 - self.index_and_broadcast(self.alpha_bar, t))
                    * self.class_gradient_guide(x_t, t, y)
                )

            x_0 = (
                self.index_and_broadcast(self.sqrt_recip_alphas_cumprod, t) * x_t
                - self.index_and_broadcast(self.sqrt_recipm1_alphas_cumprod, t)
                * model_noise
            )
            x_0.clamp_(-1.0, 1.0)

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

    def denoise(
        self,
        x_0: torch.Tensor | None = None,
        batch_size: int = 1,
        y: torch.Tensor | None = None,
        use_ddim=False,
    ) -> Generator:
        if x_0 is not None:
            x = x_0
        else:
            x = torch.randn(
                size=(
                    batch_size,
                    self.model.in_channels,
                    self.model.image_size,
                    self.model.image_size,
                )
            )

        assert batch_size == x.shape[0]

        if y is not None:
            assert y.shape[0] == x.shape[0]

        denoise_fn = self.ddim_denoise_at_t if use_ddim else self.ddpm_denoise_at_t
        for t in range(self.denoise_timesteps)[::-1]:
            t0 = time.perf_counter()
            x = denoise_fn(x, t, y)
            print(
                f"Completed {'DDIM' if use_ddim else 'DDPM'} denoise at step: {t}. Took {time.perf_counter() - t0:.2f} s"
            )
            yield {
                "timestep": t,
                "denoised_x": x,
            }
