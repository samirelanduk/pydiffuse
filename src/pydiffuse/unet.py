import safetensors
import torch


def unet(
    latent: torch.Tensor,
    noise_level: float,
    conditioning: torch.Tensor,
    model: safetensors.safe_open,
) -> torch.Tensor:
    pass
