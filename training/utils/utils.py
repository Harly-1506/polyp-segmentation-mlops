from typing import Optional

import numpy as np
import torch
from thop import clever_format, profile


def clip_gradient(optimizer: torch.optim.Optimizer, grad_clip: float) -> None:
    """Clips gradients to prevent exploding gradients."""
    torch.nn.utils.clip_grad_norm_(
        parameters=[
            p
            for group in optimizer.param_groups
            for p in group["params"]
            if p.grad is not None
        ],
        max_norm=grad_clip,
    )


def adjust_lr(
    optimizer: torch.optim.Optimizer,
    init_lr: float,
    epoch: int,
    decay_rate: float = 0.1,
    decay_epoch: int = 30,
) -> None:
    """Decays the learning rate by decay_rate every decay_epoch."""
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group["lr"] = init_lr * decay


class AvgMeter:
    """Class for tracking and averaging training losses or metrics."""

    def __init__(self, num: int = 40) -> None:
        self.num = num
        self.reset()

    def reset(self) -> None:
        self.val: float = 0.0
        self.avg: float = 0.0
        self.sum: float = 0.0
        self.count: int = 0
        self.losses: list = []

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self) -> float:
        recent_losses = self.losses[max(len(self.losses) - self.num, 0) :]
        return torch.mean(torch.tensor(recent_losses))


def cal_params(model: torch.nn.Module, input_tensor: torch.Tensor):
    """Calculates and returns model's FLOPs and parameter count."""
    flops, params = profile(model, inputs=(input_tensor,), verbose=False)
    flops_str, params_str = clever_format([flops, params], "%.3f")
    print(f"[Model Stats] FLOPs: {flops_str}, Params: {params_str}")
    return flops_str, params_str
