"""Classification losses and batch-level Mixup/CutMix augmentation."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def build_criterion(kind: str = "ce", **kw):
    """Build cross entropy, label smoothing, focal, or weighted cross entropy."""
    kind = kind.lower()
    if kind == "ce":
        return nn.CrossEntropyLoss(**{key: value for key, value in kw.items() if key in {"weight", "reduction"}})
    if kind == "ls":
        return LabelSmoothingCE(kw.get("smoothing", kw.get("label_smoothing", 0.1)))
    if kind == "focal":
        return FocalLoss(kw.get("gamma", 2.0), kw.get("alpha"))
    if kind == "ce_weighted":
        if kw.get("weight") is None:
            raise ValueError("ce_weighted requires a class-weight tensor")
        return nn.CrossEntropyLoss(weight=kw["weight"])
    raise ValueError("kind must be ce, ls, focal, or ce_weighted")


class LabelSmoothingCE(nn.Module):
    """Cross entropy with PyTorch's numerically stable uniform label smoothing."""
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        if not 0 <= smoothing < 1:
            raise ValueError("smoothing must be in [0, 1)")
        self.smoothing = float(smoothing)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, target, label_smoothing=self.smoothing)


class FocalLoss(nn.Module):
    """Multiclass focal loss. With ``gamma=0`` it exactly equals weighted CE."""
    def __init__(self, gamma: float = 2.0, alpha=None):
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be non-negative")
        self.gamma = float(gamma)
        if alpha is None:
            self.register_buffer("alpha", None)
        else:
            alpha = torch.as_tensor(alpha, dtype=torch.float32)
            if alpha.ndim != 1:
                raise ValueError("alpha must be a one-dimensional class-weight vector")
            self.register_buffer("alpha", alpha)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_prob = F.log_softmax(logits, dim=1)
        log_pt = log_prob.gather(1, target.long().unsqueeze(1)).squeeze(1)
        factor = (1.0 - log_pt.exp()).pow(self.gamma)
        loss = -factor * log_pt
        if self.alpha is not None:
            loss = loss * self.alpha.to(logits.device)[target.long()]
        return loss.mean()


def class_weights(counts, beta: float = 0.0):
    """Return normalized inverse-frequency or class-balanced weights."""
    count = torch.as_tensor(counts, dtype=torch.float32)
    if count.ndim != 1 or len(count) != 9 or (count <= 0).any():
        raise ValueError("counts must contain nine positive training-set counts")
    if not 0 <= beta < 1:
        raise ValueError("beta must be in [0, 1)")
    if beta == 0:
        weight = count.reciprocal()
    else:
        weight = (1 - beta) / (-torch.expm1(torch.log(torch.tensor(beta)) * count))
    return weight * (len(weight) / weight.sum())


def mix_batch(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0, mode: str = "cutmix"):
    """Apply Mixup or CutMix and return the two targets plus area-corrected lambda."""
    if x.ndim != 4 or len(x) != len(y):
        raise ValueError("x must be NCHW and y must contain one target per image")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if mode not in {"mixup", "cutmix"}:
        raise ValueError("mode must be mixup or cutmix")
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    perm = torch.randperm(x.size(0), device=x.device)
    y_a, y_b = y, y[perm]
    if mode == "mixup":
        return x * lam + x[perm] * (1.0 - lam), (y_a, y_b, lam)
    _, _, height, width = x.shape
    cut_ratio = math.sqrt(1.0 - lam)
    cut_w, cut_h = int(width * cut_ratio), int(height * cut_ratio)
    center_x = int(torch.randint(width, (1,), device=x.device))
    center_y = int(torch.randint(height, (1,), device=x.device))
    x1, x2 = max(0, center_x - cut_w // 2), min(width, center_x + cut_w // 2)
    y1, y2 = max(0, center_y - cut_h // 2), min(height, center_y + cut_h // 2)
    mixed = x.clone()
    mixed[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
    lam = 1.0 - ((x2 - x1) * (y2 - y1) / float(width * height))
    return mixed, (y_a, y_b, lam)


def mixed_loss(criterion, logits: torch.Tensor, targets) -> torch.Tensor:
    """Compute the lambda-weighted loss for the targets returned by ``mix_batch``."""
    y_a, y_b, lam = targets
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
