"""Inference, TTA aggregation, temperature calibration, and Conv-BN fusion."""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def predict_logits(model, loader, device, view=None):
    """Run inference without gradients and preserve loader filename order."""
    model.eval()
    filenames, labels, chunks = [], [], []
    view = view or view_identity
    device = torch.device(device)
    with torch.inference_mode():
        for images, target, names in loader:
            logits = model(view(images.to(device, non_blocking=True)))
            chunks.append(logits.detach().cpu())
            labels.append(target.detach().cpu())
            filenames.extend(str(name) for name in names)
    if not chunks:
        return filenames, np.empty(0, dtype=np.int64), np.empty((0, 0), dtype=np.float32)
    return filenames, torch.cat(labels).numpy(), torch.cat(chunks).numpy()


def view_identity(x):
    return x


def view_hflip(x):
    return torch.flip(x, dims=(-1,))


def views_multicrop(x, crop: int):
    """Return four corners and a centre crop; crops are not flipped implicitly."""
    if x.ndim != 4 or crop <= 0 or crop > min(x.shape[-2:]):
        raise ValueError("crop must fit the NCHW input")
    _, _, h, w = x.shape
    top, left = h - crop, w - crop
    centre_y, centre_x = top // 2, left // 2
    return [x[:, :, 0:crop, 0:crop], x[:, :, 0:crop, left:left + crop],
            x[:, :, top:top + crop, 0:crop], x[:, :, top:top + crop, left:left + crop],
            x[:, :, centre_y:centre_y + crop, centre_x:centre_x + crop]]


def views_multiscale(x, sizes):
    if x.ndim != 4 or not sizes or any(int(size) <= 0 for size in sizes):
        raise ValueError("sizes must be a non-empty sequence of positive integers")
    return [F.interpolate(x, size=(int(size), int(size)), mode="bilinear", align_corners=False) for size in sizes]


def _as_logits(values):
    arrays = [np.asarray(value, dtype=np.float64) for value in values]
    if not arrays:
        raise ValueError("at least one prediction array is required")
    if any(array.shape != arrays[0].shape or array.ndim != 2 for array in arrays):
        raise ValueError("all prediction arrays must have the same N x C shape")
    return arrays


def _softmax(values):
    values = values - values.max(axis=1, keepdims=True)
    exp = np.exp(values)
    return exp / exp.sum(axis=1, keepdims=True)


def aggregate_views(logits_per_view, space: str = "prob"):
    arrays = _as_logits(logits_per_view)
    if space == "prob":
        return np.mean([_softmax(array) for array in arrays], axis=0)
    if space == "logit":
        return _softmax(np.mean(arrays, axis=0))
    raise ValueError("space must be prob or logit")


def ensemble_probs(list_of_probs):
    arrays = _as_logits(list_of_probs)
    if any((array < -1e-7).any() or not np.allclose(array.sum(axis=1), 1, atol=1e-5) for array in arrays):
        raise ValueError("ensemble inputs must be normalized probabilities")
    result = np.mean(arrays, axis=0)
    return result / result.sum(axis=1, keepdims=True)


def fit_temperature(val_logits, val_labels) -> float:
    """Fit scalar temperature on validation logits using LBFGS over log(T)."""
    logits = torch.as_tensor(val_logits, dtype=torch.float64)
    labels = torch.as_tensor(val_labels, dtype=torch.long)
    if logits.ndim != 2 or len(logits) != len(labels) or len(labels) == 0:
        raise ValueError("val_logits must be non-empty N x C logits aligned with val_labels")
    log_temperature = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=100, line_search_fn="strong_wolfe")
    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / log_temperature.exp().clamp_min(1e-6), labels)
        loss.backward()
        return loss
    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(1e-3, 1e3))


def apply_temperature(logits, T: float):
    if not np.isfinite(T) or T <= 0:
        raise ValueError("T must be finite and positive")
    array = np.asarray(logits, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("logits must be N x C")
    return _softmax(array / T)


def _fuse_children(module: nn.Module) -> None:
    children = list(module.named_children())
    for index, (name, child) in enumerate(children):
        if isinstance(child, nn.Conv2d) and index + 1 < len(children) and isinstance(children[index + 1][1], nn.BatchNorm2d):
            bn_name, bn = children[index + 1]
            setattr(module, name, torch.nn.utils.fuse_conv_bn_eval(child, bn))
            setattr(module, bn_name, nn.Identity())
        else:
            _fuse_children(child)


def fuse_conv_bn(model):
    """Fuse adjacent ``Conv2d``/``BatchNorm2d`` pairs in place and return model."""
    model.eval()
    _fuse_children(model)
    return model
