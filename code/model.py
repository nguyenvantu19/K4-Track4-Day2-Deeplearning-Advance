"""Model construction, freezing, parameter groups, and model-size helpers."""
from __future__ import annotations

import math

import torch
from torch import nn

SUGGESTED_BACKBONES = {
    "resnet50": "resnet50", "resnext50": "resnext50_32x4d", "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224", "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0", "mobilenetv3": "mobilenetv3_large_100",
}


def _classifier_module(model: nn.Module) -> nn.Module:
    if hasattr(model, "get_classifier"):
        classifier = model.get_classifier()
        if isinstance(classifier, nn.Module):
            return classifier
    for name in ("classifier", "head", "fc"):
        value = getattr(model, name, None)
        if isinstance(value, nn.Module):
            return value
    raise ValueError("Cannot identify classifier head; model must expose get_classifier(), classifier, head, or fc")


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune") -> nn.Module:
    """Create a timm classifier, replacing its head with ``num_classes`` outputs."""
    if init not in {"scratch", "frozen", "finetune"}:
        raise ValueError("init must be scratch, frozen, or finetune")
    if num_classes <= 1 or drop_rate < 0:
        raise ValueError("num_classes must exceed one and drop_rate must be non-negative")
    try:
        import timm
    except ImportError as exc:
        raise ImportError("build_model requires timm. Install it with `pip install timm`.") from exc
    model_name = SUGGESTED_BACKBONES.get(name, name)
    model = timm.create_model(model_name, pretrained=pretrained and init != "scratch",
                              num_classes=num_classes, drop_rate=drop_rate)
    model.backbone_name = model_name
    model.pretrained_tag = getattr(model, "pretrained_cfg", {}).get("tag", None)
    model._backbone_frozen = False
    if init == "frozen":
        freeze_backbone(model)
    return model


def freeze_backbone(model: nn.Module) -> None:
    """Freeze every parameter except the classifier, and mark frozen BN handling."""
    head_ids = {id(parameter) for parameter in _classifier_module(model).parameters()}
    for parameter in model.parameters():
        parameter.requires_grad = id(parameter) in head_ids
    model._backbone_frozen = True


def set_frozen_batchnorm_eval(model: nn.Module) -> None:
    """Keep BatchNorm statistics fixed after ``model.train()`` for frozen models."""
    if getattr(model, "_backbone_frozen", False):
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()


def param_groups(model: nn.Module, lr_backbone: float, lr_head: float, weight_decay: float):
    """Return decay-backbone, no-decay-backbone, and classifier-head parameter groups."""
    if min(lr_backbone, lr_head, weight_decay) < 0:
        raise ValueError("learning rates and weight decay must be non-negative")
    head_ids = {id(parameter) for parameter in _classifier_module(model).parameters()}
    groups = [
        {"params": [], "lr": lr_backbone, "weight_decay": weight_decay},
        {"params": [], "lr": lr_backbone, "weight_decay": 0.0},
        {"params": [], "lr": lr_head, "weight_decay": weight_decay},
    ]
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in head_ids:
            groups[2]["params"].append(parameter)
        elif parameter.ndim <= 1:
            groups[1]["params"].append(parameter)
        else:
            groups[0]["params"].append(parameter)
    return [group for group in groups if group["params"]]


def count_params(model: nn.Module) -> float:
    """Return all parameters (including frozen ones) in millions."""
    return sum(parameter.numel() for parameter in model.parameters()) / 1_000_000


def count_gmacs(model: nn.Module, img_size: int = 224) -> float:
    """Estimate GMACs with profiler FLOP accounting (two FLOPs per MAC)."""
    if img_size <= 0:
        raise ValueError("img_size must be positive")
    try:
        parameter = next(model.parameters())
        device = parameter.device
        dtype = parameter.dtype if parameter.is_floating_point() else torch.float32
    except StopIteration:
        device, dtype = torch.device("cpu"), torch.float32
    was_training = model.training
    model.eval()
    example = torch.zeros(1, 3, img_size, img_size, device=device, dtype=dtype)
    try:
        with torch.inference_mode(), torch.profiler.profile(with_flops=True) as profiler:
            model(example)
        flops = sum(event.flops for event in profiler.key_averages() if event.flops)
    except Exception as exc:
        raise RuntimeError("Unable to profile this model for GMACs") from exc
    finally:
        model.train(was_training)
    return float(flops) / 2_000_000_000
