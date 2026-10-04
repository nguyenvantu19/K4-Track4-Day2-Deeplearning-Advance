"""One reusable, validation-driven training loop for DeepWeeds experiments."""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    from . import dataset, losses, model as model_utils
except ImportError:  # Running `python starter/train.py`.
    import dataset
    import losses
    import model as model_utils
from eval import compute_metrics, save_predictions


@dataclass
class Config:
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    backbone: str = "resnet50"
    init: str = "finetune"
    drop_rate: float = 0.0
    img_size: int = 224
    aug: str = "basic"
    sampler: str | None = None
    mix: str | None = None
    mix_alpha: float = 1.0
    loss: str = "ce"
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    amp: bool = True
    num_workers: int = 2
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"
    pred_dir: str = "predictions"
    save_test_predictions: bool = False


def run_dir(cfg: Config) -> Path:
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    if split not in {"val", "test"}:
        raise ValueError("split must be val or test")
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def set_seed(seed: int) -> None:
    """Set Python, NumPy, CPU, CUDA, and DataLoader-compatible random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except AttributeError:
        pass


def build_optimizer(net: nn.Module, cfg: Config):
    return torch.optim.AdamW(model_utils.param_groups(net, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay))


def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    """Per-iteration linear warmup followed by cosine decay to zero."""
    if steps_per_epoch <= 0:
        raise ValueError("steps_per_epoch must be positive")
    total = max(1, cfg.epochs * steps_per_epoch)
    warmup = min(total, max(0, round(cfg.warmup_epochs * steps_per_epoch)))
    def multiplier(step: int) -> float:
        step = min(step + 1, total)
        if warmup and step <= warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


class EMA:
    """Exponential moving average of parameters and buffers."""
    def __init__(self, net: nn.Module, decay: float):
        if not 0 <= decay < 1:
            raise ValueError("EMA decay must be in [0, 1)")
        self.decay = float(decay)
        self.shadow = {key: value.detach().clone() for key, value in net.state_dict().items()}

    @torch.no_grad()
    def update(self, net: nn.Module) -> None:
        for key, value in net.state_dict().items():
            value = value.detach()
            if value.is_floating_point():
                self.shadow[key].mul_(self.decay).add_(value, alpha=1 - self.decay)
            else:
                self.shadow[key].copy_(value)

    def copy_to(self, net: nn.Module) -> None:
        net.load_state_dict(self.shadow, strict=True)


def _amp_context(device: torch.device, enabled: bool):
    return torch.autocast(device_type=device.type, enabled=enabled and device.type == "cuda")


def train_one_epoch(net, loader, criterion, optimizer, scheduler, scaler, cfg: Config,
                    device, ema: EMA | None = None) -> dict:
    net.train()
    model_utils.set_frozen_batchnorm_eval(net)
    device = torch.device(device)
    amp_enabled = bool(cfg.amp and device.type == "cuda")
    running_loss, samples = 0.0, 0
    for images, target, _ in loader:
        images, target = images.to(device, non_blocking=True), target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        mixed_targets = None
        if cfg.mix is not None:
            images, mixed_targets = losses.mix_batch(images, target, cfg.mix_alpha, cfg.mix)
        with _amp_context(device, amp_enabled):
            logits = net(images)
            loss = losses.mixed_loss(criterion, logits, mixed_targets) if mixed_targets else criterion(logits, target)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        if ema is not None:
            ema.update(net)
        running_loss += float(loss.detach()) * len(target)
        samples += len(target)
    return {"train_loss": running_loss / max(1, samples), "lr": optimizer.param_groups[0]["lr"]}


def evaluate(net, loader, criterion, device):
    """Return ordered filenames, labels, logits, and mean validation loss."""
    net.eval()
    device = torch.device(device)
    names, targets, logits_all = [], [], []
    total_loss, samples = 0.0, 0
    with torch.inference_mode():
        for images, target, filenames in loader:
            images, target = images.to(device, non_blocking=True), target.to(device, non_blocking=True)
            logits = net(images)
            total_loss += float(criterion(logits, target)) * len(target)
            samples += len(target)
            names.extend(str(name) for name in filenames)
            targets.append(target.cpu())
            logits_all.append(logits.cpu())
    if not logits_all:
        raise ValueError("cannot evaluate an empty loader")
    return names, torch.cat(targets).numpy(), torch.cat(logits_all).numpy(), total_loss / samples


def plot_curves(history: list[dict], path: str | Path, title: str) -> None:
    if not history:
        raise ValueError("history must not be empty")
    import matplotlib.pyplot as plt
    frame = pd.DataFrame(history)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(frame["epoch"], frame["train_loss"], label="train")
    axes[0].plot(frame["epoch"], frame["val_loss"], label="validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy")
    axes[0].legend()
    axes[1].plot(frame["epoch"], frame["macro_f1"], label="macro-F1")
    axes[1].set(title="Validation macro-F1", xlabel="Epoch", ylabel="Macro-F1")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    figure.suptitle(title)
    figure.tight_layout()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _probabilities(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def run(cfg: Config) -> dict:
    """Train and select by validation macro-F1; access test only when requested."""
    set_seed(cfg.seed)
    output = run_dir(cfg)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    train_df, val_df, test_df = dataset.load_split(cfg.labels_dir, cfg.fold)
    split_stats = dataset.check_split(train_df, val_df, test_df, cfg.images_dir)
    train_loader = dataset.make_loader(train_df, cfg.images_dir, dataset.build_transforms(True, cfg.img_size, cfg.aug),
                                       cfg.batch_size, True, cfg.sampler, cfg.num_workers)
    val_loader = dataset.make_loader(val_df, cfg.images_dir, dataset.build_transforms(False, cfg.img_size, cfg.aug),
                                     cfg.batch_size, False, None, cfg.num_workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = model_utils.build_model(cfg.backbone, num_classes=dataset.NUM_CLASSES, drop_rate=cfg.drop_rate,
                                  init=cfg.init).to(device)
    weight = None
    if cfg.loss == "ce_weighted":
        counts = train_df.Label.value_counts().reindex(range(dataset.NUM_CLASSES), fill_value=0).to_numpy()
        weight = losses.class_weights(counts, cfg.class_weight_beta or 0.0).to(device)
    criterion = losses.build_criterion(cfg.loss, smoothing=cfg.label_smoothing, gamma=cfg.focal_gamma, weight=weight)
    criterion = criterion.to(device) if isinstance(criterion, nn.Module) else criterion
    optimizer = build_optimizer(net, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    amp_enabled = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    ema = EMA(net, cfg.ema_decay) if cfg.ema_decay is not None else None
    history, best_f1, best_epoch = [], -float("inf"), 0
    checkpoint = output / "best.pt"
    started = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        row = {"epoch": epoch, **train_one_epoch(net, train_loader, criterion, optimizer, scheduler, scaler, cfg, device, ema)}
        evaluation_net = copy.deepcopy(net) if ema is not None else net
        if ema is not None:
            ema.copy_to(evaluation_net)
        _, labels, logits, val_loss = evaluate(evaluation_net, val_loader, criterion, device)
        metrics = compute_metrics(labels, logits.argmax(1), _probabilities(logits))
        row.update({"val_loss": val_loss, "macro_f1": float(metrics["macro_f1"]), "top1": float(metrics["top1"])})
        history.append(row)
        if row["macro_f1"] > best_f1:
            best_f1, best_epoch = row["macro_f1"], epoch
            torch.save({"epoch": epoch, "state_dict": evaluation_net.state_dict(), "macro_f1": best_f1}, checkpoint)
    elapsed = time.perf_counter() - started
    pd.DataFrame(history).to_csv(output / "history.csv", index=False)
    plot_curves(history, output / "curves.png", f"{cfg.exp_id}, seed {cfg.seed}")
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    net.load_state_dict(state["state_dict"])
    names, labels, logits, _ = evaluate(net, val_loader, criterion, device)
    np.save(output / "val_logits.npy", logits)
    save_predictions(pred_path(cfg, "val"), names, labels, _probabilities(logits))
    if cfg.save_test_predictions:
        test_loader = dataset.make_loader(test_df, cfg.images_dir, dataset.build_transforms(False, cfg.img_size, cfg.aug),
                                          cfg.batch_size, False, None, cfg.num_workers)
        names, labels, logits, _ = evaluate(net, test_loader, criterion, device)
        np.save(output / "test_logits.npy", logits)
        save_predictions(pred_path(cfg, "test"), names, labels, _probabilities(logits))
    return {"best_epoch": best_epoch, "macro_f1": best_f1, "seconds": elapsed,
            "seconds_per_epoch": elapsed / cfg.epochs, "params_m": model_utils.count_params(net),
            "gmacs": model_utils.count_gmacs(net, cfg.img_size), "split": split_stats}


def parse_overrides(pairs: list[str]) -> dict:
    """Parse ``KEY=VALUE`` CLI values using the type implied by Config defaults."""
    defaults = Config()
    optional_floats = {"class_weight_beta", "ema_decay"}
    optional_strings = {"sampler", "mix"}
    result = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Expected KEY=VALUE, got {pair!r}")
        key, raw = pair.split("=", 1)
        if not hasattr(defaults, key):
            raise ValueError(f"Unknown Config field: {key}")
        current = getattr(defaults, key)
        text = raw.strip()
        if text.lower() in {"none", "null"}:
            value = None
        elif isinstance(current, bool):
            if text.lower() not in {"true", "false"}:
                raise ValueError(f"{key} must be true or false")
            value = text.lower() == "true"
        elif isinstance(current, int):
            value = int(text)
        elif isinstance(current, float) or key in optional_floats:
            value = float(text)
        elif key in optional_strings:
            value = text
        else:
            value = text
        result[key] = value
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a DeepWeeds experiment")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="Config overrides")
    args = parser.parse_args()
    cfg = Config(**parse_overrides(args.set))
    print(json.dumps(run(cfg), indent=2, default=str))


if __name__ == "__main__":
    main()
