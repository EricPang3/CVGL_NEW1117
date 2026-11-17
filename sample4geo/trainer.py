import math
import time
from contextlib import nullcontext

import torch
from tqdm import tqdm


def train_epoch(model, loader, optimizer, criterion, scaler, config, epoch, device):
    model.train()
    total, count = 0.0, 0
    started = time.monotonic()
    for step, (satellite, drone, _) in enumerate(tqdm(loader, desc=f"Epoch {epoch}")):
        # 按实际 batch 数推进 warmup/cosine，避免采样丢弃造成计划步数偏差。
        progress = epoch - 1 + step / len(loader)
        warmup = config["warmup_epochs"]
        factor = min(1.0, (progress + 1 / len(loader)) / warmup) if progress < warmup else (1 + math.cos(math.pi * (progress - warmup) / max(config["epochs"] - warmup, 1e-8))) / 2
        for group in optimizer.param_groups:
            group["lr"] = config["lr"] * factor
        satellite, drone = satellite.to(device), drone.to(device)
        optimizer.zero_grad(set_to_none=True)
        context = torch.autocast("cuda", dtype=torch.float16) if scaler.is_enabled() else nullcontext()
        with context:
            sat_features, drone_features = model(satellite, drone)
            loss = criterion(sat_features, drone_features, model.logit_scale.exp())
        if not torch.isfinite(loss):
            raise RuntimeError(f"Epoch {epoch} 出现非有限 loss")
        scaler.scale(loss).backward()
        if config["clip_grad"]:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_value_(model.parameters(), config["clip_grad"])
        scaler.step(optimizer)
        scaler.update()
        total += loss.item() * len(satellite)
        count += len(satellite)
    return {"epoch": epoch, "loss": total / count, "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.monotonic() - started, "pairs": count, "steps": len(loader),
            "dropped_pairs": loader.batch_sampler.dropped,
            "logit_scale": model.logit_scale.detach().exp().item()}
