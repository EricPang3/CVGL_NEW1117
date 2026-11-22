import math
import time
from contextlib import nullcontext

import torch
from tqdm import tqdm


def cosine_scheduler(optimizer, batches_per_epoch, config):
    """与 S4G 的 HF cosine scheduler 相同；步数在地点唯一采样前确定。"""
    total = batches_per_epoch * config["epochs"]
    warmup = batches_per_epoch * config["warmup_epochs"]

    def factor(step):
        if step < warmup:
            return step / max(1.0, warmup)
        progress = (step - warmup) / max(1.0, total - warmup)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def train_epoch(model, loader, optimizer, criterion, scaler, scheduler, config, epoch, device):
    model.train()
    total, count = 0.0, 0
    main_total, auxiliary_total = 0.0, 0.0
    started = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    skipped, first_lr = 0, optimizer.param_groups[0]["lr"]
    for satellite, drone, _ in tqdm(loader, desc=f"Epoch {epoch}"):
        last_lr = optimizer.param_groups[0]["lr"]
        satellite, drone = satellite.to(device), drone.to(device)
        optimizer.zero_grad(set_to_none=True)
        context = torch.autocast("cuda", dtype=torch.float16) if scaler.is_enabled() else nullcontext()
        with context:
            if config.get("auxiliary", "none") != "none":
                loss, main_loss, auxiliary_loss = model.training_loss(satellite, drone, criterion, config["auxiliary_weight"])
                main_total += main_loss.detach().item() * len(satellite)
                auxiliary_total += auxiliary_loss.detach().item() * len(satellite)
            else:
                sat_features, drone_features = model(satellite, drone)
                loss = criterion(sat_features, drone_features, model.logit_scale.exp())
        if not torch.isfinite(loss):
            raise RuntimeError(f"Epoch {epoch} 出现非有限 loss")
        scaler.scale(loss).backward()
        if config["clip_grad"]:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_value_(model.parameters(), config["clip_grad"])
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        skipped += int(scaler.get_scale() < previous_scale)
        # S4G 即使 AMP 跳过更新也推进 scheduler；保留这一行为并记录跳过次数。
        scheduler.step()
        total += loss.item() * len(satellite)
        count += len(satellite)
    result = {"epoch": epoch, "loss": total / count, "lr": last_lr, "first_lr": first_lr,
            "seconds": time.monotonic() - started, "pairs": count, "steps": len(loader),
            "optimizer_updates": len(loader) - skipped, "amp_skipped_steps": skipped,
            "amp_scale": scaler.get_scale(),
            "peak_gpu_mb": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
            "module_alpha": model.module_scale.detach().item() if hasattr(model, "module_scale") else None,
            "dropped_pairs": loader.batch_sampler.dropped,
            "logit_scale": model.logit_scale.detach().exp().item()}
    if config.get("auxiliary", "none") != "none":
        result.update(main_loss=main_total / count, auxiliary_loss=auxiliary_total / count, auxiliary_weight=config["auxiliary_weight"])
    return result
