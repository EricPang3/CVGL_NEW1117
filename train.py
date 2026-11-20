"""单卡训练；最后一轮保存 model.pth，训练后可单独运行 evaluate.py。"""

import argparse
import csv
import math
import platform
import hashlib
import subprocess
import zipfile
from datetime import datetime
from importlib.metadata import version

import torch
from torch.utils.data import DataLoader

from sample4geo.dataset import U1652DatasetTrain, UniqueBatchSampler, build_records, get_transforms
from sample4geo.loss import InfoNCE
from sample4geo.model import MODEL_NAME, TimmModel
from sample4geo.trainer import cosine_scheduler, train_epoch
from sample4geo.utils import ROOT, configure, experiment_config, load_weights, new_output, records_hash, save_config, setup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="实验 YAML；缺省按 --dataset 选择")
    parser.add_argument("--dataset", choices=["U1652", "SUES200"])
    parser.add_argument("--name", help="outputs 下的实验名称，已有目录会拒绝覆盖")
    parser.add_argument("--direction", choices=["D2S", "S2D"])
    parser.add_argument("--altitude", type=int, choices=[150, 200, 250, 300])
    parser.add_argument("--data-root")
    parser.add_argument("--init-checkpoint", help="加载已有模型参数；不恢复优化器")
    for key in ("epochs", "batch-size", "workers", "img-size", "seed"):
        parser.add_argument(f"--{key}", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--augmentation", choices=["s4g", "camp"])
    parser.add_argument("--device")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    config = configure(experiment_config(args), args)
    device = setup(config)
    satellite, drone = build_records(config, training=True)
    model = TimmModel(pretrained=config["pretrained"] and not args.init_checkpoint)
    if args.init_checkpoint:
        state, _ = load_weights(args.init_checkpoint)
        model.load_state_dict(state, strict=True)
        del state
    data = model.data_config()
    config.update({"model": MODEL_NAME, "pool": "avg", "shared_encoder": True,
                   "mean": list(data["mean"]), "std": list(data["std"]),
                   "started_at": datetime.now().astimezone().isoformat(),
                   "torch": str(torch.__version__), "python": platform.python_version(),
                   "torchvision": version("torchvision"), "system": platform.platform(),
                   "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                   "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
                   "amp_enabled": bool(config["amp"] and device.type == "cuda"),
                   "initialization": "checkpoint" if args.init_checkpoint else "imagenet" if config["pretrained"] else "random",
                   "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
                   "training_ids": sorted({r["id"] for r in satellite}),
                   "training_pairs": len(drone), "checkpoint_policy": "last_epoch",
                   "training_protocol": "s4g_batch24_v2",
                   "training_satellite_manifest": records_hash(satellite),
                   "training_drone_manifest": records_hash(drone),
                   "file_order": "sorted_relative_path", "sampler_rng": "global_python",
                   "synchronized_flip_rng": "numpy", "scheduler": "s4g_cosine",
                   "planned_steps_per_epoch": math.ceil(len(drone) / config["batch_size"]),
                   "timm": version("timm"), "albumentations": version("albumentations")})
    if config["dataset"] == "SUES200":
        config["split_sha256"] = hashlib.sha256((ROOT / "configs/sues200_split.yaml").read_bytes()).hexdigest()
    dataset = U1652DatasetTrain(satellite, drone, get_transforms(config, training=True))
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scaler = torch.cuda.amp.GradScaler(init_scale=config["amp_init_scale"], enabled=config["amp"] and device.type == "cuda")
    scheduler = cosine_scheduler(optimizer, config["planned_steps_per_epoch"], config)
    criterion = InfoNCE(config["label_smoothing"])
    output = new_output(config, args.name)
    # 保存实际源码，即使未 commit 的实验也能追溯；不包含数据、权重或历史输出。
    source_files = [*ROOT.glob("*.py"), *ROOT.glob("*.txt"), *ROOT.glob("*.md"),
                    *ROOT.glob("configs/*.yaml"), *ROOT.glob("sample4geo/**/*.py"),
                    *ROOT.glob("scripts/*.py"), *ROOT.glob("tests/*.py")]
    with zipfile.ZipFile(output / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source_files):
            archive.write(path, path.relative_to(ROOT))
    config["source_sha256"] = hashlib.sha256((output / "source.zip").read_bytes()).hexdigest()
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True) if (ROOT / ".git").exists() else None
    config["git_commit"] = git.stdout.strip() if git and git.returncode == 0 else None
    save_config(config, output / "config.yaml")
    print(f"训练输出：{output}")
    fields = ["epoch", "loss", "lr", "first_lr", "seconds", "pairs", "steps", "optimizer_updates",
              "amp_skipped_steps", "amp_scale", "dropped_pairs", "logit_scale"]
    with (output / "train.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, config["epochs"] + 1):
            sampler = UniqueBatchSampler(dataset.pairs, config["batch_size"])
            loader = DataLoader(dataset, batch_sampler=sampler, num_workers=config["workers"],
                                pin_memory=device.type == "cuda")
            row = train_epoch(model, loader, optimizer, criterion, scaler, scheduler, config, epoch, device)
            writer.writerow(row)
            stream.flush()
            temporary = output / "model.tmp"
            torch.save({"model": model.state_dict(), "config": config, "epoch": epoch}, temporary)
            temporary.replace(output / "model.pth")
            print(f'Epoch {epoch}: loss={row["loss"]:.4f}, pairs={row["pairs"]}, dropped={row["dropped_pairs"]}')
    print(f"训练完成。评估：python evaluate.py --checkpoint {output / 'model.pth'}")


if __name__ == "__main__":
    main()
