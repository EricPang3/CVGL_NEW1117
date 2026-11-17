"""单卡训练；最后一轮保存 model.pth，训练后可单独运行 evaluate.py。"""

import argparse
import csv
import platform
from datetime import datetime
from importlib.metadata import version

import torch
from torch.utils.data import DataLoader

from sample4geo.dataset import U1652DatasetTrain, UniqueBatchSampler, build_records, get_transforms
from sample4geo.loss import InfoNCE
from sample4geo.model import MODEL_NAME, TimmModel
from sample4geo.trainer import train_epoch
from sample4geo.utils import configure, experiment_config, load_weights, new_output, save_config, setup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="实验 YAML；缺省按 --dataset 选择")
    parser.add_argument("--dataset", choices=["U1652", "SUES200"])
    parser.add_argument("--name", help="outputs 下的实验名称，已有目录会拒绝覆盖")
    parser.add_argument("--direction", choices=["D2S", "S2D"])
    parser.add_argument("--altitude", type=int, choices=[150, 200, 250, 300])
    parser.add_argument("--data-root")
    parser.add_argument("--init-checkpoint", help="加载已有模型参数；不恢复优化器")
    for key in ("epochs", "batch-size", "workers", "img-size"):
        parser.add_argument(f"--{key}", type=int)
    parser.add_argument("--lr", type=float)
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
                   "training_pairs": len(drone), "checkpoint_policy": "last_epoch"})
    dataset = U1652DatasetTrain(satellite, drone, get_transforms(config, training=True))
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scaler = torch.cuda.amp.GradScaler(enabled=config["amp"] and device.type == "cuda")
    criterion = InfoNCE(config["label_smoothing"])
    output = new_output(config, args.name)
    save_config(config, output / "config.yaml")
    print(f"训练输出：{output}")
    fields = ["epoch", "loss", "lr", "seconds", "pairs", "steps", "dropped_pairs", "logit_scale"]
    with (output / "train.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, config["epochs"] + 1):
            sampler = UniqueBatchSampler(dataset.pairs, config["batch_size"], config["seed"] + epoch)
            loader = DataLoader(dataset, batch_sampler=sampler, num_workers=config["workers"],
                                pin_memory=device.type == "cuda")
            row = train_epoch(model, loader, optimizer, criterion, scaler, config, epoch, device)
            writer.writerow(row)
            stream.flush()
            temporary = output / "model.tmp"
            torch.save({"model": model.state_dict(), "config": config, "epoch": epoch}, temporary)
            temporary.replace(output / "model.pth")
            print(f'Epoch {epoch}: loss={row["loss"]:.4f}, pairs={row["pairs"]}, dropped={row["dropped_pairs"]}')
    print(f"训练完成。评估：python evaluate.py --checkpoint {output / 'model.pth'}")


if __name__ == "__main__":
    main()
