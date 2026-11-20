import json
import hashlib
import os
import random
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


def read_config(path):
    with Path(path).expanduser().open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def experiment_config(args, saved=None):
    if args.config:
        config = read_config(args.config)
    elif args.dataset:
        filename = "u1652.yaml" if args.dataset == "U1652" else "sues200.yaml"
        config = read_config(ROOT / "configs" / filename)
    else:
        config = dict(saved) if saved else read_config(ROOT / "configs/u1652.yaml")
    if args.dataset and config["dataset"] != args.dataset:
        raise ValueError("--dataset 与 --config 中的数据集不一致")
    return config


def save_config(config, path):
    Path(path).write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")


def save_json(data, path):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def records_hash(records):
    """相对路径和地点的指纹；跨机器一致，不代表图片内容哈希。"""
    manifest = sorted((r["key"], r["id"]) for r in records)
    return hashlib.sha256(json.dumps(manifest, ensure_ascii=False).encode()).hexdigest()


def snapshot_source(output):
    files = [*ROOT.glob("*.py"), *ROOT.glob("*.txt"), *ROOT.glob("*.md"), *ROOT.glob("configs/*.yaml"),
             *ROOT.glob("sample4geo/**/*.py"), *ROOT.glob("scripts/*.py"), *ROOT.glob("tests/*.py"),
             *[p for p in (ROOT / "third_party").rglob("*") if p.is_file() and "__pycache__" not in p.parts]]
    with zipfile.ZipFile(output / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path, path.relative_to(ROOT))
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True) if (ROOT / ".git").exists() else None
    return {"source_sha256": hashlib.sha256((output / "source.zip").read_bytes()).hexdigest(),
            "git_commit": git.stdout.strip() if git and git.returncode == 0 else None}


def configure(config, args):
    for key in ("direction", "altitude", "data_root", "device", "workers", "img_size",
                "epochs", "batch_size", "lr", "amp", "pretrained", "gallery_scope", "top_k", "seed", "augmentation", "module", "module_alpha", "freeze_backbone"):
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value
    # CLI > 当前机器的数据根目录 > YAML/checkpoint 中的路径。
    if not getattr(args, "data_root", None) and os.environ.get("CVGL_DATA_ROOT"):
        config["data_root"] = str(Path(os.environ["CVGL_DATA_ROOT"]).expanduser() / config["dataset"])
    config["data_root"] = str(Path(config["data_root"]).expanduser().resolve())
    if config["dataset"] not in ("U1652", "SUES200"):
        raise ValueError("dataset 必须是 U1652 或 SUES200")
    if config["direction"] not in ("D2S", "S2D"):
        raise ValueError("direction 必须是 D2S 或 S2D")
    if config["dataset"] == "SUES200" and config.get("altitude") not in (150, 200, 250, 300):
        raise ValueError("SUES200 高度必须为 150/200/250/300")
    if config["dataset"] == "U1652":
        config["altitude"] = None
    if config["gallery_scope"] not in ("all", "query_ids"):
        raise ValueError("gallery_scope 必须是 all 或 query_ids")
    # 旧 checkpoint 未记录这些字段；训练历史仍由原 checkpoint/config 保留。
    config.setdefault("augmentation", "s4g")
    config.setdefault("amp_init_scale", 1024)
    config.setdefault("module", "none")
    config.setdefault("module_alpha", 1.0 if config["module"] == "burst_gap" else 0.1)
    config.setdefault("freeze_backbone", False)
    if config["augmentation"] not in ("s4g", "camp") or config["amp_init_scale"] <= 0:
        raise ValueError("augmentation 必须为 s4g/camp，amp_init_scale 必须大于 0")
    for key in ("img_size", "batch_size", "eval_batch_size", "epochs", "top_k"):
        if config[key] < 1:
            raise ValueError(f"{key} 必须大于 0")
    if config["img_size"] < 32 or config["workers"] < 0 or config["lr"] <= 0:
        raise ValueError("img_size 至少为 32，workers 非负，lr 大于 0")
    if not 0 <= config["warmup_epochs"] < config["epochs"]:
        raise ValueError("warmup_epochs 必须非负且小于 epochs")
    return config


def setup(config):
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    device = torch.device(config["device"])
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用；请安装 GPU 版 PyTorch，或指定 --device cpu 做小规模检查")
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(config["seed"])
    elif device.type != "cpu":
        raise ValueError("仅支持单张 CUDA GPU 或 CPU")
    torch.backends.cudnn.benchmark = False
    return device


def new_output(config, name=None):
    name = name or f'{config["dataset"]}_{config["direction"]}_{config.get("altitude") or "na"}_{datetime.now():%Y%m%d_%H%M%S_%f}'
    if Path(name).name != name or name in (".", ".."):
        raise ValueError("name 必须是单个目录名")
    output = ROOT / "outputs" / name
    output.mkdir(parents=True, exist_ok=False)
    return output


def load_weights(path):
    loaded = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=True)
    state = loaded["model"] if "config" in loaded and "model" in loaded else loaded
    state = {key.removeprefix("module."): value for key, value in state.items()}
    return state, loaded.get("config", {})
