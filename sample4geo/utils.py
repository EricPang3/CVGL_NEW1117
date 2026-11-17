import json
import os
import random
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


def configure(config, args):
    for key in ("direction", "altitude", "data_root", "device", "workers", "img_size",
                "epochs", "batch_size", "lr", "amp", "pretrained", "gallery_scope", "top_k"):
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
