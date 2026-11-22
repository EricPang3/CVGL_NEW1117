"""固定训练地点内开发划分；不读取官方test，不改变既有SUES训练/测试划分。"""

import hashlib
import json
import random
from pathlib import Path

from ..utils import ROOT, read_config


def development_partition(config):
    fraction = config.get("development_fraction", 0.0)
    if not 0 < fraction < 1:
        raise ValueError("开发划分要求0 < development_fraction < 1")
    if config["dataset"] == "U1652":
        base = Path(config["data_root"]) / "train/satellite"
        ids = sorted(p.name for p in base.iterdir() if p.is_dir())
    else:
        ids = sorted(read_config(ROOT / "configs/sues200_split.yaml")["index"])
    if len(ids) < 3:
        raise ValueError("开发划分至少需要3个训练地点")
    shuffled = list(ids)
    random.Random(config.get("development_seed", 20261122)).shuffle(shuffled)
    count = min(len(ids) - 1, max(1, round(len(ids) * fraction)))
    development = set(shuffled[:count])
    training = set(ids) - development
    content = {"training_ids": sorted(training), "development_ids": sorted(development),
               "seed": config.get("development_seed", 20261122), "fraction": fraction}
    content["sha256"] = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
    return content
