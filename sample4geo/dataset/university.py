"""U1652 与原始 SUES200 共用的数据读取；不复制或改动数据集。"""

import random
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset, Sampler

from ..utils import ROOT, read_config

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def scan(root, ids=None, altitude=None):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"图片目录不存在：{root}")
    places = sorted(p for p in root.iterdir() if p.is_dir() and (ids is None or p.name in ids))
    if ids is not None and set(ids) != {p.name for p in places}:
        raise ValueError(f"{root} 缺少划分中的地点")
    records = []
    for place in places:
        directory = place / str(altitude) if altitude is not None else place
        files = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)
        if not files:
            raise ValueError(f"没有图片：{directory}")
        records.extend({"path": str(p.resolve()), "id": place.name} for p in files)
    if not records:
        raise ValueError(f"图片目录为空：{root}")
    return records


def build_records(config, training=False):
    root = Path(config["data_root"])
    def identify(records):
        return [{**r, "key": Path(r["path"]).relative_to(root.resolve()).as_posix()} for r in records]

    if config["dataset"] == "U1652":
        if training:
            return identify(scan(root / "train/satellite")), identify(scan(root / "train/drone"))
        base = root / "test"
        satellite, drone = ("query_satellite", "gallery_drone") if config["direction"] == "S2D" else ("gallery_satellite", "query_drone")
        sat, dro = scan(base / satellite), scan(base / drone)
    else:
        train_ids = set(read_config(ROOT / "configs/sues200_split.yaml")["index"])
        all_ids = {f"{i:04d}" for i in range(1, 201)}
        if len(train_ids) != 120 or not train_ids <= all_ids:
            raise ValueError("SUES200 划分必须包含 120 个合法且不重复的训练地点")
        test_ids = all_ids - train_ids
        if training:
            return identify(scan(root / "satellite-view", train_ids)), identify(scan(root / "drone_view_512", train_ids, config["altitude"]))
        sat_ids = test_ids if config["direction"] == "S2D" else all_ids
        dro_ids = test_ids if config["direction"] == "D2S" else all_ids
        sat = scan(root / "satellite-view", sat_ids)
        dro = scan(root / "drone_view_512", dro_ids, config["altitude"])
    return (identify(dro), identify(sat)) if config["direction"] == "D2S" else (identify(sat), identify(dro))


def read_image(path):
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"无法读取图片：{path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class U1652DatasetTrain(Dataset):
    def __init__(self, satellite, drone, transforms):
        satellites = {}
        for record in satellite:
            satellites.setdefault(record["id"], record["path"])
        if set(satellites) != {record["id"] for record in drone}:
            raise ValueError("训练两视角地点集合不一致")
        self.pairs = [(record["id"], satellites[record["id"]], record["path"]) for record in drone]
        self.sat_transform, self.drone_transform = transforms

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        label, satellite, drone = self.pairs[index]
        satellite, drone = read_image(satellite), read_image(drone)
        if np.random.random() < 0.5:
            satellite, drone = cv2.flip(satellite, 1), cv2.flip(drone, 1)
        return self.sat_transform(image=satellite)["image"], self.drone_transform(image=drone)["image"], label


class UniqueBatchSampler(Sampler):
    """保留原项目 batch 内地点唯一的约束和 512 次冲突停止规则。"""

    def __init__(self, pairs, batch_size, seed=None):
        if batch_size < 2 or len({p[0] for p in pairs}) < batch_size:
            raise ValueError("batch_size 至少为 2，且不能大于训练地点数")
        indices = list(range(len(pairs)))
        # 正式训练与 S4G 一样持续使用全局 Python RNG，不逐轮重新设 seed。
        (random if seed is None else random.Random(seed)).shuffle(indices)
        pool, batch, ids, conflicts = deque(indices), [], set(), 0
        self.batches = []
        while pool and conflicts < 512:
            index = pool.popleft()
            label = pairs[index][0]
            if label in ids:
                pool.append(index)
                conflicts += 1
                continue
            batch.append(index)
            ids.add(label)
            conflicts = 0
            if len(batch) == batch_size:
                self.batches.append(batch)
                batch, ids = [], set()
        self.dropped = len(pairs) - len(self.batches) * batch_size
        if not self.batches:
            raise ValueError("无法构成完整 batch；请减小 batch_size")

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


class U1652DatasetEval(Dataset):
    def __init__(self, records, transform):
        self.records, self.transform = records, transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.transform(image=read_image(self.records[index]["path"]))["image"]
