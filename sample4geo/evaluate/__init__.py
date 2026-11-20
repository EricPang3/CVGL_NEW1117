"""特征提取和检索；训练结束与独立入口使用同一实现。"""

from contextlib import nullcontext

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..dataset import U1652DatasetEval, build_records, get_transforms
from .metrics import score_query, summarize
from ..utils import records_hash


@torch.no_grad()
def extract_features(model, records, config, device):
    model.eval()
    loader = DataLoader(U1652DatasetEval(records, get_transforms(config)),
                        batch_size=config["eval_batch_size"], num_workers=config["workers"],
                        shuffle=False, pin_memory=device.type == "cuda")
    features = []
    for images in tqdm(loader, desc="Extract features"):
        context = torch.autocast("cuda", dtype=torch.float16) if config["amp"] and device.type == "cuda" else nullcontext()
        with context:
            embedding = model(images.to(device))
        features.append(F.normalize(embedding.float(), dim=-1).cpu())
    return torch.cat(features)


@torch.no_grad()
def evaluate(model, config, device):
    queries, raw_gallery = build_records(config)
    query_ids = {r["id"] for r in queries}
    gallery = [r for r in raw_gallery if r["id"] != "-1" and
               (config["gallery_scope"] == "all" or r["id"] in query_ids)]
    if not gallery:
        raise ValueError("有效图库为空")
    print(f"Query: {len(queries)}；gallery 原始/有效: {len(raw_gallery)}/{len(gallery)}")
    q_features = extract_features(model, queries, config, device)
    g_features = extract_features(model, gallery, config, device).to(device)
    gallery_ids = np.array([r["id"] for r in gallery])
    results = []
    # 只保留一个查询块的分数，避免持有完整 Q×G 矩阵。
    for start in tqdm(range(0, len(queries), 128), desc="Rank queries"):
        scores = (q_features[start:start + 128].to(device) @ g_features.T).cpu().numpy()
        for query, row in zip(queries[start:start + 128], scores):
            detail = score_query(row, gallery_ids, query["id"], config["top_k"])

            def candidate(index):
                if index is None:
                    return None
                return {**gallery[index], "score": float(row[index]), "correct": gallery[index]["id"] == query["id"]}

            detail["top_k"] = [candidate(i) for i in detail.pop("top_indices")]
            detail["best_positive"] = candidate(detail.pop("best_positive_index"))
            detail["best_negative"] = candidate(detail.pop("best_negative_index"))
            results.append({**query, **detail})
    metrics = summarize(results, len(gallery))
    metrics.update({"query_count": len(queries), "query_places": len(query_ids),
                    "gallery_raw_count": len(raw_gallery), "gallery_count": len(gallery),
                    "gallery_raw_places": len({r["id"] for r in raw_gallery}),
                    "gallery_places": len(set(gallery_ids)),
                    "query_manifest_sha256": records_hash(queries),
                    "gallery_manifest_sha256": records_hash(gallery),
                    "protocol": {"version": "cvgl_new_v1", "gallery_scope": config["gallery_scope"],
                                 "similarity": "cosine_fp32", "normalized_features": True,
                                 "ap": "s4g_trapezoid", "recall_1_percent_k": max(1, int(np.ceil(len(gallery) / 100))),
                                 "missing_positive": "excluded_and_recorded",
                                 "ties": "stable_path_order", "inference_amp": bool(config["amp"] and device.type == "cuda")}})
    included = {r["path"] for r in gallery}
    manifest = [{**r, "included": r["path"] in included} for r in raw_gallery]
    return metrics, {"gallery": manifest, "queries": results}
