import math

import numpy as np


def score_query(scores, gallery_ids, query_id, top_k, order=None):
    scores = np.asarray(scores)
    order = np.argsort(-scores, kind="stable") if order is None else np.asarray(order)
    correct = np.asarray(gallery_ids)[order] == query_id
    matches, negatives = np.flatnonzero(correct), np.flatnonzero(~correct)
    positive = int(order[matches[0]]) if len(matches) else None
    negative = int(order[negatives[0]]) if len(negatives) else None
    count = len(matches)
    ap = None
    if count:
        # 原项目 compute_mAP 的梯形积分，保留多正样本行为。
        ap = sum(((i + 1) / (rank + 1) + (i / rank if rank else 1.0)) / (2 * count)
                 for i, rank in enumerate(matches))
    return {"valid": bool(count), "positive_count": count, "ap": ap,
            "first_positive_rank": int(matches[0]) + 1 if count else None,
            "top1_correct": bool(matches[0] == 0) if count else None,
            "top_indices": [int(i) for i in order[:top_k]],
            "best_positive_index": positive, "best_negative_index": negative,
            "score_margin": float(scores[positive] - scores[negative]) if positive is not None and negative is not None else None}


def summarize(queries, gallery_size):
    valid = [q for q in queries if q["valid"]]
    if not valid:
        raise ValueError("所有查询在图库中均无正确匹配；请检查数据集、方向和划分")
    metrics = {"valid_queries": len(valid), "missing_positive_queries": len(queries) - len(valid),
               "mAP": float(sum(q["ap"] for q in valid) / len(valid))}
    for label, k in [("1", 1), ("5", 5), ("10", 10), ("1%", max(1, math.ceil(gallery_size / 100)))]:
        metrics[f"recall@{label}"] = sum(q["first_positive_rank"] <= k for q in valid) / len(valid)
    return metrics
