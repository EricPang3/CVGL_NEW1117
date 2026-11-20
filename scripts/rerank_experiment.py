"""训练地点拟合重排头，随后独立评估固定Top-20；不改变GAP初排模型。"""

import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from torch.nn import functional as F
from tqdm import tqdm

from evaluate import file_hash
from sample4geo.dataset import build_records
from sample4geo.evaluate.metrics import score_query, summarize
from sample4geo.evaluate.rerank import Reranker, extract, make_patch, make_r2former
from sample4geo.model import TimmModel
from sample4geo.trainer import cosine_scheduler
from sample4geo.utils import configure, load_weights, records_hash, save_config, save_json, setup, snapshot_source


def fit(args, model, config, device):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    baseline_hash = file_hash(args.checkpoint)
    if (output / "reranker.pth").exists():
        previous = torch.load(output / "reranker.pth", map_location="cpu", weights_only=True)
        if previous["baseline_sha256"] != baseline_hash or previous["method"] != args.method:
            raise RuntimeError("拟合目录属于另一份基线或方法")
        print("复用已完整保存的重排头")
        return
    metadata = {"method": args.method, "baseline_sha256": baseline_hash, "seed": args.seed,
                "candidate_budget": 20, "fit_scope": "official_training_places_only",
                "test_based_selection": False, **snapshot_source(output)}
    fitted = {"method": args.method, "baseline_sha256": baseline_hash, "metadata": metadata}
    started = time.perf_counter()
    if args.method != "superglobal_rerank":
        satellite, drone = build_records(config, training=True)
        generator = random.Random(args.seed)
        sample = generator.sample(satellite, min(256, len(satellite))) + generator.sample(drone, min(256, len(drone)))
        _, raw = extract(model, sample, config, device, raw_sample=True)
        pca = PCA(n_components=128, svd_solver="randomized", iterated_power=3, random_state=args.seed)
        reduced = pca.fit_transform(raw.numpy()).astype(np.float32)
        fitted["pca"] = {"mean": torch.from_numpy(pca.mean_.astype(np.float32)),
                         "components": torch.from_numpy(pca.components_.astype(np.float32))}
        metadata.update(pca_images=len(sample), pca_tokens=len(raw), pca_dimension=128,
                        pca_manifest=records_hash(sample), satellite_manifest=records_hash(satellite),
                        drone_manifest=records_hash(drone), training_ids=sorted({r["id"] for r in satellite}))
        del raw
        if args.method == "patch_netvlad":
            descriptors = F.normalize(torch.from_numpy(reduced), dim=1).numpy()
            km = MiniBatchKMeans(n_clusters=8, random_state=args.seed, n_init=3, batch_size=1024, max_iter=100)
            km.fit(descriptors)
            head = make_patch()
            head.init_params(km.cluster_centers_.astype(np.float32), descriptors)
            fitted["model"] = head.state_dict()
            metadata.update(clusters=8, local_dimension=128, patch_sizes=[2, 3, 4], strides=[2, 2, 2],
                            matcher="mutual_nearest_patch_cosine; multiscale average; 0.5 global + 0.5 local",
                            adaptation="ConvNeXt + train-only PCA/KMeans + multi-scale PatchNetVLAD; no VGG or RANSAC")
        else:
            sg, sl = extract(model, satellite, config, device, fitted["pca"], "r2former_rerank")
            dg, dl = extract(model, drone, config, device, fitted["pca"], "r2former_rerank")
            lookup = {r["id"]: i for i, r in enumerate(satellite)}
            positive = torch.tensor([lookup[r["id"]] for r in drone])
            scores = dg.to(device) @ sg.to(device).T
            scores[torch.arange(len(drone), device=device), positive.to(device)] = -torch.inf
            negative = scores.argmax(1).cpu()
            del scores
            head = make_r2former().to(device).train()
            optimizer = torch.optim.AdamW(head.parameters(), lr=0.001, weight_decay=0.01)
            count = len(drone) * 2
            steps = (count + 23) // 24
            scheduler = cosine_scheduler(optimizer, steps, {"epochs": 1, "warmup_epochs": 0.1})
            scaler = torch.cuda.amp.GradScaler(init_scale=1024, enabled=device.type == "cuda")
            order = torch.randperm(count, generator=torch.Generator().manual_seed(args.seed))
            total, skipped = 0.0, 0
            with (output / "head_train.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["step", "loss", "lr", "amp_scale"])
                writer.writeheader()
                for offset in tqdm(range(0, count, 24), desc="R2Former train-only binary pairs"):
                    indices = order[offset:offset + 24]
                    anchor = indices // 2
                    labels = (indices % 2 == 0).float().to(device)
                    target = torch.where(indices % 2 == 0, positive[anchor], negative[anchor])
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device.type, enabled=device.type == "cuda", dtype=torch.float16):
                        local, final = head(dg[anchor].to(device), dl[anchor].float().to(device),
                                            sg[target].to(device), sl[target].float().to(device))
                    loss = F.binary_cross_entropy(local.float(), labels) + F.binary_cross_entropy(final.float().clamp(1e-6, 1 - 1e-6), labels)
                    if not torch.isfinite(loss):
                        raise RuntimeError("R2Former拟合出现非有限loss")
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_value_(head.parameters(), 100)
                    old_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    skipped += int(scaler.get_scale() < old_scale)
                    rate = optimizer.param_groups[0]["lr"]
                    scheduler.step()
                    total += loss.item() * len(indices)
                    if offset % (24 * 100) == 0 or offset + 24 >= count:
                        writer.writerow({"step": offset // 24 + 1, "loss": loss.item(), "lr": rate, "amp_scale": scaler.get_scale()})
                        stream.flush()
            fitted["model"] = {name: value.detach().cpu() for name, value in head.state_dict().items()}
            metadata.update(head_epochs=1, head_pairs=count, head_batch_size=24, head_loss=total/count,
                            amp_skipped_steps=skipped, tokens=16, saliency="normalized ConvNeXt activation L2 norm",
                            decoder_dimension=128, decoder_depth=1, pair_stage_depth=2,
                            adaptation="author R2Former; CNN saliency/coordinates/PCA tokens; train-only positive/hard-negative BCE")
    metadata["fit_seconds"] = time.perf_counter() - started
    save_config(metadata, output / "fit_config.yaml")
    temporary = output / "reranker.tmp"
    torch.save(fitted, temporary)
    temporary.replace(output / "reranker.pth")
    print("拟合完成", output, metadata["fit_seconds"])


def evaluate_reranker(args, model, config, device):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    fitted = torch.load(args.head, map_location="cpu", weights_only=True)
    baseline_hash = file_hash(args.checkpoint)
    if fitted["method"] != args.method or fitted["baseline_sha256"] != baseline_hash:
        raise RuntimeError("重排头与方法/基线权重不匹配")
    started = time.perf_counter()
    queries, raw_gallery = build_records(config)
    ids = {r["id"] for r in queries}
    gallery = [r for r in raw_gallery if r["id"] != "-1" and (config["gallery_scope"] == "all" or r["id"] in ids)]
    qg, ql = extract(model, queries, config, device, fitted.get("pca"), args.method)
    gg, gl = extract(model, gallery, config, device, fitted.get("pca"), args.method)
    torch.cuda.synchronize(device) if device.type == "cuda" else None
    extracted = time.perf_counter()
    reranker = Reranker(args.method, fitted, qg, gg, ql, gl, device, candidates=args.candidates)
    gallery_ids = np.array([r["id"] for r in gallery])
    results, rerank_seconds = [], 0.0
    for start in tqdm(range(0, len(queries), 64), desc="Fixed-budget Top-K reranking"):
        scores = (qg[start:start + 64].to(device) @ gg.to(device).T).cpu().numpy()
        tick = time.perf_counter()
        orders, _ = reranker.rank(start, scores)
        torch.cuda.synchronize(device) if device.type == "cuda" else None
        rerank_seconds += time.perf_counter() - tick
        for query, row, order in zip(queries[start:start + 64], scores, orders):
            detail = score_query(row, gallery_ids, query["id"], config["top_k"], order=order)
            def candidate(index):
                return {**gallery[index], "score": float(row[index]), "correct": gallery[index]["id"] == query["id"]} if index is not None else None
            detail["top_k"] = [candidate(i) for i in detail.pop("top_indices")]
            detail["best_positive"] = candidate(detail.pop("best_positive_index"))
            detail["best_negative"] = candidate(detail.pop("best_negative_index"))
            results.append({**query, **detail})
    metrics = summarize(results, len(gallery))
    metrics.update(query_count=len(queries), gallery_raw_count=len(raw_gallery), gallery_count=len(gallery),
                   query_places=len(ids), gallery_places=len(set(gallery_ids)), descriptor_dim=1024,
                   query_manifest_sha256=records_hash(queries), gallery_manifest_sha256=records_hash(gallery),
                   checkpoint_sha256=baseline_hash, feature_extraction_seconds=extracted - started,
                   ranking_seconds=time.perf_counter() - extracted, evaluation_seconds=time.perf_counter() - started,
                   extra_parameters=sum(p.numel() for p in reranker.model.parameters()) if hasattr(reranker, "model") else 0,
                   reranking={"method": args.method, "candidates": reranker.candidates, "seconds": rerank_seconds,
                              "head_sha256": file_hash(args.head), "fit": fitted["metadata"], "candidate_score_field": "original GAP cosine; order after reranking"},
                   protocol={"version": "cvgl_new_v1", "gallery_scope": config["gallery_scope"],
                             "similarity": "cosine_fp32", "normalized_features": True, "ap": "s4g_trapezoid",
                             "recall_1_percent_k": max(1, int(np.ceil(len(gallery)/100))),
                             "missing_positive": "excluded_and_recorded", "ties": "stable_path_order",
                             "inference_amp": bool(config["amp"] and device.type == "cuda"), "ranking": args.method})
    config.update(checkpoint=str(Path(args.checkpoint).resolve()), checkpoint_sha256=baseline_hash,
                  reranker=str(Path(args.head).resolve()), rerank_method=args.method)
    included = {r["key"] for r in gallery}
    save_json(metrics, output / "metrics.json")
    save_json({"queries": results, "gallery": [{**r,"included":r["key"] in included} for r in raw_gallery], "evaluation":config}, output / "queries.json")
    save_config(config, output / "evaluation.yaml")
    save_config(fitted["metadata"], output / "reranking.yaml")
    for filename in ("fit_config.yaml", "head_train.csv", "source.zip"):
        source = Path(args.head).parent / filename
        if source.is_file():
            shutil.copy2(source, output / filename)
    _, original_config = load_weights(args.checkpoint)
    save_config(original_config, output / "training_config.yaml")
    train_csv = Path(args.checkpoint).parent / "train.csv"
    if train_csv.is_file():
        shutil.copy2(train_csv, output / "train.csv")
    print("R@1", metrics["recall@1"]*100,"AP",metrics["mAP"]*100,"rerank_seconds",rerank_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["fit", "evaluate"])
    parser.add_argument("--method", choices=["patch_netvlad", "r2former_rerank", "superglobal_rerank"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--head")
    parser.add_argument("--output", required=True)
    parser.add_argument("--direction", choices=["D2S", "S2D"])
    parser.add_argument("--data-root")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--candidates", type=int, default=20)
    parser.add_argument("--device")
    args = parser.parse_args()
    state, saved = load_weights(args.checkpoint)
    if saved.get("module", "none") != "none":
        parser.error("重排实验只能使用原GAP基线权重")
    config = configure(dict(saved), args)
    device = setup(config)
    model = TimmModel(pretrained=False).to(device).eval()
    model.load_state_dict(state, strict=True)
    del state
    if args.mode == "fit":
        fit(args, model, config, device)
    else:
        if not args.head or args.candidates < 1:
            parser.error("评估需要--head，candidates必须为正")
        evaluate_reranker(args, model, config, device)


if __name__ == "__main__":
    main()
