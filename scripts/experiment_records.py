"""保存实验数值表；按明确策略清理已完成且保留完整结果的模型权重。"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = [("U1652", None, d) for d in ("D2S", "S2D")] + [("SUES200", a, d) for a in (150, 200, 250, 300) for d in ("D2S", "S2D")]


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".records.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".records.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_runs(root):
    plan = root / "outputs/overnight20261120/state.json"
    names = json.loads(plan.read_text()).get("runs", []) if plan.exists() else ["modules20261120"]
    return [(name, json.loads((root / "outputs" / name / "state.json").read_text())) for name in names
            if (root / "outputs" / name / "state.json").exists()]


def task_key(row):
    return row["dataset"], row.get("altitude"), row["direction"]


def export_tables(root=ROOT):
    destination = root / "results"
    all_rows = []
    runs = read_runs(root)
    catalog = {r["module"] for _, s in runs for r in s.get("results", [])}
    for run, state in runs:
        rows = [{**r, "run": run, "seed": state.get("seed", 1 if run == "modules20261120" else 2),
                 "matrix_status": state["status"], "matrix_updated_at": state.get("updated_at", ""),
                 "baseline_R1": r["recall@1"] - r["delta_R1_pp"], "baseline_mAP": r["mAP"] - r["delta_AP_pp"]}
                for r in state.get("results", [])]
        baselines = {}
        for r in rows:
            key = task_key(r)
            label = "u1652" if r["dataset"] == "U1652" else f'sues200_{r["altitude"]}'
            metric = root / "outputs" / run / "artifacts" / f'{r["module"]}_{label}' / r["direction"] / "metrics.json"
            fallback = root / "outputs" / run / "artifacts" / f"baseline_none_{label}" / r["direction"] / "metrics.json"
            metadata = json.loads((metric if metric.exists() else fallback).read_text()) if metric.exists() or fallback.exists() else {}
            r.update({"gallery_scope": metadata.get("protocol", {}).get("gallery_scope", ""),
                      "AP_rule": metadata.get("protocol", {}).get("ap", ""),
                      **{k: metadata.get(k, "") for k in ("query_count", "gallery_count", "query_manifest_sha256", "gallery_manifest_sha256")}})
            if key in baselines:
                continue
            label = "u1652" if r["dataset"] == "U1652" else f'sues200_{r["altitude"]}'
            metrics = root / "outputs" / run / "artifacts" / f"baseline_none_{label}" / r["direction"] / "metrics.json"
            m = json.loads(metrics.read_text()) if metrics.exists() else {}
            baselines[key] = {**{k: r[k] for k in ("run", "seed", "matrix_status", "matrix_updated_at", "dataset", "altitude", "direction")},
                              "module": "baseline", "kind": "baseline", "recall@1": m.get("recall@1", r["baseline_R1"] / 100) * 100,
                              "mAP": m.get("mAP", r["baseline_mAP"] / 100) * 100, "delta_R1_pp": 0, "delta_AP_pp": 0,
                              "checkpoint_sha256": m.get("checkpoint_sha256", ""), "baseline_value_source": "metrics.json" if m else "recorded_score_minus_recorded_delta"}
            baselines[key].update({k: m[k] * 100 for k in ("recall@5", "recall@10") if k in m})
            baselines[key].update({k: r.get(k, "") for k in ("gallery_scope", "AP_rule", "query_count", "gallery_count", "query_manifest_sha256", "gallery_manifest_sha256")})
        rows = list(baselines.values()) + rows
        all_rows.extend(rows)
        columns = [f'{d}_{a or "NA"}_{direction}' for d, a, direction in TASKS]
        for metric, suffix in (("recall@1", "R1"), ("mAP", "mAP")):
            wide = []
            for module in sorted(catalog | {"baseline"}, key=lambda v: (v != "baseline", v)):
                values = {task_key(r): r[metric] for r in rows if r["module"] == module}
                wide.append({"module": module, "completed_evaluations": len(values),
                             **{c: round(values[t], 6) if t in values else "" for c, t in zip(columns, TASKS)}})
            write_csv(destination / f'{run}_{suffix}.csv', wide, ["module", "completed_evaluations", *columns])
    fields = ["run", "seed", "matrix_status", "matrix_updated_at", "module", "kind", "dataset", "altitude", "direction",
              "recall@1", "recall@5", "recall@10", "mAP", "baseline_R1", "baseline_mAP", "delta_R1_pp", "delta_AP_pp",
              "fixed_queries", "regressed_queries", "descriptor_dim", "extra_parameters", "evaluation_seconds", "checkpoint_sha256", "baseline_value_source",
              "gallery_scope", "AP_rule", "query_count", "gallery_count", "query_manifest_sha256", "gallery_manifest_sha256", "output"]
    write_csv(destination / "all_metrics.csv", all_rows, fields)
    summaries = []
    for run in dict.fromkeys(r["run"] for r in all_rows):
        for module in dict.fromkeys(r["module"] for r in all_rows if r["run"] == run):
            rr = [r for r in all_rows if r["run"] == run and r["module"] == module]
            summaries.append({"run": run, "seed": rr[0]["seed"], "module": module, "completed_evaluations": len(rr),
                              "R1_improved": sum(r["delta_R1_pp"] > 1e-7 for r in rr),
                              "R1_regressed": sum(r["delta_R1_pp"] < -1e-7 for r in rr),
                              "R1_tied": sum(abs(r["delta_R1_pp"]) <= 1e-7 for r in rr)})
    write_csv(destination / "module_summary.csv", summaries, ["run", "seed", "module", "completed_evaluations", "R1_improved", "R1_regressed", "R1_tied"])
    first = {(r["module"], *task_key(r)): r for r in all_rows if r["seed"] == 1}
    pairs = []
    for r in all_rows:
        key = (r["module"], *task_key(r))
        if r["seed"] == 1 or key not in first:
            continue
        old = first[key]
        pairs.append({"module": r["module"], "dataset": r["dataset"], "altitude": r["altitude"], "direction": r["direction"], "repeat_seed": r["seed"],
                      "seed1_R1": old["recall@1"], "repeat_R1": r["recall@1"], "seed1_delta_R1_pp": old["delta_R1_pp"], "repeat_delta_R1_pp": r["delta_R1_pp"],
                      "seed1_mAP": old["mAP"], "repeat_mAP": r["mAP"], "seed1_delta_AP_pp": old["delta_AP_pp"], "repeat_delta_AP_pp": r["delta_AP_pp"]})
    write_csv(destination / "replication.csv", pairs, ["module", "dataset", "altitude", "direction", "repeat_seed", "seed1_R1", "repeat_R1", "seed1_delta_R1_pp", "repeat_delta_R1_pp", "seed1_mAP", "repeat_mAP", "seed1_delta_AP_pp", "repeat_delta_AP_pp"])
    policy = load_policy(root)
    if policy:
        write_csv(destination / "checkpoint_retention.csv", [{"module": "baseline", "dataset": "all", "altitude": "all", "reason": "all seeds retained; runtime dependency"}, *policy["keep"]], ["module", "dataset", "altitude", "reason"])
    inventory = []
    for run, state in runs:
        path = root / "outputs" / run / "archive_manifest.json"
        if not path.exists():
            continue
        manifest = json.loads(path.read_text())
        entries = [(j, r, "current") for j, r in manifest["artifacts"].items()]
        entries += [(j, r, "superseded") for j, history in manifest.get("artifact_history", {}).items() for r in history]
        for job, ready, revision in entries:
            weight = Path(ready["local_directory"]) / "model.pth"
            record = state.get("jobs", {}).get(job, {})
            storage = "shared_baseline" if not ready.get("archive_checkpoint") else "retained" if weight.exists() else "absent"
            inventory.append({"run": run, "job": job, "revision": revision, "module": record.get("module", "baseline"),
                              "checkpoint_storage": ready.get("checkpoint_storage", storage), "local_weight_exists": weight.exists(),
                              "checkpoint_sha256": ready.get("checkpoint_sha256", ""), "local_directory": ready["local_directory"],
                              "policy_sha256": ready.get("checkpoint_retention_policy_sha256", "")})
    write_csv(destination / "checkpoint_inventory.csv", inventory, ["run", "job", "revision", "module", "checkpoint_storage", "local_weight_exists", "checkpoint_sha256", "local_directory", "policy_sha256"])
    return all_rows


def load_policy(root=ROOT):
    path = root / "configs/checkpoint_retention.json"
    return json.loads(path.read_text()) if path.exists() else None


def keep_checkpoint(policy, record, ready):
    if policy is None or ready.get("retain_remote") or ready["job"].startswith("baseline_none_"):
        return True
    if not record or record.get("module") not in policy["eligible_modules"]:
        return True
    return any((record["module"], record["dataset"], record.get("altitude")) == (p["module"], p["dataset"], p.get("altitude")) for p in policy["keep"])


def prune_local(root=ROOT):
    export_tables(root)
    policy = load_policy(root)
    if policy is None:
        raise RuntimeError("缺少权重保留策略")
    log = root / "results/weight_cleanup.csv"
    events = []
    if log.exists():
        with log.open(encoding="utf-8-sig") as stream:
            events = list(csv.DictReader(stream))
    fields = ["time", "status", "run", "job", "relative_path", "bytes", "sha256", "reason"]
    freed = count = 0
    for run, state in read_runs(root):
        directory = root / "outputs" / run
        manifest_file = directory / "archive_manifest.json"
        if not manifest_file.exists():
            continue
        manifest = json.loads(manifest_file.read_text())
        entries = [(job, ready, False) for job, ready in manifest["artifacts"].items()]
        entries += [(job, ready, True) for job, history in manifest.get("artifact_history", {}).items() for ready in history]
        for job, ready, historical in entries:
            record = state.get("jobs", {}).get(job)
            if not historical and record and record.get("status") != "complete":
                continue
            if not historical and keep_checkpoint(policy, record, ready):
                continue
            model = Path(ready["local_directory"]).resolve()
            weight = model / "model.pth"
            if not weight.exists():
                continue
            if not model.is_relative_to((root / "outputs").resolve()) or weight.is_symlink():
                raise RuntimeError("权重清理路径不在outputs内")
            required = [model / name for name in ("config.yaml", "train.csv", "source.zip")]
            required += [model / d / name for d in ("D2S", "S2D") for name in ("metrics.json", "queries.json", "report.zip")]
            if not all(p.is_file() for p in required):
                raise RuntimeError(f"结果记录不完整，保留权重：{job}")
            event = {"time": datetime.now().astimezone().isoformat(), "status": "planned", "run": run, "job": job,
                     "relative_path": str(weight.relative_to(root)), "bytes": weight.stat().st_size, "sha256": ready["checkpoint_sha256"],
                     "reason": "superseded PAM revision" if historical else "completed experiment outside selected checkpoint set"}
            events.append(event)
            write_csv(log, events, fields)  # 删除前先落盘路径、SHA和理由。
            weight.unlink()
            event["status"] = "deleted"
            ready["checkpoint_storage"] = "discarded_by_policy"
            freed += event["bytes"]
            count += 1
            save_json(manifest_file, manifest)
            write_csv(log, events, fields)
    return {"deleted_weights": count, "freed_bytes": freed, "freed_GiB": freed / 2**30}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="先导出完整表格，再清理策略外的已完成权重")
    args = parser.parse_args()
    rows = export_tables()
    print("table_rows", len(rows), "directory", ROOT / "results")
    if args.apply:
        print(json.dumps(prune_local(), ensure_ascii=False))


if __name__ == "__main__":
    main()
