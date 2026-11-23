"""导出单轮研究记录；开发集/官方测试、图库和seed始终分列，不与旧33模块结果混表。"""

import argparse
import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def export_research(state, run, root=ROOT):
    directory = Path(root) / "results" / run
    write_csv(directory / "metrics.csv", state.get("results", []))
    write_csv(directory / "diagnostics.csv", state.get("diagnostic_summary", []))
    selection = []
    for phase, key in (("screen", "screen_selection"), ("confirmation", "confirmation_selection")):
        for row in state.get(key, []):
            selection.append({"phase": phase, "variant": row["variant"], "complete": row["complete"],
                "eligible": row["eligible"], "mean_delta_AP_pp": row["mean_delta_AP_pp"],
                "dataset_delta_R1_pp": json.dumps(row["dataset_delta_R1_pp"], sort_keys=True),
                "phase_delta_AP_pp": json.dumps(row["phase_delta_AP_pp"], sort_keys=True)})
    write_csv(directory / "selection.csv", selection)
    groups = {}
    for row in state.get("results", []):
        if row["phase"] != "final" or row["split"] != "test":
            continue
        key = tuple(row[k] for k in ("variant", "dataset", "altitude", "gallery_scope", "direction"))
        groups.setdefault(key, []).append(row)
    paired = []
    for key, rows in sorted(groups.items(), key=lambda item: str(item[0])):
        paired.append({**dict(zip(("variant", "dataset", "altitude", "gallery_scope", "direction"), key)),
            "seeds": ",".join(str(r["seed"]) for r in sorted(rows, key=lambda r: r["seed"])), "n": len(rows),
            "mean_R1_percent": statistics.mean(r["recall@1"] for r in rows),
            "std_R1_percent": statistics.stdev(r["recall@1"] for r in rows) if len(rows) > 1 else None,
            "mean_AP_percent": statistics.mean(r["mAP"] for r in rows),
            "std_AP_percent": statistics.stdev(r["mAP"] for r in rows) if len(rows) > 1 else None,
            "mean_paired_delta_R1_pp": statistics.mean(r["delta_R1_pp"] for r in rows),
            "mean_paired_delta_AP_pp": statistics.mean(r["delta_AP_pp"] for r in rows)})
    write_csv(directory / "paired_seeds.csv", paired)
    if state.get("selection_mode") == "fixed_ablation":
        export_ablation(state, directory)


def export_ablation(state, directory):
    contrasts = [("wtconv", "baseline"), ("dwconv_matched", "baseline"), ("consensus", "baseline"),
                 ("visible_consensus", "consensus"), ("wt_consensus", "consensus"),
                 ("dw_consensus", "consensus"), ("wt_consensus", "dw_consensus"),
                 ("wt_consensus", "wtconv"), ("dw_consensus", "dwconv_matched")]
    keys = ("seed", "dataset", "altitude", "gallery_scope", "direction")
    lookup = {(r["variant"], *[r[k] for k in keys]): r for r in state.get("results", [])
              if r["phase"] == "final" and r["split"] == "test" and
              state.get("jobs", {}).get(r["artifact_job"], {}).get("status") == "complete"}
    effects = []
    for candidate, control in contrasts:
        for key, row in lookup.items():
            if key[0] != candidate:
                continue
            base = lookup.get((control, *key[1:]))
            if base is None:
                continue
            if any(row.get(k) != base.get(k) for k in ("epochs", "batch_size", "img_size", "schedule_epochs")):
                raise ValueError("消融配方不一致，不能计算配对差值")
            effects.append({"candidate": candidate, "control": control, **{k: row[k] for k in keys},
                "delta_R1_pp": row["recall@1"] - base["recall@1"], "delta_AP_pp": row["mAP"] - base["mAP"],
                "candidate_sha256": row["checkpoint_sha256"], "control_sha256": base["checkpoint_sha256"]})
    write_csv(directory / "ablation_effects.csv", effects)
    groups = {}
    for row in effects:
        key = tuple(row[k] for k in ("candidate", "control", *keys[1:]))
        groups.setdefault(key, []).append(row)
    summary = []
    for key, rows in sorted(groups.items(), key=lambda item: str(item[0])):
        summary.append({**dict(zip(("candidate", "control", *keys[1:]), key)), "n": len(rows),
            "seeds": ",".join(str(r["seed"]) for r in sorted(rows, key=lambda r: r["seed"])),
            "complete": {r["seed"] for r in rows} == set(state["final_seeds"]),
            "mean_delta_R1_pp": statistics.mean(r["delta_R1_pp"] for r in rows),
            "std_delta_R1_pp": statistics.stdev(r["delta_R1_pp"] for r in rows) if len(rows) > 1 else None,
            "mean_delta_AP_pp": statistics.mean(r["delta_AP_pp"] for r in rows),
            "positive_R1_seeds": sum(r["delta_R1_pp"] > 0 for r in rows),
            "negative_R1_seeds": sum(r["delta_R1_pp"] < 0 for r in rows)})
    write_csv(directory / "ablation_summary.csv", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="research20261122")
    args = parser.parse_args()
    if Path(args.run).name != args.run:
        parser.error("非法运行名")
    state = json.loads((ROOT / "outputs" / args.run / "state.json").read_text())
    export_research(state, args.run)


if __name__ == "__main__":
    main()
