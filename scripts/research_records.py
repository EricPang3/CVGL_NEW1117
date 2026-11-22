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
