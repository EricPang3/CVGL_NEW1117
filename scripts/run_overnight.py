"""服务器备用队列：原矩阵完成后，按固定顺序增加成对seed验证。"""

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

try:
    from scripts.experiment_budget import budget_expired, parse_deadline
    from scripts.module_suite import now, write_json
except ModuleNotFoundError:
    from experiment_budget import budget_expired, parse_deadline
    from module_suite import now, write_json

ROOT = Path(__file__).resolve().parents[1]


def repeat_config(base, seed, deadline):
    return {**base, "run": f'{base["run"]}_seed{seed}',
            "baseline_run": f'{base["run"]}_seed{seed}_baseline', "seed": seed,
            "bootstrap_baseline": True, "stop_starting_at": deadline.isoformat()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--until", required=True)
    parser.add_argument("--name", default="overnight20261120")
    parser.add_argument("--config", default=str(ROOT / "configs/module_suite.yaml"))
    parser.add_argument("--data-root", default=str(ROOT / "data"))
    args = parser.parse_args()
    deadline = parse_deadline(args.until)
    if Path(args.name).name != args.name:
        parser.error("name必须为单个目录名")
    directory = ROOT / "outputs" / args.name
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / "worker.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    base = yaml.safe_load(Path(args.config).read_text())
    state_file = directory / "state.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else {
        "started_at": now(), "runs": [base["run"]], "seed_order": [], "primary_run": base["run"]}
    if state.get("until", deadline.isoformat()) != deadline.isoformat():
        raise RuntimeError("续跑预算变化，请另建计划名")
    state.update(until=deadline.isoformat(), status="waiting_primary", active_run=base["run"])

    def save():
        state["updated_at"] = now()
        write_json(state_file, state)

    def failed(run, details):
        state.update(status="needs_attention", active_run=run, error=details)
        save()
        print("needs_attention", run, details, flush=True)

    save()
    while True:
        path = ROOT / "outputs" / base["run"] / "state.json"
        primary = json.loads(path.read_text()) if path.exists() else {}
        if primary.get("status") == "complete":
            break
        if primary.get("status") in ("partial_failure", "preflight_failed", "baseline_failed"):
            failed(base["run"], primary.get("failures", primary.get("error")))
            return
        state["primary_progress"] = {"status": primary.get("status"), "active": primary.get("active"),
                                     "evaluations": len(primary.get("results", [])), "updated_at": primary.get("updated_at")}
        save()
        time.sleep(30)
    seed = base["seed"] + 1
    while not budget_expired(deadline):
        config = repeat_config(base, seed, deadline)
        run = config["run"]
        path = ROOT / "outputs" / run / "state.json"
        existing = json.loads(path.read_text()) if path.exists() else {}
        if existing.get("status") == "complete":
            seed += 1
            continue
        if existing.get("status") == "budget_reached":
            break
        config_file = directory / f"seed{seed}.yaml"
        text = yaml.safe_dump(config, allow_unicode=True, sort_keys=False)
        if config_file.exists() and config_file.read_text() != text:
            raise RuntimeError("已注册补充实验配置变化")
        config_file.write_text(text)
        if run not in state["runs"]:
            state["runs"].append(run)
            state["seed_order"].append(seed)
        state.update(status="running_repeats", active_run=run)
        save()
        with (ROOT / "outputs" / (run + ".launch.log")).open("a") as log:
            result = subprocess.run([sys.executable, "-u", "scripts/module_suite.py", "--config", str(config_file),
                                     "--data-root", args.data_root], cwd=ROOT,
                                    env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
                                    stdout=log, stderr=subprocess.STDOUT)
        finished = json.loads(path.read_text()) if path.exists() else {}
        if result.returncode or finished.get("status") not in ("complete", "budget_reached"):
            failed(run, finished.get("failures", finished.get("error", f"exit={result.returncode}")))
            return
        if finished["status"] == "budget_reached":
            break
        seed += 1
    state.update(status="finished", active_run=None, finished_at=now())
    save()


if __name__ == "__main__":
    main()
