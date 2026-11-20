"""按预注册配方逐项跑33模块；复用训练/评估/报告，逐任务留痕并支持续跑。"""

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now().astimezone().isoformat()


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def file_hash(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def prepare_archive(directory, model_dir, checkpoint, job, archive_checkpoint, retain_remote=False):
    bundle = directory / f"{job}.tar.gz"
    with tarfile.open(bundle.with_suffix(".tmp"), "w:gz", compresslevel=1) as archive:
        for path in sorted(model_dir.rglob("*")):
            relative = path.relative_to(model_dir)
            if path.is_file() and path.name != "model.pth" and "report" not in relative.parts:
                archive.add(path, arcname=relative.as_posix())
    bundle.with_suffix(".tmp").replace(bundle)
    metrics = json.loads((model_dir / "D2S/metrics.json").read_text())
    ready = {"job": job, "bundle": str(bundle), "bundle_sha256": file_hash(bundle), "model_dir": str(model_dir),
             "checkpoint": str(checkpoint), "archive_checkpoint": archive_checkpoint, "retain_remote": retain_remote,
             "checkpoint_sha256": metrics["checkpoint_sha256"], "completed_at": now()}
    write_json(directory / "archive_ready" / f"{job}.json", ready)
    return bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/module_suite.yaml"))
    parser.add_argument("--data-root", default=str(ROOT / "data"))
    args = parser.parse_args()
    config_bytes = Path(args.config).read_bytes()
    config = yaml.safe_load(config_bytes)
    directory = ROOT / "outputs" / config["run"]
    directory.mkdir(parents=True, exist_ok=True)
    archive_ready = directory / "archive_ready"
    archive_ready.mkdir(exist_ok=True)
    state_file = directory / "state.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else {
        "started_at": now(), "jobs": {}, "results": [], "events": [], "status": "running"}
    digest = hashlib.sha256(config_bytes).hexdigest()
    if state.get("config_sha256", digest) != digest:
        raise RuntimeError("续跑配置与原预注册文件不同；请另建运行名")
    state["config_sha256"] = digest
    state["expected_archive_tasks"] = (len(config["modules"]) + 1) * len(config["tasks"])
    shutil.copy2(args.config, directory / "protocol.yaml")
    environment = {**os.environ, "CVGL_DATA_ROOT": str(Path(args.data_root).resolve()),
                   "HF_HUB_OFFLINE": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}

    def save():
        state["updated_at"] = now()
        write_json(state_file, state)
        if state["results"]:
            with (directory / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(state["results"][0]))
                writer.writeheader()
                writer.writerows(state["results"])

    def run(job, phase, command):
        record = state["jobs"][job]
        if phase in record.get("completed", []):
            return
        record.update(status="running", phase=phase)
        state["active"] = f"{job}/{phase}"
        save()
        event = {"at": now(), "job": job, "phase": phase, "command": command}
        with (directory / "commands.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        with (directory / f"{job}.log").open("a", encoding="utf-8") as stream:
            print(now(), job, phase, flush=True)
            subprocess.run([sys.executable, *command], cwd=ROOT, env=environment,
                           stdout=stream, stderr=subprocess.STDOUT, check=True)
        record.setdefault("completed", []).append(phase)
        save()

    baseline_state = ROOT / "outputs" / config["baseline_run"] / "run.json"
    state["status"] = "waiting_baseline"
    save()
    while True:
        baseline = json.loads(baseline_state.read_text()) if baseline_state.exists() else {}
        if baseline.get("status") == "complete":
            break
        if baseline.get("status") == "failed":
            state["status"] = "baseline_failed"
            save()
            raise RuntimeError("基线失败；修复基线后重新启动同一配方即可续跑")
        time.sleep(30)
    state["status"] = "running"
    save()
    preflight = directory / "gpu_preflight.json"
    if not preflight.exists() or not json.loads(preflight.read_text()).get("passed"):
        state["active"] = "gpu_preflight"
        save()
        try:
            subprocess.run([sys.executable, "scripts/check_modules.py", "--device", "cuda:0", "--output", str(preflight)],
                           cwd=ROOT, env=environment, check=True)
        except Exception as error:
            state.update(status="preflight_failed", error=str(error))
            save()
            raise
    for task in config["tasks"]:
        label = "u1652" if task["dataset"] == "U1652" else f'sues200_{task["altitude"]}'
        job = f"baseline_none_{label}"
        base = ROOT / "outputs" / f'{config["baseline_run"]}_{label}'
        if not (archive_ready / f"{job}.json").exists():
            prepare_archive(directory, base, base / "model.pth", job, True, retain_remote=True)
    for module in config["modules"]:
        for task in config["tasks"]:
            label = "u1652" if task["dataset"] == "U1652" else f'sues200_{task["altitude"]}'
            job = f'{module["id"]}_{label}'
            record = state["jobs"].setdefault(job, {"module": module["id"], **task, "completed": []})
            if record.get("status") == "complete":
                continue
            # 接收端成功归档后会释放模型副本；空间不足时暂停，绝不删除未校验产物。
            while shutil.disk_usage(ROOT / "outputs").free < 3 * 2**30:
                state["status"] = "waiting_archive_space"
                save()
                time.sleep(30)
            state["status"] = "running"
            base = ROOT / "outputs" / f'{config["baseline_run"]}_{label}'
            model_dir = ROOT / "outputs" / f'{config["run"]}_{job}'
            checkpoint = model_dir / "model.pth" if module["kind"] == "train" else base / "model.pth"
            try:
                if module["kind"] == "train":
                    if "train" not in record["completed"] and model_dir.exists():
                        attempt = 1
                        while model_dir.with_name(model_dir.name + f"_failed_attempt{attempt}").exists():
                            attempt += 1
                        model_dir.rename(model_dir.with_name(model_dir.name + f"_failed_attempt{attempt}"))
                    command = ["train.py", "--dataset", task["dataset"], "--name", model_dir.name,
                               "--module", module["id"], "--module-alpha", str(module["alpha"]),
                               "--epochs", str(config["epochs"]), "--batch-size", str(config["batch_size"]),
                               "--seed", str(config["seed"])]
                    if task["altitude"]:
                        command += ["--altitude", str(task["altitude"])]
                    run(job, "train", command)
                elif module["kind"] == "rerank":
                    run(job, "fit_reranker", ["scripts/rerank_experiment.py", "fit", "--method", module["id"],
                        "--checkpoint", str(checkpoint), "--output", str(model_dir), "--seed", str(config["seed"])])
                for direction in config["directions"]:
                    output = model_dir / direction
                    if module["kind"] == "rerank":
                        command = ["scripts/rerank_experiment.py", "evaluate", "--method", module["id"],
                                   "--checkpoint", str(checkpoint), "--head", str(model_dir / "reranker.pth"),
                                   "--direction", direction, "--output", str(output), "--candidates", str(config["rerank_candidates"])]
                    else:
                        command = ["evaluate.py", "--checkpoint", str(checkpoint), "--direction", direction,
                                   "--output", str(output), "--gallery-scope", config["gallery_scope"], "--top-k", str(config["top_k"]), "--overwrite"]
                        if module["kind"] == "frozen":
                            command += ["--module", module["id"], "--module-alpha", str(module["alpha"])]
                    run(job, f"evaluate_{direction}", command)
                    report_command = ["report.py", "--experiment", str(output),
                        "--limit", str(config["report_limit"]), "--compare", str(base / direction)]
                    if module["kind"] == "rerank":
                        report_command += ["--allow-reranking-comparison"]
                    run(job, f"report_{direction}", report_command)
                    metrics = json.loads((output / "metrics.json").read_text())
                    original = json.loads((base / direction / "metrics.json").read_text())
                    comparison = json.loads((output / "report/comparison.json").read_text())
                    row = {"module": module["id"], "kind": module["kind"], "dataset": task["dataset"],
                           "altitude": task["altitude"], "direction": direction,
                           **{k: metrics[k] * 100 for k in ("recall@1", "recall@5", "recall@10", "mAP")},
                           "delta_R1_pp": (metrics["recall@1"] - original["recall@1"]) * 100,
                           "delta_AP_pp": (metrics["mAP"] - original["mAP"]) * 100,
                           "fixed_queries": comparison["counts"]["fixed"], "regressed_queries": comparison["counts"]["regressed"],
                           "descriptor_dim": metrics.get("descriptor_dim", 1024),
                           "evaluation_seconds": metrics.get("evaluation_seconds"),
                           "extra_parameters": metrics.get("extra_parameters"),
                           "checkpoint_sha256": metrics.get("checkpoint_sha256"),
                           "output": str(output)}
                    state["results"] = [r for r in state["results"] if (r["module"],r["dataset"],r["altitude"],r["direction"]) != (row["module"],row["dataset"],row["altitude"],row["direction"])] + [row]
                    save()
                bundle = prepare_archive(directory, model_dir, checkpoint, job, module["kind"] == "train")
                record.update(status="complete", completed_at=now(), artifact=str(bundle))
            except Exception as error:
                record.update(status="failed", error=str(error), failed_at=now())
                state["events"].append({"at": now(), "job": job, "error": str(error)})
                print("FAILED", job, error, flush=True)
            save()
    failures = [name for name, record in state["jobs"].items() if record.get("status") != "complete"]
    state.update(status="complete" if not failures else "partial_failure", failures=failures, finished_at=now(), active=None)
    save()
    print(state["status"], "results", len(state["results"]), "failed jobs", failures, flush=True)


if __name__ == "__main__":
    main()
