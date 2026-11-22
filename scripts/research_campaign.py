"""固定候选→训练地点开发集→配对复验→完整矩阵；有限预算、留痕、续跑与受保护清理。"""

import argparse
import csv
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def label(task):
    return "u1652" if task["dataset"] == "U1652" else f'sues200_{task["altitude"]}'


def job_name(phase, seed, arm, task):
    return f'{phase}_s{seed}_{arm["id"]}_{label(task)}'


def source_digest():
    paths = [*ROOT.glob("*.py"), *ROOT.glob("sample4geo/**/*.py"), *ROOT.glob("scripts/*.py"),
             *ROOT.glob("configs/*.yaml"), *ROOT.glob("configs/*.json"), *ROOT.glob("third_party/**/*.py")]
    value = hashlib.sha256()
    for path in sorted(paths):
        value.update(path.relative_to(ROOT).as_posix().encode() + b"\0" + path.read_bytes())
    return value.hexdigest()


def assess(results, arm_ids, phases, tasks, minimum_gain, maximum_drop):
    """只读开发集主图库；缺任务/seed不晋级，绝不参考official test。单位：百分点。"""
    lookup = {(r["phase"], r["variant"], r["dataset"], r["altitude"], r["direction"]): r
              for r in results if r["split"] == "development" and r["gallery_scope"] == "query_ids"}
    assessed = []
    for arm in arm_ids:
        gains, by_dataset, by_phase = [], {}, {}
        for phase in phases:
            for task in tasks:
                for direction in ("D2S", "S2D"):
                    key = (phase, arm, task["dataset"], task["altitude"], direction)
                    base = lookup.get((phase, "baseline", *key[2:]))
                    row = lookup.get(key)
                    if row is None or base is None:
                        continue
                    gain = row["mAP"] - base["mAP"]
                    gains.append(gain)
                    by_phase.setdefault(phase, []).append(gain)
                    by_dataset.setdefault(task["dataset"], []).append(row["recall@1"] - base["recall@1"])
        complete = len(gains) == len(phases) * len(tasks) * 2
        mean = sum(gains) / len(gains) if gains else None
        guard = complete and all(sum(v) / len(v) >= -maximum_drop for v in by_dataset.values())
        repeat_positive = len(phases) == 1 or all(sum(v) / len(v) > 0 for v in by_phase.values())
        assessed.append({"variant": arm, "complete": complete, "mean_delta_AP_pp": mean,
                         "eligible": bool(guard and repeat_positive and mean >= minimum_gain),
                         "dataset_delta_R1_pp": {k: sum(v) / len(v) for k, v in by_dataset.items()},
                         "phase_delta_AP_pp": {k: sum(v) / len(v) for k, v in by_phase.items()}})
    return sorted(assessed, key=lambda a: (not a["eligible"], -(a["mean_delta_AP_pp"] or 0), a["variant"]))


class BudgetReached(Exception):
    pass


class Campaign:
    def __init__(self, config_path, data_root, hours, retry_failed=False, repair_note=None):
        self.config_path = Path(config_path)
        self.config = yaml.safe_load(self.config_path.read_text())
        self.name = self.config["run"]
        names = [self.name, *[a["id"] for a in self.config["arms"]]]
        if any(not re.fullmatch(r"[a-zA-Z0-9_-]+", n) for n in names) or len(set(names[1:])) != len(names[1:]) or "baseline" in names[1:]:
            raise ValueError("运行名和候选ID必须安全且唯一，baseline为保留名")
        self.directory = ROOT / "outputs" / self.name
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "archive_ready").mkdir(exist_ok=True)
        state_file = self.directory / "state.json"
        config_hash = digest(self.config_path)
        code_hash = source_digest()
        self.state = json.loads(state_file.read_text()) if state_file.exists() else {
            "schema": "cvgl_research_v1", "started_at": now(), "jobs": {}, "results": [], "events": [],
            "config_sha256": config_hash, "code_sha256": code_hash, "hours": hours,
            "deadline": (datetime.now(timezone.utc) + timedelta(hours=hours)).isoformat()}
        if self.state["config_sha256"] != config_hash or self.state["hours"] != hours:
            raise ValueError("续跑配置/预算改变，请新建运行名；不会静默延长预算")
        if self.state["code_sha256"] != code_hash:
            if not repair_note:
                raise ValueError("源码改变；修复续跑需--repair-note说明原因，或使用新运行名")
            self.state["events"].append({"at": now(), "type": "code_repair", "old": self.state["code_sha256"], "new": code_hash, "note": repair_note})
            if (self.directory / "source.zip").exists():
                shutil.copy2(self.directory / "source.zip", self.directory / f'source_before_{self.state["code_sha256"][:12]}.zip')
            self.state["code_sha256"] = code_hash
            self.state.pop("preflight_passed", None)
        self.baseline = {"id": "baseline", "module": "none", "auxiliary": "none"}
        self.arms = {a["id"]: a for a in self.config["arms"]}
        self.environment = {**os.environ, "CVGL_DATA_ROOT": str(Path(data_root).expanduser().resolve()),
                            "HF_HUB_OFFLINE": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
        self.state.update(status="running", active=None)
        if retry_failed:
            for record in self.state["jobs"].values():
                if record.get("status") == "failed":
                    record["status"] = "retry_pending"
        shutil.copy2(self.config_path, self.directory / "protocol.yaml")
        from sample4geo.utils import snapshot_source
        snapshot_source(self.directory)
        self.save()

    def save(self):
        self.state["updated_at"] = now()
        self.state["expected_archive_tasks"] = sum(j.get("status") == "complete" for j in self.state["jobs"].values())
        atomic_json(self.directory / "state.json", self.state)
        if self.state["results"]:
            with (self.directory / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(self.state["results"][0]), lineterminator="\n")
                writer.writeheader()
                writer.writerows(self.state["results"])

    def boundary(self):
        if datetime.now(timezone.utc) >= datetime.fromisoformat(self.state["deadline"]):
            raise BudgetReached()
        if shutil.disk_usage(self.directory).free / 2**30 < self.config["minimum_free_gib"]:
            raise RuntimeError("空间低于安全阈值；已保存结果，停止队列，避免等待本地上线而长期空转")

    def command(self, job, phase, arguments):
        record = self.state["jobs"][job]
        if phase in record.get("completed", []):
            return
        self.state["active"] = f"{job}/{phase}"
        record.update(status="running", active_phase=phase)
        event = {"at": now(), "job": job, "phase": phase, "argv": [sys.executable, *map(str, arguments)]}
        with (self.directory / "commands.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        self.save()
        with (self.directory / f"{job}.log").open("a", encoding="utf-8") as stream:
            print(now(), job, phase, flush=True)
            subprocess.run(event["argv"], cwd=ROOT, env=self.environment, stdout=stream, stderr=subprocess.STDOUT, check=True)
        record.setdefault("completed", []).append(phase)
        self.save()

    def archive(self, job):
        record = self.state["jobs"][job]
        model_dir = Path(record["model_dir"])
        bundle = self.directory / f"{job}.tar.gz"
        temporary = bundle.with_suffix(".tmp")
        with tarfile.open(temporary, "w:gz", compresslevel=1) as archive:
            for path in sorted(model_dir.rglob("*")):
                relative = path.relative_to(model_dir)
                if path.is_file() and path.suffix != ".pth" and "report" not in relative.parts and path.name != "model.tmp":
                    archive.add(path, arcname=relative.as_posix())
        with tarfile.open(temporary) as archive:
            required = {"config.yaml", "source.zip", "train.csv", "D2S/metrics.json", "D2S/queries.json", "S2D/metrics.json", "S2D/queries.json"}
            if not required <= set(archive.getnames()):
                raise RuntimeError("缺少完整实验记录，拒绝归档与清理权重")
            for member in archive.getmembers():
                if member.isfile():
                    stream = archive.extractfile(member)
                    while stream.read(1048576):
                        pass
        temporary.replace(bundle)
        ready = {"job": job, "bundle": str(bundle), "bundle_sha256": digest(bundle),
                 "checkpoint": str(model_dir / "model.pth"), "model_dir": str(model_dir),
                 "checkpoint_sha256": record["checkpoint_sha256"], "archive_checkpoint": False, "retain_remote": True,
                 "checkpoint_storage": "retained_on_server" if (model_dir / "model.pth").is_file() else "discarded_on_server_after_records",
                 "phase": record["phase"], "seed": record["seed"], "variant": record["variant"]}
        atomic_json(self.directory / "archive_ready" / f"{job}.json", ready)

    def prune(self, job):
        record = self.state["jobs"][job]
        if record["variant"] == "baseline" or record.get("status") != "complete":
            return
        weight = Path(record["model_dir"]) / "model.pth"
        if not weight.exists():
            return
        ready_file = self.directory / "archive_ready" / f"{job}.json"
        if not ready_file.is_file():
            raise RuntimeError("未归档权重禁止清理")
        ready = json.loads(ready_file.read_text())
        if digest(ready["bundle"]) != ready["bundle_sha256"] or digest(weight) != record["checkpoint_sha256"]:
            raise RuntimeError("归档或权重指纹变化，拒绝清理")
        deletion = {"at": now(), "checkpoint_sha256": record["checkpoint_sha256"], "path": str(weight),
                    "verified_bundle_sha256": ready["bundle_sha256"], "bytes": weight.stat().st_size,
                    "reason": "预注册存储策略；完整逐图/训练/配置/源码已校验归档", "status": "authorized"}
        atomic_json(weight.parent / "WEIGHT_DELETED.json", deletion)
        weight.unlink()
        deletion["status"] = "deleted"
        atomic_json(weight.parent / "WEIGHT_DELETED.json", deletion)
        with (self.directory / "weight_cleanup.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(deletion, ensure_ascii=False) + "\n")
        self.archive(job)

    def train_and_evaluate(self, phase, seed, arm, task):
        job = job_name(phase, seed, arm, task)
        existing = self.state["jobs"].get(job, {})
        if existing.get("status") in ("complete", "failed"):
            return existing["status"] == "complete"
        self.boundary()
        output_name = f"{self.name}_{job}"
        model_dir = ROOT / "outputs" / output_name
        record = self.state["jobs"].setdefault(job, {"completed": [], "phase": phase, "seed": seed,
            "variant": arm["id"], "module": arm["module"], **task, "model_dir": str(model_dir)})
        development = phase != "final"
        split = "development" if development else "test"
        try:
            if "train" not in record["completed"] and model_dir.exists():
                interrupted = model_dir.with_name(model_dir.name + f"_interrupted_{time.time_ns()}")
                model_dir.rename(interrupted)
                self.state["events"].append({"at": now(), "job": job, "type": "interrupted_training_restart", "preserved": str(interrupted)})
            train_args = ["train.py", "--dataset", task["dataset"], "--name", output_name, "--seed", seed,
                "--epochs", self.config["screen_epochs"] if development else self.config["final_epochs"],
                "--batch-size", self.config["batch_size"], "--img-size", self.config["img_size"], "--workers", self.config["workers"],
                "--module", arm["module"], "--module-alpha", self.config["module_alpha"],
                "--auxiliary", arm.get("auxiliary", "none"), "--auxiliary-weight", arm.get("auxiliary_weight", 0.5),
                "--development-fraction", self.config["development_fraction"] if development else 0,
                "--development-seed", self.config["development_seed"]]
            if task["altitude"] is not None:
                train_args += ["--altitude", task["altitude"]]
            self.command(job, "train", train_args)
            checkpoint = model_dir / "model.pth"
            record["checkpoint_sha256"] = digest(checkpoint)
            scopes = ["query_ids"] if development or task["dataset"] == "U1652" else ["query_ids", "all"]
            for scope in scopes:
                for direction in ("D2S", "S2D"):
                    output = model_dir / direction if scope == "query_ids" else model_dir / "all" / direction
                    self.command(job, f"eval_{scope}_{direction}", ["evaluate.py", "--checkpoint", checkpoint, "--output", output,
                        "--direction", direction, "--split", split, "--gallery-scope", scope, "--batch-size", self.config["eval_batch_size"], "--overwrite"])
                    metrics = json.loads((output / "metrics.json").read_text())
                    base_job = job_name(phase, seed, self.baseline, task)
                    base_path = Path(self.state["jobs"][base_job]["model_dir"])
                    base_path = base_path / direction if scope == "query_ids" else base_path / "all" / direction
                    base = json.loads((base_path / "metrics.json").read_text())
                    from report import compare_results, load_result
                    data, _ = load_result(output)
                    base_data, _ = load_result(base_path)
                    comparison, _ = compare_results(data, metrics, base_data, base)
                    delta_r1 = (metrics["recall@1"] - base["recall@1"]) * 100
                    delta_ap = (metrics["mAP"] - base["mAP"]) * 100
                    row = {"artifact_job": job, "phase": phase, "seed": seed, "variant": arm["id"], "module": arm["module"],
                        "auxiliary": arm.get("auxiliary", "none"), "kind": "baseline" if arm["id"] == "baseline" else "research",
                        **task, "split": split, "gallery_scope": scope, "direction": direction,
                        "recall@1": metrics["recall@1"] * 100, "mAP": metrics["mAP"] * 100,
                        "delta_R1_pp": delta_r1, "delta_AP_pp": delta_ap, "fixed_queries": comparison["counts"]["fixed"],
                        "regressed_queries": comparison["counts"]["regressed"], "extra_parameters": metrics["extra_parameters"],
                        "training_only_parameters": metrics.get("training_only_parameters", 0), "checkpoint_sha256": record["checkpoint_sha256"]}
                    rows = self.state["results"]
                    rows[:] = [r for r in rows if (r["artifact_job"], r["gallery_scope"], r["direction"]) != (job, scope, direction)]
                    rows.append(row)
                    self.save()
                    # 先看增益再分析；final为开发集通过后的固定验证，全部生成案例以避免挑结果展示。
                    if not development or arm["id"] == "baseline" or delta_ap >= self.config["minimum_gain_ap_pp"]:
                        arguments = ["report.py", "--experiment", output, "--limit", self.config["report_limit"]]
                        if arm["id"] != "baseline":
                            arguments += ["--compare", base_path]
                        self.command(job, f"report_{scope}_{direction}", arguments)
            record.update(status="complete", finished_at=now())
            self.archive(job)
            self.save()
            if not development and seed != self.config["final_seeds"][0]:
                self.prune(job)
            return True
        except Exception:
            record.update(status="failed", error=traceback.format_exc(), failed_at=now())
            self.save()
            print(record["error"], flush=True)
            return False

    def selection(self, phases, arms):
        return assess(self.state["results"], arms, phases, self.config["screen_tasks"],
                      self.config["minimum_gain_ap_pp"], self.config["maximum_dataset_r1_drop_pp"])

    def prune_screen(self, phase):
        ranking = self.selection([phase], list(self.arms))
        # 控制存储：不足晋级阈值也保留当前前K；此排名不改变后续训练顺序。
        complete = [a for a in ranking if a["complete"]]
        complete.sort(key=lambda a: (-(a["mean_delta_AP_pp"] or 0), a["variant"]))
        keep = {a["variant"] for a in complete[:self.config["screen_keep_top_k"]]}
        for job, record in self.state["jobs"].items():
            if record["phase"] == phase and record["variant"] not in keep and any(a["variant"] == record["variant"] for a in complete):
                self.prune(job)

    def execute(self):
        try:
            self.boundary()
            if not self.state.get("preflight_passed"):
                with (self.directory / "preflight.log").open("a") as stream:
                    subprocess.run([sys.executable, "scripts/check_research.py", "--config", str(self.config_path.resolve()),
                        "--output", str(self.directory / "gpu_preflight.json")], cwd=ROOT, env=self.environment,
                        stdout=stream, stderr=subprocess.STDOUT, check=True)
                proof = json.loads((self.directory / "gpu_preflight.json").read_text())
                if not proof["passed"] or proof["map_only"] or not proof["device"].startswith("cuda"):
                    raise RuntimeError("缺少完整GPU预检证明")
                self.state["preflight_passed"] = True
                self.save()
            phases = [("screen", self.config["screen_seed"], [self.baseline, *self.arms.values()])]
            for phase, seed, arms in phases:
                for arm in arms:
                    for task in self.config["screen_tasks"]:
                        success = self.train_and_evaluate(phase, seed, arm, task)
                        if not success and arm["id"] == "baseline":
                            raise RuntimeError("同预算baseline失败，停止比较")
                    self.prune_screen(phase)
            ranking = self.selection(["screen"], list(self.arms))
            self.state["screen_selection"] = ranking
            shortlist = [r["variant"] for r in ranking if r["eligible"]][:self.config["confirm_top_k"]]
            self.state["shortlist"] = shortlist
            self.save()
            if shortlist:
                for arm in [self.baseline, *[self.arms[n] for n in shortlist]]:
                    for task in self.config["screen_tasks"]:
                        success = self.train_and_evaluate("confirm", self.config["confirmation_seed"], arm, task)
                        if not success and arm["id"] == "baseline":
                            raise RuntimeError("复验baseline失败，停止晋级")
                ranking = self.selection(["screen", "confirm"], shortlist)
                finalists = [r["variant"] for r in ranking if r["eligible"]][:self.config["final_top_k"]]
            else:
                ranking, finalists = [], []
            self.state.update(confirmation_selection=ranking, finalists=finalists)
            self.save()
            # 官方测试无论好坏均记录；不再由它决定新增候选或调参。
            if finalists:
                for seed in self.config["final_seeds"]:
                    for task in self.config["final_tasks"]:
                        for arm in [self.baseline, *[self.arms[n] for n in finalists]]:
                            success = self.train_and_evaluate("final", seed, arm, task)
                            if not success and arm["id"] == "baseline":
                                raise RuntimeError("最终baseline失败，停止同任务比较")
            failures = [k for k, v in self.state["jobs"].items() if v["status"] == "failed"]
            self.state.update(status="partial_failure" if failures else "complete", outcome="final_matrix" if finalists else "no_promotions", failures=failures)
        except BudgetReached:
            self.state.update(status="budget_reached")
        except Exception:
            self.state.update(status="partial_failure", error=traceback.format_exc())
        finally:
            self.state.update(active=None, finished_at=now())
            self.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/research_campaign.yaml"))
    parser.add_argument("--data-root", default=str(ROOT / "data"))
    parser.add_argument("--hours", type=float, help="新增任务的壁钟预算，从首次启动计时；正在运行的任务完成后停止")
    parser.add_argument("--plan", action="store_true", help="只显示预注册最大任务数，不训练、不写运行状态")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--repair-note", help="明确记录源码修复；保留旧源码快照，不延长截止时间")
    parser.add_argument("--shutdown", action="store_true", help="终止时先验证全部成功任务归档，然后请求AutoDL关机")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    if args.plan:
        print(json.dumps({"run": config["run"], "arms": len(config["arms"]),
            "screen_trainings": (len(config["arms"]) + 1) * len(config["screen_tasks"]),
            "maximum_confirmation_trainings": (config["confirm_top_k"] + 1) * len(config["screen_tasks"]),
            "maximum_final_trainings": (config["final_top_k"] + 1) * len(config["final_tasks"]) * len(config["final_seeds"]),
            "promotion": "development_only", "gpu_running": False}, indent=2))
        return
    if args.hours is None or not 0 < args.hours <= 168:
        parser.error("执行必须显式指定0–168小时内的--hours预算；无默认无限运行")
    (ROOT / "outputs").mkdir(exist_ok=True)
    with (ROOT / "outputs/.research_gpu.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("已有研究队列持有GPU锁")
        campaign = Campaign(args.config, args.data_root, args.hours, args.retry_failed, args.repair_note)
        campaign.execute()
        if args.shutdown:
            for job, record in campaign.state["jobs"].items():
                if record["status"] == "complete":
                    ready = json.loads((campaign.directory / "archive_ready" / f"{job}.json").read_text())
                    if digest(ready["bundle"]) != ready["bundle_sha256"]:
                        raise RuntimeError("归档验证失败，暂不关机")
            atomic_json(campaign.directory / "shutdown.json", {"at": now(), "reason": campaign.state["status"],
                        "location": "server_data_disk_archives", "local_download_verified": False})
            subprocess.run(["/bin/bash", "/usr/bin/shutdown"], check=True)


if __name__ == "__main__":
    main()
