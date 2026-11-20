"""本地跟随备用队列；到预算边界且各运行归档成功后才请求关闭AutoDL。"""

import argparse
import html
import json
import shlex
import subprocess
import time
from pathlib import Path

try:
    from scripts.collect_experiments import main as collect, save
    from scripts.experiment_budget import budget_expired, parse_deadline
except ModuleNotFoundError:
    from collect_experiments import main as collect, save
    from experiment_budget import budget_expired, parse_deadline

ROOT = Path(__file__).resolve().parents[1]


def verified_manifest(manifest):
    return (manifest.get("status") in ("complete", "budget_reached")
            and "expected_archive_tasks" in manifest
            and len(manifest.get("artifacts", {})) == manifest["expected_archive_tasks"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--name", default="overnight20261120")
    parser.add_argument("--socket", default="/tmp/cvgl-ssh-22352")
    parser.add_argument("--shutdown", action="store_true")
    args = parser.parse_args()
    if Path(args.name).name != args.name:
        parser.error("name必须为单个目录名")
    directory = ROOT / "outputs" / args.name
    directory.mkdir(parents=True, exist_ok=True)
    ssh = ["ssh", "-o", f"ControlPath={args.socket}", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
           "-o", "UserKnownHostsFile=/tmp/cvgl_known_hosts", "-o", "StrictHostKeyChecking=yes", "-p", str(args.port), args.host]
    state = {"status": "following", "verified_runs": []}
    while True:
        try:
            code = f"from pathlib import Path; print(Path('/root/autodl-tmp/CVGL_NEW/outputs/{args.name}/state.json').read_text())"
            plan = json.loads(subprocess.check_output([*ssh, "/root/miniconda3/bin/python -c " + shlex.quote(code)],
                                                     text=True, timeout=60))
            save(directory / "state.json", plan)
            links = ''.join(f'<li><a href="../{html.escape(run)}/index.html">{html.escape(run)}</a></li>' for run in plan["runs"])
            (directory / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>夜间实验</title>'
                '<h1>夜间实验</h1><p>原矩阵完成后固定按seed2、3…及原模块顺序复验，各seed独立训练baseline。</p>'
                f'<p>状态：{html.escape(plan["status"])}；当前：{html.escape(str(plan.get("active_run")))}；新任务截止：{html.escape(plan["until"])}</p>'
                f'<ul>{links}</ul><p>只在任务边界停止。未完成的后续任务保持未运行，不计为成功。所有结果校验归档后关机。</p>', encoding="utf-8")
            for run in plan["runs"]:
                manifest = ROOT / "outputs" / run / "archive_manifest.json"
                value = json.loads(manifest.read_text()) if manifest.exists() else {}
                if not verified_manifest(value):
                    collect(["--host", args.host, "--port", str(args.port), "--run", run, "--socket", args.socket])
                    value = json.loads(manifest.read_text())
                    if not verified_manifest(value):
                        raise RuntimeError(f"归档不完整：{run}")
                if run not in state["verified_runs"]:
                    state["verified_runs"].append(run)
            state.update(status=plan["status"], last_poll=time.time())
            state.pop("last_error", None)
            save(directory / "client_state.json", state)
            if (plan["status"] == "finished" and budget_expired(parse_deadline(plan["until"]))
                    and set(state["verified_runs"]) == set(plan["runs"])):
                if args.shutdown:
                    subprocess.run([*ssh, "/bin/bash /usr/bin/shutdown"], check=True, timeout=120)
                    state["shutdown"] = "requested_after_budget_and_verified_archives"
                    save(directory / "client_state.json", state)
                return
        except Exception as error:
            state["last_error"] = str(error)
            save(directory / "client_state.json", state)
            print("overnight receiver retry", type(error).__name__, str(error), flush=True)
        time.sleep(30)


if __name__ == "__main__":
    main()
