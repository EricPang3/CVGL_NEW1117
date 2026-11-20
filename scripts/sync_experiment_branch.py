"""认证恢复后同步指定的已验证提交；不强推，不上传后续未指定的改动或实验产物。"""

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--branch", default="experiments/20261120-module-screen")
    parser.add_argument("--interval", type=int, default=180)
    args = parser.parse_args()
    commit = subprocess.check_output(["git", "rev-parse", args.commit + "^{commit}"], cwd=ROOT, text=True).strip()
    status_file = ROOT / "outputs/modules20261120/git_sync.json"
    status_file.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "never"}

    def remote_contains():
        try:
            result = subprocess.run(["git","ls-remote","origin",f"refs/heads/{args.branch}"],cwd=ROOT,env=env,
                                    capture_output=True,text=True,timeout=30)
            remote = result.stdout.split()[0] if result.returncode == 0 and result.stdout.strip() else None
            if remote == commit:
                return True
            return bool(remote and subprocess.run(["git","merge-base","--is-ancestor",commit,remote],cwd=ROOT,
                                                   stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode == 0)
        except subprocess.TimeoutExpired:
            return False

    def push(ref):
        command = ["git", "-c", "http.version=HTTP/1.1", "-c", "http.lowSpeedLimit=1", "-c", "http.lowSpeedTime=30", "push", "origin", ref]
        with open(os.devnull, "w") as output:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=output, stderr=output, start_new_session=True)
            try:
                return process.wait(timeout=90) == 0
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                return False

    state = {"commit": commit, "branch": args.branch, "status": "waiting_credentials_or_network", "attempts": 0}
    while True:
        state["attempts"] += 1
        state["updated_at"] = time.time()
        ok = remote_contains() or push(f"{commit}:refs/heads/{args.branch}")
        if ok:
            state["baseline_tag_synced"] = push("refs/tags/baseline-batch24-v2-20261120:refs/tags/baseline-batch24-v2-20261120")
            state["status"] = "complete"
        temporary = status_file.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
        temporary.replace(status_file)
        if ok:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
