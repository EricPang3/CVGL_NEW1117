"""仅在原队列结束后执行已验证的指定模块重跑，保留旧记录和目录。"""

import json
import subprocess
from pathlib import Path

try:
    from scripts.module_suite import now, write_json
except ModuleNotFoundError:
    from module_suite import now, write_json


def prepare_repair(root, config, request):
    identifier = request["id"]
    if Path(identifier).name != identifier or identifier in (".", ".."):
        raise ValueError("非法修复标识")
    modules = set(request["modules"])
    if not modules or not modules.issubset({m["id"] for m in config["modules"]}):
        raise ValueError("修复模块不在注册矩阵中")
    if (root / ".git").exists():
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        if commit != request["validated_commit"]:
            raise RuntimeError("当前代码与验证过的修复提交不同")
    directory = root / "outputs" / config["run"]
    history = directory / "repairs" / identifier
    if (history / "applied.json").exists():
        return False
    state_file = directory / "state.json"
    state = json.loads(state_file.read_text())
    if state["status"] not in ("complete", "partial_failure"):
        raise RuntimeError("原队列尚未结束，禁止修改其状态")
    history.mkdir(parents=True, exist_ok=True)
    if not (history / "state_before.json").exists():
        write_json(history / "state_before.json", state)
    selected = [k for k, v in state["jobs"].items() if v.get("module") in modules]
    if not selected:
        raise ValueError("指定模块没有可修复任务")
    old_ready = history / "archive_ready"
    old_ready.mkdir(exist_ok=True)
    for job in selected:
        model = root / "outputs" / f'{config["run"]}_{job}'
        backup = model.with_name(model.name + "_before_" + identifier)
        if model.exists():
            if backup.exists():
                raise RuntimeError(f"发现已有修复备份，保留现场：{model}")
            model.rename(backup)
        ready = directory / "archive_ready" / (job + ".json")
        if ready.exists():
            ready.rename(old_ready / ready.name)
        previous = state["jobs"][job]
        state["jobs"][job] = {k: previous[k] for k in ("module", "dataset", "altitude")}
        state["jobs"][job].update(completed=[], status="pending", repair_id=identifier)
    state["results"] = [r for r in state["results"] if r["module"] not in modules]
    state.update(status="repair_pending", active=None, updated_at=now())
    state["failures"] = [k for k, v in state["jobs"].items() if v.get("status") == "failed"]
    state.setdefault("events", []).append({"at": now(), "repair_id": identifier, "modules": sorted(modules),
                                           "reason": request["reason"], "validated_commit": request["validated_commit"]})
    write_json(state_file, state)
    write_json(history / "applied.json", {**request, "jobs": selected, "applied_at": now()})
    return True
