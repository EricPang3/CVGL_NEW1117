"""本地接收后台实验：校验后归档权重/报告，生成总览，完成后可关闭AutoDL。"""

import argparse
import hashlib
import html
import json
import shlex
import subprocess
import tarfile
import time
import zipfile
from pathlib import Path

try:
    from scripts.experiment_records import export_tables, keep_checkpoint, load_policy
except ModuleNotFoundError:
    from experiment_records import export_tables, keep_checkpoint, load_policy

ROOT = Path(__file__).resolve().parents[1]

REMOTE_POLL = r'''
import json,sys,hashlib
from pathlib import Path
import yaml
name=sys.argv[1]
if Path(name).name!=name:raise RuntimeError('非法运行名')
p=Path('/root/autodl-tmp/CVGL_NEW/outputs')/name
state=json.loads((p/'state.json').read_text()) if (p/'state.json').exists() else {'status':'not_started'}
ready=[json.loads(f.read_text()) for f in sorted((p/'archive_ready').glob('*.json'))]
logs=[]
terminal=state.get('status') in ('complete','partial_failure','budget_reached')
paths=[(f,f.name,state.get('jobs',{}).get(f.stem,{}).get('status') in ('complete','failed') or terminal) for f in p.glob('*.log')]
protocol=yaml.safe_load((p/'protocol.yaml').read_text()) if (p/'protocol.yaml').exists() else {}
baseline=p.parent/protocol.get('baseline_run','matrix20261120_none')
if (baseline/'run.log').exists():paths.append((baseline/'run.log','baseline_run.log',True))
launcher=p.parent/(name+'.launch.log')
if launcher.exists():paths.append((launcher,'suite_launcher.log',terminal))
for f,label,complete in paths:
 info=f.stat();logs.append({'path':str(f),'name':label,'size':info.st_size,'mtime':info.st_mtime_ns,'sha256':hashlib.sha256(f.read_bytes()).hexdigest() if complete else None})
summary=[n for n in ('commands.jsonl','metrics.csv','protocol.yaml','gpu_preflight.json') if (p/n).is_file()]
print(json.dumps({'state':state,'ready':ready,'logs':logs,'summary_files':summary}))
'''

REMOTE_ACK = r'''
import json,sys,hashlib,shutil
from pathlib import Path
a=json.loads(sys.argv[1]);root=Path('/root/autodl-tmp/CVGL_NEW/outputs');p=Path(a['model_dir']).resolve()
if p.parent!=root or not (p.name.startswith(a['run']+'_') or a.get('retain_remote')):raise RuntimeError('归档路径不在本批次')
if a.get('archive_checkpoint'):
 w=Path(a['checkpoint'])
 if w.resolve().parent!=p:raise RuntimeError('权重路径不在当前实验目录')
 if w.exists():
  h=hashlib.sha256()
  with w.open('rb') as stream:
   for b in iter(lambda:stream.read(1048576),b''):h.update(b)
  if h.hexdigest()!=a['checkpoint_sha256']:raise RuntimeError('远程权重已变化，禁止清理')
 if not a.get('retain_remote') and w.exists():
  if w.parent!=p:raise RuntimeError('权重路径不在当前实验目录')
  w.unlink()
if not a.get('retain_remote'):
 for direction in ('D2S','S2D'):
  e=p/direction
  for name in ('queries.json','report.zip'):(e/name).unlink(missing_ok=True)
  if (e/'report').is_dir():shutil.rmtree(e/'report')
 Path(a['bundle']).unlink(missing_ok=True)
(p/'ARCHIVED_LOCALLY.json').write_text(json.dumps(a,ensure_ascii=False,indent=2))
print('acknowledged',a['job'])
'''


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def begin_revision(directory, manifest, ready):
    """重跑产生新包时归档旧版本；相同包不重复接收，绝不覆盖旧权重。"""
    job = ready["job"]
    previous = manifest["artifacts"].get(job)
    if previous is None:
        return True
    if all(previous.get(k) == ready.get(k) for k in ("bundle_sha256", "checkpoint_sha256")):
        return False
    old = directory / "history" / job / previous["bundle_sha256"]
    target = directory / "artifacts" / job
    if not old.exists() and target.exists():
        old.parent.mkdir(parents=True, exist_ok=True)
        target.rename(old)
    records = manifest.setdefault("artifact_history", {}).setdefault(job, [])
    if not any(r["bundle_sha256"] == previous["bundle_sha256"] for r in records):
        records.append({**previous, "local_directory": str(old)})
    del manifest["artifacts"][job]
    save(directory / "archive_manifest.json", manifest)
    return True


def dashboard(directory, remote, manifest):
    results = remote.get("results", [])
    rows = []
    for row in results:
        label = "u1652" if row["dataset"] == "U1652" else f'sues200_{row["altitude"]}'
        job = f'{row["module"]}_{label}'
        url = f'artifacts/{job}/{row["direction"]}/report/index.html'
        entry = manifest["artifacts"].get(job, {})
        current = entry.get("checkpoint_sha256") == row.get("checkpoint_sha256")
        link = f'<a href="{url}">{html.escape(row["module"])}</a>' if entry and current else html.escape(row["module"])
        cells = [link, html.escape(row["kind"]), html.escape(row["dataset"]), str(row["altitude"] or "—"), row["direction"],
                 f'{row["recall@1"]:.4f}', f'{row["mAP"]:.4f}', f'{row["delta_R1_pp"]:+.4f}', f'{row["delta_AP_pp"]:+.4f}',
                 str(row.get("fixed_queries", "—")), str(row.get("regressed_queries", "—")), str(row.get("extra_parameters", "—")),
                 html.escape(entry.get("checkpoint_storage", "待归档"))]
        rows.append('<tr>' + ''.join('<td>'+c+'</td>' for c in cells) + '</tr>')
    status = html.escape(str(remote.get("status")))
    active = html.escape(str(remote.get("active", "")))
    page = '<!doctype html><meta charset="utf-8"><title>33模块实验总览</title><style>body{font-family:system-ui;padding:24px;color:#243543}table{border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}input{padding:10px;width:350px}a{color:#176b75}</style>'
    page += f'<h1>33模块实验总览</h1><p>运行：{html.escape(directory.name)}；seed：{remote.get("seed", 1)}；状态：{status}；当前：{active}</p><p>真实评估 {len(results)}/370；本地已归档 {len(manifest["artifacts"])}/{remote.get("expected_archive_tasks", 190)} 个任务。点击已归档模块看案例；分数为百分比，差值为百分点。</p>'
    page += '<p>每页默认展示最差200例，全部逐图结果在对应queries.json。权重按configs/checkpoint_retention.json保留；discarded_by_policy表示只保留结果，shared_baseline表示共用baseline权重。基线训练、模块训练和重排拟合预算不同，比较时同时查看配置、参数与耗时。</p>'
    page += '<input placeholder="筛选模块、数据集或方向" oninput="for(const r of document.querySelectorAll(\'tbody tr\'))r.hidden=!r.textContent.toLowerCase().includes(this.value.toLowerCase())">'
    page += '<table><thead><tr>'+''.join('<th>'+k+'</th>' for k in ('模块','类型','数据集','高度','方向','R@1','AP','ΔR@1','ΔAP','修好','退步','额外参数','权重状态'))+'</tr></thead><tbody>'+''.join(rows)+'</tbody></table>'
    page += '<p>状态与命令：state.json、commands.jsonl；归档清单：archive_manifest.json。单seed差异不能证明稳定增益；人工查看案例用于形成假设。</p>'
    (directory / "index.html").write_text(page, encoding="utf-8")


def main(argv=None, on_poll=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--run", default="modules20261120")
    parser.add_argument("--socket", default="/tmp/cvgl-ssh-22352")
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--shutdown-when-complete", action="store_true")
    args = parser.parse_args(argv)
    if Path(args.run).name != args.run:
        parser.error("run必须是单个目录名")
    directory = ROOT / "outputs" / args.run
    directory.mkdir(parents=True, exist_ok=True)
    artifacts = directory / "artifacts"
    artifacts.mkdir(exist_ok=True)
    manifest_file = directory / "archive_manifest.json"
    manifest = json.loads(manifest_file.read_text()) if manifest_file.exists() else {"artifacts": {}, "status": "collecting"}
    policy = load_policy(ROOT)
    policy_hash = file_hash(ROOT / "configs/checkpoint_retention.json") if policy else None
    common = ["-o", f"ControlPath={args.socket}", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
              "-o", "UserKnownHostsFile=/tmp/cvgl_known_hosts", "-o", "StrictHostKeyChecking=yes"]
    ssh = ["ssh", *common, "-p", str(args.port), args.host]

    def command(code, argument=None):
        shell = "/root/miniconda3/bin/python -c " + shlex.quote(code)
        shell += " " + shlex.quote(json.dumps(argument, ensure_ascii=False) if argument is not None else args.run)
        return subprocess.check_output([*ssh, shell], text=True, timeout=120)

    def download(remote, local):
        partial = local.with_suffix(local.suffix + ".partial")
        subprocess.run(["scp", *common, "-P", str(args.port), f"{args.host}:{remote}", str(partial)], check=True, timeout=1800)
        partial.replace(local)

    while True:
        try:
            polled = json.loads(command(REMOTE_POLL))
            remote = polled["state"]
            save(directory / "state.json", remote)
            if on_poll is not None:
                on_poll(remote)
            export_tables(ROOT)
            for ready in polled["ready"]:
                job = ready["job"]
                if Path(job).name != job:
                    raise RuntimeError("非法任务名")
                if not begin_revision(directory, manifest, ready):
                    continue
                target = artifacts / job
                target.mkdir(exist_ok=True)
                print("receiving", job, flush=True)
                bundle = target / "artifacts.tar.gz"
                if not bundle.exists() or file_hash(bundle) != ready["bundle_sha256"]:
                    download(ready["bundle"], bundle)
                if file_hash(bundle) != ready["bundle_sha256"]:
                    raise RuntimeError(f"结果包校验失败：{job}")
                retain_weight = keep_checkpoint(policy, remote.get("jobs", {}).get(job), ready)
                if ready["archive_checkpoint"] and retain_weight:
                    weight = target / "model.pth"
                    if not weight.exists() or file_hash(weight) != ready["checkpoint_sha256"]:
                        download(ready["checkpoint"], weight)
                    if file_hash(weight) != ready["checkpoint_sha256"]:
                        raise RuntimeError(f"权重校验失败：{job}")
                with tarfile.open(bundle) as archive:
                    for member in archive.getmembers():
                        if not (target / member.name).resolve().is_relative_to(target.resolve()) or member.issym() or member.islnk():
                            raise RuntimeError("结果包出现非法路径")
                    archive.extractall(target)
                for direction in ("D2S", "S2D"):
                    report_zip = target / direction / "report.zip"
                    if report_zip.is_file():
                        with zipfile.ZipFile(report_zip) as archive:
                            for name in archive.namelist():
                                if not (target / direction / "report" / name).resolve().is_relative_to(target.resolve()):
                                    raise RuntimeError("报告ZIP出现非法路径")
                            archive.extractall(target / direction / "report")
                if ready["archive_checkpoint"]:
                    ready["checkpoint_storage"] = "retained" if retain_weight else "discarded_by_policy"
                    if not retain_weight:
                        (target / "model.pth").unlink(missing_ok=True)
                        (target / "model.pth.partial").unlink(missing_ok=True)
                else:
                    ready["checkpoint_storage"] = "shared_baseline"
                ready["checkpoint_retention_policy_sha256"] = policy_hash
                ready["local_directory"] = str(target)
                ready["run"] = args.run
                # 结果包必须先校验；选中权重也必须本地校验。用户允许丢弃的权重由远端校验SHA后清理。
                command(REMOTE_ACK, ready)
                manifest["artifacts"][job] = ready
                save(manifest_file, manifest)
                dashboard(directory, remote, manifest)
                print("archived", job, flush=True)
            dashboard(directory, remote, manifest)
            logs_dir = directory / "logs"
            logs_dir.mkdir(exist_ok=True)
            known_logs = manifest.setdefault("logs", {})
            for record in polled.get("logs", []):
                name = record["name"]
                if Path(name).name != name:
                    raise RuntimeError("非法日志名")
                target = logs_dir / name
                if known_logs.get(name) != record or not target.exists():
                    download(record["path"], target)
                    if record["sha256"] and file_hash(target) != record["sha256"]:
                        raise RuntimeError(f"日志写入中，下一轮重试：{name}")
                    known_logs[name] = record
                    save(manifest_file, manifest)
            if remote.get("status") in ("complete", "partial_failure", "budget_reached"):
                manifest["status"] = remote["status"]
                manifest["expected_archive_tasks"] = remote.get("expected_archive_tasks", 190)
                save(manifest_file, manifest)
                # 所有ready任务已经校验归档；保存完整命令和汇总后才能关机。
                for name in polled["summary_files"]:
                    download(f"/root/autodl-tmp/CVGL_NEW/outputs/{args.run}/{name}", directory / name)
                if remote["status"] in ("complete", "budget_reached") and len(manifest["artifacts"]) == remote.get("expected_archive_tasks", 190):
                    export_tables(ROOT)
                    if args.shutdown_when_complete:
                        subprocess.run([*ssh, "/bin/bash /usr/bin/shutdown"], check=True, timeout=120)
                        manifest["shutdown"] = "requested_after_verified_local_archive"
                        save(manifest_file, manifest)
                    print("complete: local results", directory, flush=True)
                    return
                if remote["status"] == "partial_failure":
                    print("有失败任务，保留运行环境，等待修复续跑：", remote.get("failures"), flush=True)
            manifest["last_poll"] = time.time()
            manifest.pop("last_error", None)
            save(manifest_file, manifest)
        except Exception as error:
            manifest["last_error"] = str(error)
            save(manifest_file, manifest)
            print("receiver retry", type(error).__name__, str(error), flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
