"""串行训练 U1652 / SUES 四高度，双向评估、离线报告并打包；不重复实现训练。"""

import argparse
import csv
import html
import json
import os
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="新的运行名称；已有目录拒绝覆盖")
    parser.add_argument("--datasets", nargs="+", choices=["U1652", "SUES200"], default=["U1652", "SUES200"])
    parser.add_argument("--altitudes", nargs="+", type=int, choices=[150, 200, 250, 300], default=[150, 200, 250, 300])
    parser.add_argument("--data-root", help="两个数据集的父目录，等同 CVGL_DATA_ROOT")
    parser.add_argument("--u1652-config", default=str(ROOT / "configs/u1652.yaml"))
    parser.add_argument("--sues200-config", default=str(ROOT / "configs/sues200.yaml"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--gallery-scope", choices=["query_ids", "all"])
    parser.add_argument("--report-limit", type=int, default=0, help="0 包含全部缩略图；限量包不能离线查看未收录的图片")
    parser.add_argument("--shutdown", action="store_true", help="仅 AutoDL：全部成功并完成打包后关机")
    args = parser.parse_args()
    if Path(args.name).name != args.name or args.name in (".", "..") or args.report_limit < 0:
        parser.error("name 必须是单个目录名，report-limit 必须非负")
    tasks = []
    for dataset in dict.fromkeys(args.datasets):
        for altitude in dict.fromkeys(args.altitudes) if dataset == "SUES200" else [None]:
            label = "u1652" if altitude is None else f"sues200_{altitude}"
            name = f"{args.name}_{label}"
            if (ROOT / "outputs" / name).exists():
                parser.error(f"实验已存在：{name}；请换 --name")
            tasks.append((dataset, altitude, label, name))
    directory = ROOT / "outputs" / args.name
    directory.mkdir(parents=True, exist_ok=False)
    status = {"status": "running", "started_at": datetime.now().astimezone().isoformat(), "steps": [], "results": []}
    environment = dict(os.environ)
    if args.data_root:
        environment["CVGL_DATA_ROOT"] = str(Path(args.data_root).expanduser().resolve())

    def save():
        (directory / "run.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")

    def run(label, command):
        status["active"] = label
        save()
        print(f"{label}；日志：{directory / 'run.log'}", flush=True)
        with (directory / "run.log").open("a", encoding="utf-8") as log:
            log.write(json.dumps(command, ensure_ascii=False) + "\n")
            log.flush()
            subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        status["steps"].append(label)
        save()

    save()
    try:
        reports = []
        for dataset, altitude, label, name in tasks:
            model = ROOT / "outputs" / name
            command = [sys.executable, "train.py", "--config", args.u1652_config if dataset == "U1652" else args.sues200_config,
                       "--dataset", dataset, "--name", name]
            if altitude:
                command += ["--altitude", str(altitude)]
            if args.seed is not None:
                command += ["--seed", str(args.seed)]
            run(f"train_{label}", command)
            for direction in ("D2S", "S2D"):
                output = model / direction
                command = [sys.executable, "evaluate.py", "--checkpoint", str(model / "model.pth"),
                           "--direction", direction, "--output", str(output)]
                if args.gallery_scope:
                    command += ["--gallery-scope", args.gallery_scope]
                run(f"eval_{label}_{direction}", command)
                run(f"report_{label}_{direction}", [sys.executable, "report.py", "--experiment", str(output), "--limit", str(args.report_limit)])
                metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
                row = {"dataset": dataset, "altitude": altitude, "direction": direction,
                       "gallery_scope": metrics["protocol"]["gallery_scope"],
                       "queries": metrics["query_count"], "gallery": metrics["gallery_count"],
                       **{key: metrics[key] * 100 for key in ("recall@1", "recall@5", "recall@10", "recall@1%", "mAP")}}
                status["results"].append(row)
                reports.append((f"{label}_{direction}", output / "report", row))
                save()
        with (directory / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(status["results"][0]))
            writer.writeheader()
            writer.writerows(status["results"])
        rows = "".join(f'<tr><td><a href="{label}/index.html">{html.escape(label)}</a></td>'
                       f'<td>{row["gallery_scope"]}</td><td>{row["queries"]}</td><td>{row["gallery"]}</td>'
                       f'<td>{row["recall@1"]:.2f}%</td><td>{row["mAP"]:.2f}%</td></tr>' for label, _, row in reports)
        (directory / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>CVGL 实验结果</title>'
            '<h1>CVGL 实验结果</h1><p>点击实验查看离线案例；CSV 指标为百分比。模型权重保留在服务器。</p>'
            '<table cellpadding="10"><tr><th>实验</th><th>图库规则</th><th>查询数</th><th>图库数</th><th>R@1</th><th>mAP</th></tr>' + rows + '</table>', encoding="utf-8")
        status.update(status="complete", finished_at=datetime.now().astimezone().isoformat(), active=None)
        save()
        with zipfile.ZipFile(directory / "results.tmp", "w", zipfile.ZIP_DEFLATED) as archive:
            for filename in ("index.html", "metrics.csv", "run.json"):
                archive.write(directory / filename, filename)
            for label, report, _ in reports:
                for path in sorted(report.rglob("*")):
                    if path.is_file():
                        archive.write(path, Path(label) / path.relative_to(report))
        (directory / "results.tmp").replace(directory / "results.zip")
    except Exception as error:
        status.update(status="failed", error=str(error))
        save()
        raise
    print(f"完成。下载 {directory / 'results.zip'}，解压并打开 index.html。", flush=True)
    if args.shutdown:
        # AutoDL 此路径是无 shebang 脚本，需要显式交给 bash；只在全部成功后执行。
        subprocess.run(["/bin/bash", "/usr/bin/shutdown"], check=True)


if __name__ == "__main__":
    main()
