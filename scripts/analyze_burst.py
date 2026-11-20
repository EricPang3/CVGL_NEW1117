"""对已归档burst_gap按预固定重复强度分组，报告观察性变化，不据测试集调参。"""

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BINS = [(0, .1), (.1, .3), (.3, .6), (.6, 1.000001)]


def main():
    directory = ROOT / "outputs/modules20261120"
    artifacts = directory / "artifacts"
    groups, tables = [], []
    for current in sorted(artifacts.glob("burst_gap_*")):
        baseline = artifacts / ("baseline_none_" + current.name.removeprefix("burst_gap_"))
        for direction in ("D2S", "S2D"):
            path = current / direction / "queries.json"
            reference = baseline / direction / "queries.json"
            if not path.exists() or not reference.exists():
                continue
            after = json.loads(path.read_text(encoding="utf-8"))["queries"]
            before = {q["key"]: q for q in json.loads(reference.read_text(encoding="utf-8"))["queries"]}
            rows = []
            for low, high in BINS:
                selected = [q for q in after if q["valid"] and low <= q.get("diagnostics", {}).get("repetition_density", -1) < high]
                if not selected:
                    continue
                old = [before[q["key"]] for q in selected]
                row = {"task": current.name, "direction": direction, "density_min": low, "density_max": min(1.,high),
                       "queries": len(selected), "baseline_R1": sum(q["top1_correct"] for q in old)/len(old)*100,
                       "burst_R1": sum(q["top1_correct"] for q in selected)/len(selected)*100,
                       "baseline_AP": sum(q["ap"] for q in old)/len(old)*100,
                       "burst_AP": sum(q["ap"] for q in selected)/len(selected)*100,
                       "fixed": sum(not a["top1_correct"] and b["top1_correct"] for a,b in zip(old,selected)),
                       "regressed": sum(a["top1_correct"] and not b["top1_correct"] for a,b in zip(old,selected))}
                groups.append(row)
                rows.append('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in (
                    f'{low:.1f}–{min(1.,high):.1f}',len(selected),f'{row["baseline_R1"]:.3f}',f'{row["burst_R1"]:.3f}',
                    f'{row["burst_R1"]-row["baseline_R1"]:+.3f}',f'{row["baseline_AP"]:.3f}',f'{row["burst_AP"]:.3f}',row["fixed"],row["regressed"]))+'</tr>')
            tables.append('<h2>'+html.escape(current.name+' '+direction)+'</h2><table><tr>'+''.join('<th>'+x+'</th>' for x in ('重复强度','查询数','基线R@1','降权R@1','差值pp','基线AP','降权AP','修好','退步'))+'</tr>'+''.join(rows)+'</table>')
    (directory / "burst_mechanism.json").write_text(json.dumps({"fixed_bins":BINS,"interpretation":"observational; not causal; no test-based threshold tuning","groups":groups},ensure_ascii=False,indent=2),encoding="utf-8")
    page='<!doctype html><meta charset="utf-8"><title>重复纹理分组诊断</title><style>body{font-family:system-ui;padding:24px}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}table{border-collapse:collapse}</style><h1>burst_gap：重复纹理分组诊断</h1><p>分组边界固定为0/.1/.3/.6/1，强度来自特征token的非对角自相似软计数。表中是观察性相关，不能单独证明因果，也未根据测试结果选阈值。</p><p>同权重、同查询/图库，仅改变无训练的重复区域降权；数值为百分比，差值为百分点。</p>'+''.join(tables)
    (directory / "burst_mechanism.html").write_text(page,encoding="utf-8")
    print("burst analysis",len(groups),"groups",directory/'burst_mechanism.html')


if __name__ == "__main__":
    main()
