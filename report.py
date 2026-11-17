"""将评估结果转换为自包含离线案例报告，并打包 report.zip。"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from PIL import Image


def build_report(directory, limit=200):
    directory = Path(directory).expanduser().resolve()
    data = json.loads((directory / "queries.json").read_text(encoding="utf-8"))
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    if limit < 0:
        raise ValueError("limit 必须非负；0 表示全部")
    # 无正样本和排名靠后的案例优先；所有查询仍保存在 queries.json。
    queries = sorted(data["queries"], key=lambda q: (q["valid"], -(q["first_positive_rank"] or 0)))
    queries = queries[:limit] if limit else queries
    report = directory / "report"
    images = report / "images"
    images.mkdir(parents=True, exist_ok=True)
    copied = {}

    def picture(record):
        if record is None:
            return None
        path = record["path"]
        if path not in copied:
            name = hashlib.sha256(path.encode()).hexdigest()[:24] + ".jpg"
            with Image.open(path) as image:
                image = image.convert("RGB")
                image.thumbnail((384, 384))
                image.save(images / name, quality=85)
            copied[path] = f"images/{name}"
        return {**record, "image": copied[path]}

    cases = []
    for query in queries:
        cases.append({**query, "image": picture(query)["image"],
                      "best_positive": picture(query["best_positive"]),
                      "top_k": [picture(candidate) for candidate in query["top_k"]]})
    used_images = {Path(path).name for path in copied.values()}
    for image in images.glob("*.jpg"):
        if image.name not in used_images:
            image.unlink()
    payload = {"metrics": metrics, "evaluation": data.get("evaluation", {}),
               "cases": cases, "total": len(data["queries"])}
    encoded = json.dumps(payload, ensure_ascii=False).replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    (report / "index.html").write_text(HTML.replace("__DATA__", encoded), encoding="utf-8")
    shutil.make_archive(str(directory / "report"), "zip", root_dir=report)
    print(f"报告：{report / 'index.html'}；包含 {len(cases)}/{payload['total']} 个案例")
    print(f"离线下载：{directory / 'report.zip'}")
    return report


HTML = r'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CVGL · 检索案例报告</title>
<style>
:root{font-family:system-ui,sans-serif;color:#172b3a;background:#f2f5f7}body{max-width:1440px;margin:auto;padding:32px}h1{font-size:28px;margin:8px 0}p{color:#576b78}header{border-bottom:1px solid #ccd8df;padding-bottom:20px}.tag{color:#16766a;font-size:13px;letter-spacing:2px}.metrics{display:flex;gap:12px;flex-wrap:wrap;margin:24px 0}.metric{background:white;border:1px solid #dae2e7;border-radius:8px;padding:16px 24px}.metric strong{display:block;font-size:26px}nav{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:20px 0}input,select,button{font:inherit;padding:9px 12px;border:1px solid #bdcdd7;border-radius:6px;background:white}input{min-width:240px}button{cursor:pointer}button:disabled{opacity:.4}.case{padding:20px;background:white;border:1px solid #dae2e7;border-radius:10px;margin:20px 0}.case h2{font-size:16px;margin-top:0}.strip{display:flex;gap:12px;overflow-x:auto;padding-bottom:10px}figure{margin:0;flex:0 0 168px}figure img{width:164px;height:150px;object-fit:contain;background:#f5f7f8;border:2px solid #cad5dd;border-radius:5px}figure.ok img{border-color:#16826b}figure.bad img{border-color:#ce6760}figcaption{font-size:12px;line-height:1.7;overflow-wrap:anywhere}.muted{color:#637987}#status{font-size:14px}footer{margin:30px 0;color:#637987}@media(max-width:600px){body{padding:16px}.metric{padding:12px}figure{flex-basis:145px}figure img{width:140px;height:130px}}
</style>
<header><div class="tag">CVGL / SAMPLE4GEO BASELINE</div><h1>无人机—卫星检索案例</h1><p id="context"></p></header>
<section class="metrics" id="metrics"></section>
<p id="coverage"></p><p class="muted">绿色：正确地点；红色：错误地点。分数为余弦相似度。AP 使用 S4G 梯形积分。</p>
<nav><label>筛选 <select id="filter"><option value="all">全部案例</option><option value="error">检索错误</option><option value="correct">检索正确</option><option value="missing">图库无正确匹配</option></select></label><input id="search" aria-label="搜索地点或文件名" placeholder="搜索地点 ID 或文件名"><span id="status"></span></nav>
<main id="cases"></main><nav><button id="prev">上一页</button><span id="page"></span><button id="next">下一页</button></nav>
<footer>本报告包含本地缩略图，无外部脚本、字体或网络请求。queries.json 保存全部逐图结果。</footer>
<script id="data" type="application/json">__DATA__</script>
<script>
const data=JSON.parse(document.getElementById('data').textContent),$=id=>document.getElementById(id),size=20;
let page=0;
function node(tag,text,cls){const el=document.createElement(tag);if(text!==undefined)el.textContent=text;if(cls)el.className=cls;return el;}
const cfg=data.evaluation;
$('context').textContent=[cfg.dataset,cfg.direction,cfg.altitude?cfg.altitude+' m':'',cfg.checkpoint?'权重：'+cfg.checkpoint.split('/').pop():''].filter(Boolean).join(' · ');
for(const key of ['recall@1','recall@5','recall@10','recall@1%','mAP']){const el=node('div',undefined,'metric');el.append(node('span',key),node('strong',(data.metrics[key]*100).toFixed(2)+'%'));$('metrics').append(el);}
$('coverage').textContent=`查询 ${data.total} 张；报告收录 ${data.cases.length} 张，优先展示排名靠后和缺少正确匹配的案例。有效图库 ${data.metrics.gallery_count} 张；规则 ${data.metrics.protocol.gallery_scope}。`;
function figure(record,title,cls){const el=node('figure',undefined,cls);if(!record){el.append(node('figcaption',title+'：无'));return el;}const img=node('img');img.src=record.image;img.alt=title+' / '+record.id;img.loading='lazy';const caption=node('figcaption',title+' · 地点 '+record.id);if(record.score!==undefined)caption.append(node('div','相似度 '+record.score.toFixed(4)));caption.append(node('div',record.path.split('/').pop(),'muted'));el.append(img,caption);return el;}
function render(){const value=$('filter').value,search=$('search').value.toLowerCase();const cases=data.cases.filter(q=>(value==='all'||value==='error'&&q.valid&&!q.top1_correct||value==='correct'&&q.top1_correct||value==='missing'&&!q.valid)&&(!search||(q.id+' '+q.path).toLowerCase().includes(search)));const pages=Math.max(1,Math.ceil(cases.length/size));page=Math.min(page,pages-1);$('cases').replaceChildren();for(const q of cases.slice(page*size,(page+1)*size)){const box=node('article',undefined,'case');box.append(node('h2',`地点 ${q.id} · 首个正确排名 ${q.first_positive_rank??'无'} · AP ${q.ap===null?'无':(q.ap*100).toFixed(2)+'%'}`));const strip=node('div',undefined,'strip');strip.append(figure(q,'查询图'),figure(q.best_positive,'最佳正确匹配','ok'));q.top_k.forEach((r,i)=>strip.append(figure(r,'Top '+(i+1),r.correct?'ok':'bad')));box.append(strip);$('cases').append(box);}if(!cases.length)$('cases').append(node('p','没有符合条件的案例。'));$('status').textContent=`匹配 ${cases.length} 个案例`;$('page').textContent=`${page+1} / ${pages}`;$('prev').disabled=page===0;$('next').disabled=page===pages-1;}
$('filter').onchange=$('search').oninput=()=>{page=0;render();};$('prev').onclick=()=>{page--;render();};$('next').onclick=()=>{page++;render();};render();
</script></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, help="包含 metrics.json 和 queries.json 的目录")
    parser.add_argument("--limit", type=int, default=200, help="最多展示的案例数；0 表示全部")
    args = parser.parse_args()
    build_report(args.experiment, args.limit)


if __name__ == "__main__":
    main()
