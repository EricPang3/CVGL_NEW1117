"""将评估结果转换为自包含离线案例报告，并打包 report.zip。"""

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

from PIL import Image


def query_key(record, evaluation):
    if record.get("key"):
        return record["key"]
    try:
        return Path(record["path"]).relative_to(evaluation["data_root"]).as_posix()
    except (KeyError, ValueError):
        return record["path"]


def load_result(directory):
    data = json.loads((directory / "queries.json").read_text(encoding="utf-8"))
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    for record in [*data["queries"], *data.get("gallery", [])]:
        record["key"] = query_key(record, data.get("evaluation", {}))
    return data, metrics


def compare_results(current, current_metrics, previous, previous_metrics, allow_reranking=False):
    """只允许同一评估任务比较，避免把换图库/划分误认为模型进步。"""
    for key in ("dataset", "direction", "altitude", "img_size", "mean", "std", "amp"):
        if current.get("evaluation", {}).get(key) != previous.get("evaluation", {}).get(key):
            raise ValueError(f"无法比较：评估设置 {key} 不一致")
    current_protocol, previous_protocol = dict(current_metrics["protocol"]), dict(previous_metrics["protocol"])
    if allow_reranking:
        if not current_metrics.get("reranking"):
            raise ValueError("显式重排对比要求结果中有完整reranking记录")
        current_protocol.pop("ranking", None)
        previous_protocol.pop("ranking", None)
    if current_protocol != previous_protocol:
        raise ValueError("无法比较：指标协议不一致")
    manifest = lambda records: sorted((r["key"], r["id"]) for r in records if r.get("included", True))
    if manifest(current.get("gallery", [])) != manifest(previous.get("gallery", [])):
        raise ValueError("无法比较：有效图库清单不一致")
    if manifest(current["queries"]) != manifest(previous["queries"]):
        raise ValueError("无法比较：查询清单不一致")
    before = {q["key"]: q for q in previous["queries"]}
    if len(before) != len(previous["queries"]) or len({q["key"] for q in current["queries"]}) != len(current["queries"]):
        raise ValueError("无法比较：查询 key 重复")
    counts = dict.fromkeys(("fixed", "regressed", "still_wrong", "still_correct", "missing"), 0)
    changes = []
    for query in current["queries"]:
        old = before[query["key"]]
        if not query["valid"] or not old["valid"]:
            change = "missing"
        else:
            change = {(False, True): "fixed", (True, False): "regressed",
                      (False, False): "still_wrong", (True, True): "still_correct"}[(bool(old["top1_correct"]), bool(query["top1_correct"]))]
        counts[change] += 1
        changes.append({"key": query["key"], "change": change,
                        "before_rank": old["first_positive_rank"], "after_rank": query["first_positive_rank"],
                        "before_ap": old["ap"], "after_ap": query["ap"]})
    delta = {k: current_metrics[k] - previous_metrics[k] for k in ("recall@1", "recall@5", "recall@10", "recall@1%", "mAP")}
    return {"counts": counts, "delta": delta, "queries": changes}, before


def build_report(directory, limit=0, compare=None, allow_reranking=False):
    directory = Path(directory).expanduser().resolve()
    data, metrics = load_result(directory)
    comparison, previous = None, {}
    if compare:
        compare = Path(compare).expanduser().resolve()
        old_data, old_metrics = load_result(compare)
        comparison, previous = compare_results(data, metrics, old_data, old_metrics, allow_reranking)
        comparison["baseline"] = str(compare)
    if limit < 0:
        raise ValueError("limit 必须非负；0 表示全部")
    # 无正样本和排名靠后的案例优先；所有查询仍保存在 queries.json。
    queries = sorted(data["queries"], key=lambda q: (q["valid"], -(q["first_positive_rank"] or 0)))
    queries = queries[:limit] if limit else queries
    report = directory / "report"
    images = report / "images"
    images.mkdir(parents=True, exist_ok=True)
    copied = {}

    def picture(record, source=directory):
        if record is None:
            return None
        path = record["path"]
        if path not in copied:
            name = hashlib.sha256(path.encode()).hexdigest()[:24] + ".jpg"
            cached = next((p for p in (images / name, source / "images" / name, source / "report/images" / name) if p.is_file()), None)
            if cached:
                if cached.resolve() != (images / name).resolve():
                    shutil.copy2(cached, images / name)
            elif Path(path).is_file():
                with Image.open(path) as image:
                    image = image.convert("RGB")
                    image.thumbnail((256, 256))
                    image.save(images / name, quality=80)
            else:
                raise FileNotFoundError(f"缺少图片或离线缩略图：{path}；请在服务器用 --limit 0 生成完整报告后下载")
            copied[path] = f"images/{name}"
        return {**record, "image": copied[path]}

    cases = []
    changes = {q["key"]: q["change"] for q in comparison["queries"]} if comparison else {}
    for query in queries:
        case = {**query, "image": picture(query)["image"],
                "best_positive": picture(query["best_positive"]),
                "best_negative": picture(query.get("best_negative")),
                "top_k": [picture(candidate) for candidate in query["top_k"]]}
        if comparison:
            old = previous[query["key"]]
            case.update(change=changes[query["key"]], previous={
                "ap": old["ap"], "rank": old["first_positive_rank"],
                "top1": picture(old["top_k"][0], compare) if old["top_k"] else None})
        cases.append(case)
    used_images = {Path(path).name for path in copied.values()}
    for image in images.glob("*.jpg"):
        if image.name not in used_images:
            image.unlink()
    training = directory / "train.csv"
    if training.is_file():
        with training.open(encoding="utf-8", newline="") as stream:
            training = list(csv.DictReader(stream))
    else:
        training = []
    places = {}
    for query in data["queries"]:
        place = places.setdefault(query["id"], {"queries": 0, "errors": 0})
        place["queries"] += 1
        place["errors"] += int(query["valid"] and not query["top1_correct"])
    payload = {"metrics": metrics, "evaluation": data.get("evaluation", {}), "training": training,
               "places": places, "comparison": {k: v for k, v in comparison.items() if k != "queries"} if comparison else None,
               "cases": cases, "total": len(data["queries"])}
    encoded = json.dumps(payload, ensure_ascii=False).replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    (report / "index.html").write_text(HTML.replace("__DATA__", encoded), encoding="utf-8")
    for filename in ("metrics.json", "queries.json", "evaluation.yaml", "training_config.yaml", "config.yaml", "train.csv", "source.zip", "evaluation_source.zip", "head_train.csv", "fit_config.yaml", "reranking.yaml", "mechanism_analysis.json"):
        source = directory / filename
        if source.is_file():
            shutil.copy2(source, report / filename)
        elif (report / filename).is_file():
            (report / filename).unlink()
    if comparison:
        (report / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        (report / "comparison.json").unlink(missing_ok=True)
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
<style>textarea{display:block;width:95%;min-height:50px;padding:10px;margin-top:8px;font:inherit}pre{white-space:pre-wrap;overflow-wrap:anywhere}table{border-collapse:collapse}td,th{text-align:left;padding:8px;border-bottom:1px solid #ccd8df}.change{color:#16766a}details{margin:14px 0}#annotation-status{font-size:13px}.tools{display:flex;gap:8px;flex-wrap:wrap}</style>
<header><div class="tag">CVGL / SAMPLE4GEO BASELINE</div><h1>无人机—卫星检索案例</h1><p id="context"></p></header>
<section class="metrics" id="metrics"></section>
<p id="comparison" class="change"></p>
<details><summary>实验设置与训练记录</summary><pre id="config"></pre><div id="training"></div></details>
<p id="coverage"></p><p class="muted">绿色：正确地点；红色：错误地点。分数为余弦相似度。AP 使用 S4G 梯形积分。</p>
<div class="tools"><button id="export">导出案例标注</button><label>导入标注 <input id="import" type="file" accept=".json"></label><span id="annotation-status">标注是人工判断；请导出 JSON 保存。</span></div>
<nav><label>筛选 <select id="filter"><option value="all">全部案例</option><option value="error">检索错误</option><option value="correct">检索正确</option><option value="missing">图库无正确匹配</option><option value="fixed">相比基线修好</option><option value="regressed">相比基线退步</option><option value="still_wrong">两次均错误</option></select></label><select id="place" aria-label="按地点筛选"><option value="">所有地点</option></select><select id="sort" aria-label="排序"><option value="rank">正确排名最差</option><option value="ap">AP 最低</option><option value="margin">错误候选领先最多</option></select><input id="search" aria-label="搜索地点或文件名" placeholder="搜索地点 ID 或文件名"><span id="status"></span></nav>
<main id="cases"></main><nav><button id="prev">上一页</button><span id="page"></span><button id="next">下一页</button></nav>
<footer>离线包包含缩略图、全部逐图结果、指标和实验记录。标注只描述观察，不能单凭图片证明失败原因；测试集案例用于诊断，方案选择请使用验证集。</footer>
<script id="data" type="application/json">__DATA__</script>
<script>
const data=JSON.parse(document.getElementById('data').textContent),$=id=>document.getElementById(id),size=20;
let page=0;
function node(tag,text,cls){const el=document.createElement(tag);if(text!==undefined)el.textContent=text;if(cls)el.className=cls;return el;}
const cfg=data.evaluation;
const storageKey='cvgl-notes-'+[cfg.checkpoint_sha256,cfg.dataset,cfg.direction,cfg.altitude,data.metrics.protocol.gallery_scope].join('|');
let notes=Object.create(null);try{Object.assign(notes,JSON.parse(localStorage.getItem(storageKey)||'{}'));}catch(e){}
function saveNotes(){try{localStorage.setItem(storageKey,JSON.stringify(notes));$('annotation-status').textContent='已保存到当前浏览器；请导出 JSON 作为正式记录。';}catch(e){$('annotation-status').textContent='浏览器未持久保存，请导出 JSON。';}}
$('export').onclick=()=>{const a=node('a');a.href=URL.createObjectURL(new Blob([JSON.stringify({evaluation:cfg,annotations:notes},null,2)],{type:'application/json'}));a.download='annotations.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);};
$('import').onchange=async event=>{const file=event.target.files[0];if(!file)return;try{const imported=JSON.parse(await file.text());if(!imported.annotations||typeof imported.annotations!=='object'||Array.isArray(imported.annotations))throw Error();for(const [key,value] of Object.entries(imported.annotations)){if(value&&typeof value.category==='string'&&typeof value.note==='string')notes[key]={category:value.category,note:value.note};}saveNotes();render();}catch(e){$('annotation-status').textContent='导入失败：请选择导出的标注 JSON。';}};
$('context').textContent=[cfg.dataset,cfg.direction,cfg.altitude?cfg.altitude+' m':'',cfg.checkpoint?'权重：'+cfg.checkpoint.split('/').pop():''].filter(Boolean).join(' · ');
if(data.metrics.reranking)$('context').textContent+=' · 重排 '+data.metrics.reranking.method+'；候选图片分数仍为原GAP余弦，排名经过重排';
for(const key of ['recall@1','recall@5','recall@10','recall@1%','mAP']){const el=node('div',undefined,'metric');el.append(node('span',key),node('strong',(data.metrics[key]*100).toFixed(2)+'%'));if(data.comparison)el.append(node('small','变化 '+(data.comparison.delta[key]*100).toFixed(2)+' 个百分点'));$('metrics').append(el);}
if(data.comparison){const c=data.comparison.counts;$('comparison').textContent=`相比基线：修好 ${c.fixed} 张，退步 ${c.regressed} 张，两次均错 ${c.still_wrong} 张，两次均对 ${c.still_correct} 张。`;}
$('config').textContent=JSON.stringify(cfg,null,2);
if(data.training.length){const table=node('table'),head=node('tr'),fields=['epoch','loss','lr','steps','optimizer_updates','amp_skipped_steps'];fields.forEach(k=>head.append(node('th',k)));table.append(head);data.training.forEach(row=>{const tr=node('tr');fields.forEach(k=>tr.append(node('td',row[k]??'—')));table.append(tr);});$('training').append(table);}
Object.entries(data.places).sort((a,b)=>b[1].errors-a[1].errors).forEach(([id,p])=>{const option=node('option',`${id} · 错 ${p.errors}/${p.queries}`);option.value=id;$('place').append(option);});
$('coverage').textContent=`查询 ${data.total} 张；报告收录 ${data.cases.length} 张，优先展示排名靠后和缺少正确匹配的案例。有效图库 ${data.metrics.gallery_count} 张；规则 ${data.metrics.protocol.gallery_scope}。`;
function figure(record,title,cls){const el=node('figure',undefined,cls);if(!record){el.append(node('figcaption',title+'：无'));return el;}const img=node('img');img.src=record.image;img.alt=title+' / '+record.id;img.loading='lazy';const caption=node('figcaption',title+' · 地点 '+record.id);if(record.score!==undefined)caption.append(node('div','相似度 '+record.score.toFixed(4)));caption.append(node('div',record.path.split('/').pop(),'muted'));el.append(img,caption);return el;}
function annotate(q,box){const saved=notes[q.key]||{},select=node('select'),text=node('textarea');for(const category of ['待判断','视角差异','遮挡','纹理相似','尺度差异','疑似数据问题','其他']){const opt=node('option',category);opt.value=category;select.append(opt);}select.value=saved.category||'待判断';text.value=saved.note||'';text.placeholder='记录观察、改进假设和待验证事项';const update=()=>{notes[q.key]={category:select.value,note:text.value};saveNotes();};select.onchange=text.oninput=update;box.append(select,text);}
function render(){const value=$('filter').value,search=$('search').value.toLowerCase(),place=$('place').value;const cases=data.cases.filter(q=>(value==='all'||value==='error'&&q.valid&&!q.top1_correct||value==='correct'&&q.top1_correct||value==='missing'&&!q.valid||q.change===value)&&(!place||q.id===place)&&(!search||(q.id+' '+q.path).toLowerCase().includes(search)));const sort=$('sort').value;cases.sort((a,b)=>sort==='ap'?(a.ap??-1)-(b.ap??-1):sort==='margin'?(a.score_margin??Infinity)-(b.score_margin??Infinity):(b.first_positive_rank??Infinity)-(a.first_positive_rank??Infinity));const pages=Math.max(1,Math.ceil(cases.length/size));page=Math.min(page,pages-1);$('cases').replaceChildren();for(const q of cases.slice(page*size,(page+1)*size)){const box=node('article',undefined,'case');box.append(node('h2',`地点 ${q.id} · 首个正确排名 ${q.first_positive_rank??'无'} · AP ${q.ap===null?'无':(q.ap*100).toFixed(2)+'%'}`));if(q.score_margin!==null&&q.score_margin!==undefined)box.append(node('p','最佳正确分数 − 最佳错误分数 = '+q.score_margin.toFixed(4)));const strip=node('div',undefined,'strip');strip.append(figure(q,'查询图'),figure(q.best_positive,'最佳正确匹配','ok'));if(q.best_negative)strip.append(figure(q.best_negative,'最佳错误匹配','bad'));if(q.previous){box.append(node('p',`基线正确排名 ${q.previous.rank??'无'} · AP ${q.previous.ap===null?'无':(q.previous.ap*100).toFixed(2)+'%'}`,'change'));strip.append(figure(q.previous.top1,'基线 Top 1'));}q.top_k.forEach((r,i)=>strip.append(figure(r,'Top '+(i+1),r.correct?'ok':'bad')));box.append(strip);annotate(q,box);$('cases').append(box);}if(!cases.length)$('cases').append(node('p','没有符合条件的案例。'));$('status').textContent=`匹配 ${cases.length} 个案例`;$('page').textContent=`${page+1} / ${pages}`;$('prev').disabled=page===0;$('next').disabled=page===pages-1;}
$('filter').onchange=$('place').onchange=$('sort').onchange=$('search').oninput=()=>{page=0;render();};$('prev').onclick=()=>{page--;render();};$('next').onclick=()=>{page++;render();};render();
</script></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, help="包含 metrics.json 和 queries.json 的目录")
    parser.add_argument("--limit", type=int, default=0, help="最多展示的案例数；默认 0 表示全部，完整离线包可能较大")
    parser.add_argument("--compare", help="基线评估目录，或解压后的完整离线报告目录")
    parser.add_argument("--allow-reranking-comparison", action="store_true", help="显式允许与同查询/图库/指标定义的GAP初排比较；重排方法必须完整记录")
    args = parser.parse_args()
    build_report(args.experiment, args.limit, args.compare, args.allow_reranking_comparison)


if __name__ == "__main__":
    main()
