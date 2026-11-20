"""独立评估并导出逐图结果；不读取训练目录，不下载预训练权重。"""

import argparse
import hashlib
import shutil
from datetime import datetime
from pathlib import Path

from sample4geo.evaluate import evaluate
from sample4geo.model import MODEL_NAME, TimmModel
from sample4geo.utils import configure, experiment_config, load_weights, save_config, save_json, setup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", help="缺省使用权重内配置；旧权重缺省使用 u1652.yaml")
    parser.add_argument("--dataset", choices=["U1652", "SUES200"])
    parser.add_argument("--output", help="评估目录；缺省使用权重所在目录")
    parser.add_argument("--direction", choices=["D2S", "S2D"])
    parser.add_argument("--altitude", type=int, choices=[150, 200, 250, 300])
    parser.add_argument("--data-root")
    parser.add_argument("--gallery-scope", choices=["all", "query_ids"])
    parser.add_argument("--device")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--overwrite", action="store_true", help="明确允许替换该目录的已有评估结果")
    for key in ("batch-size", "workers", "img-size", "top-k"):
        parser.add_argument(f"--{key}", type=int)
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    state, saved = load_weights(checkpoint)
    config = experiment_config(args, saved)
    for key in ("mean", "std", "img_size"):
        if key in saved:
            config[key] = saved[key]
    # batch-size 在此表示评估批量，不改写训练批量记录。
    batch_size, args.batch_size = args.batch_size, None
    config = configure(config, args)
    if batch_size is not None:
        if batch_size < 1:
            parser.error("batch-size 必须大于 0")
        config["eval_batch_size"] = batch_size
    device = setup(config)
    model = TimmModel(pretrained=False)
    model.load_state_dict(state, strict=True)
    del state
    data = model.data_config()
    config.setdefault("mean", list(data["mean"]))
    config.setdefault("std", list(data["std"]))
    model.to(device)
    output = Path(args.output).expanduser().resolve() if args.output else checkpoint.parent
    output.mkdir(parents=True, exist_ok=True)
    for filename in ("metrics.json", "queries.json", "evaluation.yaml"):
        if (output / filename).exists() and not args.overwrite:
            parser.error(f"{output / filename} 已存在；请换 --output 或明确使用 --overwrite")
    config.update({"model": MODEL_NAME, "checkpoint": str(checkpoint),
                   "checkpoint_sha256": file_hash(checkpoint), "checkpoint_metadata": bool(saved),
                   "evaluated_at": datetime.now().astimezone().isoformat()})
    metrics, queries = evaluate(model, config, device)
    queries["evaluation"] = config
    metrics["checkpoint_sha256"] = config["checkpoint_sha256"]
    save_json(queries, output / "queries.json")
    save_json(metrics, output / "metrics.json")
    # 原训练配置保持可追溯；独立保存本次实际评估参数。
    save_config(config, output / "evaluation.yaml")
    if saved:
        save_config(saved, output / "training_config.yaml")
    for filename in ("train.csv", "source.zip"):
        source = checkpoint.parent / filename
        if source.is_file() and source.resolve() != (output / filename).resolve():
            shutil.copy2(source, output / filename)
    if not (output / "config.yaml").exists():
        save_config(config, output / "config.yaml")
    print(" - ".join(f"{key}: {metrics[key] * 100:.2f}%" for key in ("recall@1", "recall@5", "recall@10", "recall@1%", "mAP")))
    print(f"评估结果：{output}")


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
