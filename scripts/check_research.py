"""新候选预检：真实特征图与大幅值梯度；GPU另测物理B24完整编码器的多步优化。"""

import argparse
import gc
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from sample4geo.auxiliary import AUXILIARIES, TrainingObjective
from sample4geo.loss import InfoNCE
from sample4geo.model import TimmModel
from sample4geo.modules import make_module
from sample4geo.research_modules import FEATURE_MODULES
from sample4geo.utils import read_config, save_json


def assert_gradients(module, inputs):
    if not all(value.grad is not None and torch.isfinite(value.grad).all() for value in inputs):
        raise RuntimeError("输入梯度非有限或缺失")
    if not all(torch.isfinite(p.grad).all() for p in module.parameters() if p.grad is not None):
        raise RuntimeError("参数梯度非有限")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default=str(ROOT / "configs/research_campaign.yaml"))
    parser.add_argument("--map-only", action="store_true", help="CPU检查不能代替GPU全模型显存验证")
    args = parser.parse_args()
    device = torch.device(args.device)
    config = read_config(args.config)
    torch.set_num_threads(2)
    batch = config["batch_size"] if device.type == "cuda" else 2
    checks = {}
    for name in [*FEATURE_MODULES, *["aux:" + a for a in AUXILIARIES[1:]]]:
        module = None
        try:
            module = (TrainingObjective(name[4:], 1024) if name.startswith("aux:") else make_module(name)[0]).to(device)
            for amplitude in (1.0, 32.0):
                module.zero_grad(set_to_none=True)
                x = (torch.randn(batch, 1024, 12, 12, device=device) * amplitude).requires_grad_()
                with torch.autocast(device.type, dtype=torch.float16 if device.type == "cuda" else torch.bfloat16, enabled=device.type == "cuda"):
                    if name.startswith("aux:"):
                        y = (torch.randn_like(x) * amplitude).requires_grad_()
                        out = module(x, y, x.mean((2, 3)), y.mean((2, 3)), torch.tensor(14.28, device=device), InfoNCE())
                    else:
                        y = None
                        out = module(x)
                if not torch.isfinite(out).all():
                    raise RuntimeError("输出非有限")
                out.float().square().mean().backward()
                assert_gradients(module, [x, y] if y is not None else [x])
            checks[name] = {"passed": True, "batch_size": batch, "amplitudes": [1, 32]}
        except Exception:
            checks[name] = {"passed": False, "traceback": traceback.format_exc()}
        print(name, checks[name]["passed"], flush=True)
        del module
    if not args.map_only:
        if device.type != "cuda":
            raise ValueError("完整B24预检需要CUDA；本地可用--map-only")
        for arm in [{"id": "baseline", "module": "none"}, *config["arms"]]:
            model, optimizer, loss, a, b, fa, fb = None, None, None, None, None, None, None
            started = time.monotonic()
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                model = TimmModel(pretrained=True, module=arm["module"], auxiliary=arm.get("auxiliary", "none")).to(device).train()
                optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
                scaler = torch.cuda.amp.GradScaler(init_scale=1024)
                for step in range(4):
                    a, b = [torch.randn(batch, 3, config["img_size"], config["img_size"], device=device) for _ in range(2)]
                    optimizer.zero_grad(set_to_none=True)
                    for group in optimizer.param_groups:
                        group["lr"] = [0., 0.0001, 0.001, 0.001][step]
                    with torch.autocast("cuda", dtype=torch.float16):
                        if arm.get("auxiliary", "none") != "none":
                            loss, _, _ = model.training_loss(a, b, InfoNCE(), arm["auxiliary_weight"])
                        else:
                            fa, fb = model(a, b)
                            loss = InfoNCE()(fa, fb, model.logit_scale.exp())
                    if not torch.isfinite(loss):
                        raise RuntimeError("全模型loss非有限")
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    assert_gradients(model, [])
                    torch.nn.utils.clip_grad_value_(model.parameters(), 100)
                    scaler.step(optimizer)
                    scaler.update()
                checks["full:" + arm["id"]] = {"passed": True, "batch_size": batch, "steps": 4,
                    "peak_gpu_mb": torch.cuda.max_memory_allocated(device) / 2**20, "seconds": time.monotonic() - started}
            except Exception:
                checks["full:" + arm["id"]] = {"passed": False, "traceback": traceback.format_exc()}
            print("full:" + arm["id"], checks["full:" + arm["id"]]["passed"], flush=True)
            del model, optimizer, loss, a, b, fa, fb
            gc.collect()
            torch.cuda.empty_cache()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_json({"passed": all(c["passed"] for c in checks.values()), "device": str(device),
               "map_only": args.map_only, "torch": str(torch.__version__), "checks": checks}, output)
    if not all(c["passed"] for c in checks.values()):
        raise SystemExit("新候选预检失败；不启动正式队列")


if __name__ == "__main__":
    main()
