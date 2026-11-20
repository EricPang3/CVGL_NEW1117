"""以真实末级尺寸检查所有模块的有限值、反向传播、AMP与关闭分支等价。"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from sample4geo.model import TimmModel, load_model_state
from sample4geo.modules import make_module
from sample4geo.utils import read_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.set_num_threads(2)
    results = {"torch": str(torch.__version__), "device": str(device), "checks": {}}
    for entry in read_config(ROOT / "configs/module_suite.yaml")["modules"]:
        if entry["kind"] == "rerank":
            continue
        started = time.monotonic()
        name = entry["id"]
        try:
            module, mode, dimension = make_module(name)
            module.to(device).train()
            batch = 24 if device.type == "cuda" else 2
            inputs = torch.randn(batch, 1024, 12, 12, device=device, requires_grad=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                output = module(inputs)
                output = output[0] if isinstance(output, tuple) else output
            expected = (batch, dimension) if mode == "parallel" else (batch, 1024, 12, 12)
            assert tuple(output.shape) == expected, (output.shape, expected)
            assert torch.isfinite(output).all(), "nonfinite output"
            loss = (output.float() * torch.randn_like(output.float())).mean()
            loss.backward()
            assert inputs.grad is not None and torch.isfinite(inputs.grad).all(), "invalid input gradient"
            assert all(torch.isfinite(p.grad).all() for p in module.parameters() if p.grad is not None), "nonfinite parameter gradient"
            results["checks"][name] = {"passed": True, "output_shape": list(output.shape), "mode": mode,
                                       "parameters": sum(p.numel() for p in module.parameters()), "seconds": time.monotonic()-started}
        except Exception:
            results["checks"][name] = {"passed": False, "traceback": traceback.format_exc()}
        print(name, results["checks"][name]["passed"], flush=True)
        if device.type == "cuda":
            del module, inputs
            torch.cuda.empty_cache()
    try:
        reference = TimmModel().to(device).eval()
        images = torch.randn(1, 3, 384, 384, device=device)
        with torch.no_grad():
            expected = reference.model(images)
            assert torch.equal(reference(images), expected), "none changed original encoder"
            features = reference.model.forward_features(images)
            assert tuple(features.shape) == (1, 1024, 12, 12)
        for name in ("eca", "wtconv", "boq", "burst_gap", "grn", "salad", "gem"):
            with patch("sample4geo.model.timm.create_model", return_value=reference.model):
                candidate = TimmModel(module=name, module_alpha=0).to(device).eval()
            load_model_state(candidate, reference.state_dict(), initialize_module=True)
            with torch.no_grad():
                assert torch.equal(candidate.encode_map(features), expected), f"{name} zero branch changed descriptor"
        results["checks"]["baseline_equivalence"] = {"passed": True, "feature_shape": list(features.shape)}
    except Exception:
        results["checks"]["baseline_equivalence"] = {"passed": False, "traceback": traceback.format_exc()}
    results["passed"] = all(item["passed"] for item in results["checks"].values())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    if not results["passed"]:
        raise SystemExit("模块预检失败，停止正式矩阵；详情见检查JSON")


if __name__ == "__main__":
    main()
