"""检查已训练ACB/RepVGG/UniRepLK适配层融合等价与耗时，不改写原权重。"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from sample4geo.modules import make_module
from sample4geo.utils import load_weights


def fused_copy(module):
    module = copy.deepcopy(module).eval()
    for child in list(module.modules()):
        if hasattr(child, "switch_to_deploy"):
            child.switch_to_deploy()
        elif hasattr(child, "merge_dilated_branches"):
            child.merge_dilated_branches()
    return module


@torch.no_grad()
def compare(module, features, repeats=20):
    module = module.eval()
    fused = fused_copy(module)
    before, after = module(features), fused(features)
    maximum = (before - after).abs().max().item()
    if not torch.allclose(before, after, atol=1e-4, rtol=1e-4):
        raise RuntimeError(f"融合输出不一致：max_abs={maximum}")
    durations = []
    for candidate in (module, fused):
        for _ in range(5):
            candidate(features)
        if features.is_cuda:
            torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(repeats):
            candidate(features)
        if features.is_cuda:
            torch.cuda.synchronize()
        durations.append((time.perf_counter()-started)/repeats)
    return {"max_absolute_error":maximum, "original_seconds_per_batch":durations[0],
            "fused_seconds_per_batch":durations[1], "shape":list(features.shape),
            "device":str(features.device), "scope":"adapter only; random feature maps; original checkpoint unchanged"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    state, config = load_weights(args.checkpoint)
    name = config.get("module")
    if name not in ("acb", "repvgg_block", "uni_replk"):
        parser.error("仅用于ACB/RepVGG/UniRepLK模块")
    module, _, _ = make_module(name)
    module.load_state_dict({k.removeprefix("feature_module."):v for k,v in state.items() if k.startswith("feature_module.")},strict=True)
    del state
    device = torch.device(args.device)
    torch.set_num_threads(2)
    result = compare(module.to(device), torch.randn(24,1024,12,12,device=device))
    Path(args.output).write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
