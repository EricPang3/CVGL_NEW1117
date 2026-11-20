"""基线全局初排之外的独立 Top-K 方法；不访问其他查询，不用测试标签拟合。"""

from collections import OrderedDict
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..dataset import U1652DatasetEval, get_transforms
from ..module_adapters import component


def project_map(feature, pca):
    flat = feature.float().flatten(2).transpose(1, 2)
    values = (flat - pca["mean"].to(feature.device)) @ pca["components"].to(feature.device).T
    return values.transpose(1, 2).reshape(len(feature), -1, *feature.shape[-2:])


def salient_tokens(feature, reduced, count=16):
    b, c, h, w = reduced.shape
    saliency = feature.float().square().sum(1).sqrt().flatten(1)
    saliency = saliency / saliency.amax(1, keepdim=True).clamp_min(1e-8)
    count = min(count, h * w)
    indices = saliency.topk(count, dim=1).indices
    y, x = torch.meshgrid(torch.linspace(0, 1, h, device=feature.device),
                          torch.linspace(0, 1, w, device=feature.device), indexing="ij")
    coords = torch.stack((x.flatten(), y.flatten()), dim=-1).expand(b, -1, -1)
    tokens = torch.cat((coords, saliency.unsqueeze(-1), reduced.flatten(2).transpose(1, 2)), dim=-1)
    return tokens.gather(1, indices.unsqueeze(-1).expand(-1, -1, c + 3))


@torch.no_grad()
def extract(model, records, config, device, pca=None, token_mode=None, raw_sample=False):
    loader = DataLoader(U1652DatasetEval(records, get_transforms(config)), batch_size=config["eval_batch_size"],
                        num_workers=config["workers"], shuffle=False, pin_memory=device.type == "cuda")
    global_features, local_features = [], []
    for images in tqdm(loader, desc="Rerank feature extraction"):
        context = torch.autocast("cuda", dtype=torch.float16) if config["amp"] and device.type == "cuda" else nullcontext()
        with context:
            feature = model.model.forward_features(images.to(device))
            descriptor = model.model.forward_head(feature)
        global_features.append(F.normalize(descriptor.float(), dim=-1).cpu())
        if raw_sample:
            # 固定均匀位置，只为训练地点拟合PCA；最多每图64个token。
            tokens = feature.float().flatten(2).transpose(1, 2)
            selected = torch.linspace(0, tokens.shape[1] - 1, min(64, tokens.shape[1]), device=device).long()
            local_features.append(tokens[:, selected].reshape(-1, feature.shape[1]).cpu())
        elif pca is not None:
            reduced = project_map(feature, pca)
            local = salient_tokens(feature, reduced) if token_mode == "r2former_rerank" else reduced
            local_features.append(local.half().cpu())
    return torch.cat(global_features), torch.cat(local_features) if local_features else None


def make_r2former():
    return component("r2former_rerank", "R2Former")(decoder_embed_dim=128, decoder_depth=1,
        decoder_num_heads=4, decoder_mlp_ratio=4, num_patches=16, input_dim=128, num_classes=1, num_corr=1)


def make_patch():
    return component("patch_netvlad", "PatchNetVLAD")(num_clusters=8, dim=128, use_faiss=False,
                                                     patch_sizes="2,3,4", strides="2,2,2")


class Reranker:
    def __init__(self, method, fitted, qglobal, gglobal, qlocal, glocal, device, candidates=20):
        self.method, self.device = method, device
        self.qglobal, self.gglobal = qglobal, gglobal.to(device)
        self.qlocal, self.glocal = qlocal, glocal
        self.candidates = min(candidates, len(gglobal))
        self.cache = OrderedDict()
        if method == "r2former_rerank":
            self.model = make_r2former().to(device).eval()
            self.model.load_state_dict(fitted["model"], strict=True)
        elif method == "patch_netvlad":
            self.model = make_patch().to(device).eval()
            self.model.load_state_dict(fitted["model"], strict=True)
        elif method == "superglobal_rerank":
            self.mda = component("mda", "MDescAug")(M=self.candidates, K=min(9, self.candidates - 1), beta=0.15)
            self.rerank = component("superglobal", "RerankwMDA")(M=self.candidates, K=min(9, self.candidates - 1), beta=0.15)
        else:
            raise ValueError(method)

    @torch.no_grad()
    def rank(self, start, scores):
        initial = np.argsort(-scores, axis=1, kind="stable")
        if self.method == "superglobal_rerank":
            ranks = torch.from_numpy(initial.T.copy()).to(self.device)
            refined = self.mda(self.gglobal, self.qglobal[start:start + len(scores)].to(self.device), ranks)
            return self.rerank(ranks, *refined).T.cpu().numpy(), None
        if self.method == "r2former_rerank":
            selected = torch.from_numpy(initial[:, :self.candidates].copy())
            qindices = torch.arange(start, start + len(scores)).repeat_interleave(self.candidates)
            gindices = selected.flatten()
            outputs = []
            for offset in range(0, len(qindices), 128):
                qi, gi = qindices[offset:offset + 128], gindices[offset:offset + 128]
                _, final = self.model(self.qglobal[qi].to(self.device), self.qlocal[qi].float().to(self.device),
                                      self.gglobal[gi.to(self.device)], self.glocal[gi].float().to(self.device))
                outputs.append(final.float().cpu())
            values = torch.cat(outputs).reshape(len(scores), self.candidates).numpy()
            for i in range(len(scores)):
                initial[i, :self.candidates] = initial[i, :self.candidates][np.argsort(-values[i], kind="stable")]
            return initial, values
        for i in range(len(scores)):
            candidates = initial[i, :self.candidates].copy()
            qpatch = self.model(self.qlocal[start + i:start + i + 1].float().to(self.device))[0]
            values = []
            for index in candidates:
                index = int(index)
                if index not in self.cache:
                    patches = self.model(self.glocal[index:index + 1].float().to(self.device))[0]
                    self.cache[index] = [p.half().cpu() for p in patches]
                    if len(self.cache) > 1024:
                        self.cache.popitem(last=False)
                else:
                    self.cache.move_to_end(index)
                local_score = []
                for q, reference in zip(qpatch, self.cache[index]):
                    correlation = q[0].T @ reference[0].float().to(self.device)
                    # 多尺度局部验证适配：互为最近邻的patch分数；没有声称复现原VGG/RANSAC管线。
                    nearest = correlation.argmax(1)
                    reverse = correlation.argmax(0)
                    mutual = reverse[nearest] == torch.arange(len(nearest), device=self.device)
                    best = correlation[torch.arange(len(nearest), device=self.device), nearest]
                    local_score.append(best[mutual].mean() if mutual.any() else best.mean())
                values.append(0.5 * scores[i, index] + 0.5 * float(torch.stack(local_score).mean()))
            initial[i, :self.candidates] = candidates[np.argsort(-np.asarray(values), kind="stable")]
        return initial, None
