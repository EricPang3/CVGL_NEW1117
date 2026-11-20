# Port: device=pos.device replaces hardcoded cuda; algorithm unchanged.
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from timm.layers import trunc_normal_
from timm.models.vision_transformer import Block
class R2Former(nn.Module):
    def __init__(self, img_size=(224,224), decoder_embed_dim=384, decoder_depth=4, decoder_num_heads=6,
                 decoder_mlp_ratio=4., decoder_norm_layer=nn.LayerNorm, embed_dim=384, num_classes=1, num_patches=16, input_dim=256, num_corr=1):
        super().__init__()
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.num_patches = num_patches
        self.num_corr = num_corr
        self.img_size = img_size
        self.patch_size = [16, 16]
        # self.patch_embed.num_patches
        # MAE decoder specifics
        self.cls_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.cls_token_2 = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_token_2, std=.02)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=True)
        self.ratio = nn.Parameter(torch.ones(1) * 0.5, requires_grad=True)  # fixed sin-cos embedding
        self.blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, decoder_mlp_ratio, qkv_bias=True, norm_layer=decoder_norm_layer)
            for i in range(decoder_depth)])

        self.blocks_2 = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, decoder_mlp_ratio, qkv_bias=True, norm_layer=decoder_norm_layer)
            for i in range(2)])

        # self.local_dim = 128
        self.decoder_norm = decoder_norm_layer(decoder_embed_dim)
        # self.global_head = nn.Linear(self.input_dim, decoder_embed_dim, bias=True)
        # self.local_head = nn.Linear(self.local_dim, decoder_embed_dim, bias=True)
        self.pair_head = nn.Linear(7, decoder_embed_dim, bias=True)
        self.pair_head_2 = nn.Linear(decoder_embed_dim, decoder_embed_dim, bias=True)
        # self.pair_head = nn.Sequential(nn.Linear(7 * self.num_corr, decoder_embed_dim, bias=False),
        #                                 nn.BatchNorm1d(decoder_embed_dim),
                                        # nn.ReLU(inplace=True), # hidden layer
                                        # nn.Linear(decoder_embed_dim, decoder_embed_dim)) # output layer

        self.decoder_pred = nn.Linear(decoder_embed_dim, num_classes, bias=True)  # decoder to patch
        # --------------------------------------------------------------------------
        self.decoder_pos_embed.data[:, 1:].normal_(mean=0.0, std=0.01)
        self.initialize_weights_all()
        self.cos = nn.CosineSimilarity(dim=1)
        self.sm = torch.nn.Softmax(dim=1)

        # self.decoder_pred.weight.data.normal_(mean=0.0, std=0.01)
        # self.decoder_pred.bias.data.zero_()

    def initialize_weights_all(self):
        # initialization
        # pos_embed = get_2d_sincos_pos_embed_wh(self.decoder_pos_embed.shape[-1], (self.img_size[0]//self.patch_size[0],self.img_size[1]//self.patch_size[1]*2),
        #                                     cls_token=True)
        # pos_embed = torch.from_numpy(pos_embed).float()
        # self.decoder_pos_embed[0, :(1 + self.num_patches // 2)] = pos_embed[pos_embed.shape[0]//2-self.img_size[1]//self.patch_size[1]//2]
        # self.decoder_pos_embed[0, (1 + self.num_patches // 2):] = pos_embed[
        #     pos_embed.shape[0] // 2 + self.img_size[1] // self.patch_size[1] // 2]

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def to_map(self, x):
        p = self.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, x.shape[-1]))
        x = torch.permute(x, (0, 3, 1, 2))
        return x

    def forward(self, x_global, x_rerank=None, y_global=None, y_rerank=None):
        if x_rerank is None and y_global is None and y_rerank is None:
            x_rerank = y_rerank = torch.rand(1,500,131).cuda()
            y_global = x_global = torch.rand(1,256).cuda()
        # print(x_global.shape, x_rerank.shape)
        B = x_global.shape[0]
        N = x_rerank.shape[1]
        global_score = self.cos(x_global.detach(), y_global.detach())

        x_rerank_token = F.normalize(x_rerank[:, :, 3:], p=2, dim=2)
        y_rerank_token = F.normalize(y_rerank[:, :, 3:], p=2, dim=2)
        x_coordinate = x_rerank[:, :, :3].detach().clamp(min=0, max=1)
        y_coordinate = y_rerank[:, :, :3].detach().clamp(min=0, max=1)

        correlation = torch.matmul(x_rerank_token, y_rerank_token.permute((0, 2, 1)))
        xy_matrix = torch.cat(
            [x_coordinate.unsqueeze(2).repeat(1, 1, y_rerank_token.shape[1], 1),
             y_coordinate.unsqueeze(1).repeat(1, x_rerank_token.shape[1], 1, 1), correlation.unsqueeze(3)],
            dim=3)
        order_q = torch.argsort(correlation.unsqueeze(3), dim=2, descending=True).repeat(1, 1, 1, 7)
        order_k = torch.argsort(correlation.unsqueeze(3), dim=1, descending=True).repeat(1, 1, 1, 7)
        select_q = torch.gather(input=xy_matrix, index=order_q[:, :, :self.num_corr, :], dim=2)
        select_k = torch.gather(input=xy_matrix, index=order_k[:, :self.num_corr, :, :], dim=1)
        select_k_copy = select_k.clone()
        select_k_copy[:,:,:,:6] = torch.flip(select_k[:,:,:,:6].reshape(select_k.shape[0], select_k.shape[1],select_k.shape[2],2,3),dims=(3,)).reshape(select_k.shape[0], select_k.shape[1],select_k.shape[2],6)
        select = torch.cat([select_q, select_k.permute((0, 2, 1, 3))], dim=1)
        select_copy = torch.cat([select_q, select_k.permute((0, 2, 1, 3))], dim=1)
        N_select = select.shape[1]

        pair_matrix = self.pair_head(select.reshape(B * N_select * self.num_corr, 7)).reshape(B * N_select, self.num_corr, self.decoder_embed_dim)
        pair_matrix += get_2d_sincos_pos_embed_from_grid(self.decoder_embed_dim, select_copy.reshape(B * N_select, self.num_corr, 7)[:,:,3:5])
        x = torch.cat([self.cls_token_2.repeat(B*N_select, 1, 1), pair_matrix], dim=1)
        for blk in self.blocks_2:
            x = blk(x)
        x = self.decoder_norm(x)

        x = self.pair_head_2(x[:,0,:].reshape(B*N_select, self.decoder_embed_dim)).reshape(B, N_select, self.decoder_embed_dim)
        x = x.reshape(B, N_select, self.decoder_embed_dim) + get_2d_sincos_pos_embed_from_grid(self.decoder_embed_dim, select_copy[:,:,0,0:2])
        x = torch.cat([self.cls_token.repeat(B, 1, 1), x], dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        if self.num_classes == 1:
            local_score = self.decoder_pred(x[:, 0]).reshape(-1)
            local_score = torch.sigmoid(local_score)
            final_score = global_score.detach() * self.ratio.clamp(min=0.1, max=0.9) + local_score.detach() * (
                    1 - self.ratio.clamp(min=0.1, max=0.9))
        elif self.num_classes == 2:
            local_score = self.decoder_pred(x[:, 0])
            final_score = global_score.detach() * self.ratio.clamp(min=0.1, max=0.9) + self.sm(local_score).detach()[:,1] * (
                    1 - self.ratio.clamp(min=0.1, max=0.9))
        return local_score, final_score

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[:,:, 0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[:,:, 1])  # (H*W, D/2)

    emb = torch.cat([emb_h, emb_w], dim=2) # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    # pos = pos.reshape(-1)  # (M,)
    out = torch.einsum('bm,d->bmd', pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out) # (M, D/2)
    emb_cos = torch.cos(out) # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=2)  # (M, D)
    return emb
