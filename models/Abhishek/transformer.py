from pointnet_util import index_points, square_distance
from models.Abhishek.global_attention import GlobalAttention
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TransformerBlock(nn.Module):
    def __init__(self, d_points, d_model, k, g=1, global_attn=True, share_params=False) -> None:
        super().__init__()

        self.fc1 = nn.Linear(d_points, d_model)
        self.fc2 = nn.Linear(d_model, d_points)
        self.fc_delta = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.fc_gamma = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.k = k

        self.global_attn = global_attn
        self.share_params = share_params
        self.g = g
        if self.global_attn and share_params:
            self.g_delta = nn.Sequential(
                nn.Linear(3, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model)
            )
            self.g_gamma = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model)
            )
        elif self.global_attn:
            self.g_attn = GlobalAttention(d_model=d_model, g=g)

    def forward(self, xyz, features):
        dists = square_distance(xyz, xyz)
        dist_arg_sort = dists.argsort()
        knn_idx = dist_arg_sort[:, :, :self.k+1]
        if self.global_attn:
            knn_idx[:, :, -1] = 0
            knn_idx = torch.roll(knn_idx, 1, -1)
        knn_xyz = index_points(xyz, knn_idx)

        pre = features
        x = self.fc1(features)

        k_proj, v_proj = self.w_ks(x), self.w_vs(x)
        q, k, v = self.w_qs(x), index_points(k_proj, knn_idx), index_points(v_proj, knn_idx)
        pos_enc = self.fc_delta(xyz[:, :, None] - knn_xyz)
        attn = self.fc_gamma(q[:, :, None] - k + pos_enc)

        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)
        res = torch.einsum('bmnf,bmnf->bmf', attn, v + pos_enc)
        if self.global_attn and self.share_params:
            g_idx = torch.sort(dist_arg_sort[:, :self.g])[0]
            g_xyz = index_points(xyz, g_idx)
            gq, gk, gv = q[:, :self.g], index_points(k_proj, g_idx), index_points(v_proj, g_idx)
            pos_enc = self.fc_delta(xyz[:, :self.g, None] - g_xyz)
            g_attn = self.fc_gamma(gq[:, :self.g, None] - gk + pos_enc)
            g_attn = F.softmax(g_attn / np.sqrt(gk.size(-1)), dim=-2)
            g_res = torch.einsum('bmnf,bmnf->bmf', g_attn, gv + pos_enc)
            res[:, :self.g] = g_res
        elif self.global_attn:
            g_res = self.g_attn(xyz=xyz, x=x, dist_arg_sort=dist_arg_sort)
            res[:, :self.g] = g_res

        res = self.fc2(res) + pre
        return res
