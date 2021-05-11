from pointnet_util import index_points
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GlobalAttention(nn.Module):
    def __init__(self, d_model, g) -> None:
        """Global attention"""
        super().__init__()

        self.g_qs = nn.Linear(d_model, d_model, bias=False)
        self.g_ks = nn.Linear(d_model, d_model, bias=False)
        self.g_vs = nn.Linear(d_model, d_model, bias=False)

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

        self.g = g

    def forward(self, xyz, x, dist_arg_sort):
        """Attention on first g points of every layer."""
        g_idx, _ = torch.sort(dist_arg_sort[:, :self.g])
        g_xyz = index_points(xyz, g_idx)
        gq, gk, gv = self.g_qs(x[:, :self.g]), index_points(self.g_ks(x), g_idx), index_points(self.g_vs(x), g_idx)
        g_pos_enc = self.g_delta(xyz[:, :self.g, None] - g_xyz)
        g_attn = self.g_gamma(gq[:, :self.g, None] - gk + g_pos_enc)
        g_attn = F.softmax(g_attn / np.sqrt(gk.size(-1)), dim=-2)
        g_res = torch.einsum('bmnf,bmnf->bmf', g_attn, gv + g_pos_enc)
        return g_res
