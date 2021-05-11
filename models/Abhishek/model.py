import torch
import torch.nn as nn
from pointnet_util import PointNetSetAbstraction
from models.Abhishek.transformer import TransformerBlock


class TransitionDown(nn.Module):
    def __init__(self, k, g, nneighbor, channels) -> None:
        super().__init__()
        self.sa = PointNetSetAbstraction(k, 0, nneighbor, channels[0], channels[1:], group_all=False, knn=True)

        # Separate MLP for classification token
        self.fc_cls = nn.Sequential(
            nn.Linear(channels[0]-3, channels[1]),
            nn.ReLU(),
            nn.Linear(channels[1], channels[2])
        )
        self.g = g

    def forward(self, xyz, points):
        # Forward pass handles global fixed points seperately
        cls_xyz, cls_points = xyz[:, :self.g], points[:, :self.g]
        xyz, points = self.sa(xyz, points)
        cls_points = self.fc_cls(cls_points)
        xyz = torch.cat((xyz, cls_xyz), dim=1)
        points = torch.cat((points, cls_points), dim=1)
        return xyz, points


class PointTransformer(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        npoints, nblocks, nneighbor, n_c, d_points = cfg.num_point, cfg.model.nblocks, \
                                                     cfg.model.nneighbor, cfg.num_class, cfg.input_dim

        n_global_points = cfg.model.n_global_points
        share_params = cfg.model.share_params

        self.fc1 = nn.Sequential(
            nn.Linear(d_points, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        self.transformer1 = TransformerBlock(d_points=32, d_model=cfg.model.transformer_dim, k=nneighbor,
                                             g=n_global_points, global_attn=False, share_params=False)
        self.transition_downs = nn.ModuleList()
        self.transformers = nn.ModuleList()
        self.global_attns = []
        for i in range(nblocks):
            channel = 32 * 2 ** (i + 1)
            self.transition_downs.append(TransitionDown(npoints // 4 ** (i + 1), n_global_points, nneighbor,
                                                        [channel // 2 + 3, channel, channel]))
            self.transformers.append(TransformerBlock(d_points=channel, d_model=cfg.model.transformer_dim, k=nneighbor,
                                                      g=n_global_points, global_attn=True, share_params=share_params))

        self.fc2 = nn.Sequential(
            nn.Linear(32 * 2 ** nblocks, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, n_c)
        )
        self.nblocks = nblocks

    def forward(self, x):
        xyz = x[..., :3]
        points = self.transformer1(xyz, self.fc1(x))
        for i in range(self.nblocks):
            xyz, points = self.transition_downs[i](xyz, points)
            points = self.transformers[i](xyz, points)
        res = self.fc2(points.mean(1))
        return res