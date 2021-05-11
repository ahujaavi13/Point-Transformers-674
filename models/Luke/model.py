"""
Author: Luke
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.Luke.transformer import TransformerBlock
from pointnet_util import PointNetSetAbstraction


class PointTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        n_points, k, n_c, d_points, dropout = cfg.num_point, cfg.model.nneighbor, cfg.num_class, cfg.input_dim, cfg.model.dropout
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # initial MLP
        self.fc1 = nn.Linear(d_points, 32)
        self.fc2 = nn.Linear(32, 32)
        # point transformers
        self.pt1 = TransformerBlock(32, cfg.model.transformer_dim, k, dropout, device)
        self.td1 = PointNetSetAbstraction(n_points // 4, 0, k, 35, [64, 64], group_all=False, knn=True)
        self.pt2 = TransformerBlock(64, cfg.model.transformer_dim, k, dropout, device)
        self.td2 = PointNetSetAbstraction(n_points // 16, 0, k, 67, [128, 128], group_all=False, knn=True)
        self.pt3 = TransformerBlock(128, cfg.model.transformer_dim, k, dropout, device)
        self.td3 = PointNetSetAbstraction(n_points // 64, 0, k, 131, [256, 256], group_all=False, knn=True)
        self.pt4 = TransformerBlock(256, cfg.model.transformer_dim, k, dropout, device)
        self.td4 = PointNetSetAbstraction(n_points // 256, 0, k, 259, [512, 512], group_all=False, knn=True)
        self.pt5 = TransformerBlock(512, cfg.model.transformer_dim, k, dropout, device)
        # final MLP
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 64)
        self.fc5 = nn.Linear(64, n_c)
                

    def forward(self, x_in):
        x = x_in[..., :3]
        l1 = self.pt1(x, self.fc2(F.relu(self.fc1(x_in))))
        x, pts = self.td1(x, l1)
        l2 = self.pt2(x, pts)
        x, pts = self.td2(x, l2)
        l3 = self.pt3(x, pts)
        x, pts = self.td3(x, l3)
        l4 = self.pt4(x, pts)
        x, pts = self.td4(x, l4)
        l5 = self.pt5(x, pts)
        return self.fc5(F.relu(self.fc4(F.relu(self.fc3(l5.mean(dim=1))))))
