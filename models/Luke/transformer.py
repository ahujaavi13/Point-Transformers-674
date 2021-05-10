"""
Author: Luke
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet_util import index_points, square_distance



class TransformerBlock(nn.Module):
    def __init__(self, points_dim, model_dim, k, dropout, device):
        super().__init__()
        self.k = k
        self.dropout = nn.Dropout(dropout)
        self.device = device
        # Dropout embeddings mlp
        self.dpt_mlp1 = nn.Linear(model_dim, model_dim)
        self.dpt_mlp2 = nn.Linear(model_dim, model_dim)
        # Phi layer
        self.phi_fc = nn.Linear(model_dim, model_dim, bias=False)
        # Psi layer
        self.psi_fc = nn.Linear(model_dim, model_dim, bias=False)
        # Alpha layer
        self.alpha_fc = nn.Linear(model_dim, model_dim, bias=False)
        # Gamma layers
        self.gam_mlp1 = nn.Linear(model_dim, model_dim)
        self.gam_mlp2 = nn.Linear(model_dim, model_dim)
        # Delta layers
        self.del_mlp1 = nn.Linear(3, model_dim)
        self.del_mlp2 = nn.Linear(model_dim, model_dim)
        # Linear layers
        self.fc1 = nn.Linear(points_dim, model_dim)
        self.fc2 = nn.Linear(model_dim, points_dim)

    def forward(self, x, in_f):
        points_diff = square_distance(x, x)
        knn_idx = points_diff.argsort()[:, :, :self.k]
        points_knn = index_points(x, knn_idx)

        f = self.fc1(in_f)
        #print(points_knn.shape)
        true_k = points_knn.shape[2]
        delta = self.del_mlp2(F.relu(self.del_mlp1(x.repeat((1, 1, true_k)).reshape(points_knn.shape) - points_knn)))
        psi = index_points(self.psi_fc(f), knn_idx)
        phi = self.phi_fc(f).repeat((1, 1, true_k)).reshape(delta.shape)
        alpha = index_points(self.alpha_fc(f), knn_idx)

        dropped_mlp = self.dropout(self.dpt_mlp2(F.relu(self.dpt_mlp1(f))).repeat((1, 1, true_k)).reshape(delta.shape))
        #dropout_refactored = dropped_mlp*self.dropout(torch.ones(f.shape[1]).unsqueeze(1).unsqueeze(0).to(self.device))

        
        gamma = self.gam_mlp2(F.relu(self.gam_mlp1(phi - psi + delta + dropped_mlp)))
        rho = F.softmax(gamma / (true_k ** 0.5), dim=2)
        y = torch.sum(rho * (alpha  + delta), dim=2)
        out_f = self.fc2(y)
        return out_f + in_f
        

        

        
