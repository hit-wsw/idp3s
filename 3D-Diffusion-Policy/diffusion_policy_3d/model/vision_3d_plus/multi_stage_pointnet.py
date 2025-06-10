import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from termcolor import cprint

def meanpool(x, dim=-1, keepdim=False):
    out = x.mean(dim=dim, keepdim=keepdim)
    return out

def maxpool(x, dim=-1, keepdim=False):
    out = x.max(dim=dim, keepdim=keepdim).values
    return out

class PointNetPreprocessor(nn.Module):
    """PointNet Preprocessing Module"""
    def __init__(self, fps_num: int = 512, group_num: int = 32, radius: float = 0.2):
        super().__init__()
        self.fps_num = fps_num
        self.group_num = group_num
        self.radius = radius
        
    def farthest_point_sampling(self, xyz: torch.Tensor) -> torch.Tensor:
        B, N, _ = xyz.shape
        device = xyz.device
        
        centroids = torch.zeros((B, self.fps_num), dtype=torch.long, device=device)
        distance = torch.ones((B, N), device=device) * 1e10
        
        # Random start index for each batch
        farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
        
        for i in range(self.fps_num):
            centroids[:, i] = farthest
            centroid = xyz[torch.arange(B), farthest, :].view(B, 1, 3)
            dist = torch.sum((xyz - centroid) ** 2, dim=-1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, dim=1)[1]
            
        return centroids

    def ball_query(self, xyz: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
        """
        Vectorized ball query. Input:
            xyz: [B, N, 3]
            centroids: [B, fps_num]
        Output:
            group_idx: [B, fps_num, group_num]
        """
        B, N, _ = xyz.shape
        device = xyz.device

        # [B, fps_num, 3]
        centroids_xyz = torch.gather(xyz, 1, centroids.unsqueeze(-1).expand(-1, -1, 3))

        # [B, fps_num, N]: 距离平方
        dist = torch.sum((xyz.unsqueeze(1) - centroids_xyz.unsqueeze(2)) ** 2, dim=-1)

        # [B, fps_num, N]: 把超过半径平方的设为大数
        mask = dist > self.radius ** 2
        dist[mask] = 1e10

        # 为了确保有 group_num 个点，即使不足也补足最近点
        idx = dist.argsort(dim=-1)[:, :, :self.group_num]  # [B, fps_num, group_num]

        return idx


    def forward(self, xyz: torch.Tensor):
        """Input: [B, N, 3], Output: combined_points [B, fps_num, group_num+1, 3], representative_points [B, fps_num, 3]"""
        centroids = self.farthest_point_sampling(xyz)  # [B, fps_num]
        group_idx = self.ball_query(xyz, centroids)    # [B, fps_num, group_num]
        
        # Get grouped points
        B = xyz.shape[0]
        grouped_xyz = torch.gather(
            xyz.unsqueeze(1).expand(-1, self.fps_num, -1, -1),
            2,
            group_idx.unsqueeze(-1).expand(-1, -1, -1, 3)
        )  # [B, fps_num, group_num, 3]
        
        # Get centroids
        centroids_xyz = torch.gather(xyz, 1, centroids.unsqueeze(-1).expand(-1, -1, 3))  # [B, fps_num, 3]
        
        # Compute relative coordinates
        relative_xyz = grouped_xyz - centroids_xyz.unsqueeze(2)
        
        # Combine centroids and relative coordinates
        combined_points = torch.cat([
            centroids_xyz.unsqueeze(2),  # [B, fps_num, 1, 3]
            relative_xyz
        ], dim=2)  # [B, fps_num, group_num+1, 3]
        
        # Representative points (centroids)
        representative_points = centroids_xyz  # [B, fps_num, 3]
        
        return combined_points, representative_points

class MultiStagePointNetEncoder(nn.Module):
    def __init__(self, h_dim=128, out_channels=128, num_layers=4, **kwargs):
        super().__init__()

        fps_num = 8;group_num = 512;radius = 0.2
        self.h_dim = h_dim
        self.out_channels = out_channels
        self.num_layers = num_layers

        in_channels = 3; block_channel = [64, 128, 256]; use_layernorm = True
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
        )
        
        self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], self.out_channels),
                nn.LayerNorm(self.out_channels)
            )
        
        self.preprocessor = PointNetPreprocessor(fps_num=fps_num, group_num=group_num, radius=radius)

        self.act = nn.LeakyReLU(negative_slope=0.0, inplace=False)

        self.conv_in = nn.Conv1d(3, h_dim, kernel_size=1)
        self.layers, self.global_layers = nn.ModuleList(), nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(nn.Conv1d(h_dim, h_dim, kernel_size=1))
            self.global_layers.append(nn.Conv1d(h_dim * 2, h_dim, kernel_size=1))
        self.conv_out = nn.Conv1d(h_dim * self.num_layers, out_channels, kernel_size=1)

    def forward(self, x):
    # 1. 预处理并释放原始点云
        grouped_points, center_x = self.preprocessor(x)  # [B, K, N, 3]
        del x
        # 2. 向量化处理所有组
        B, K, N, _ = grouped_points.shape
        grouped_points = grouped_points.reshape(B * K, N, 3).transpose(1, 2)  # [B*K, 3, N]
        
        # 3. 批量卷积处理
        y = self.act(self.conv_in(grouped_points))  # [B*K, h_dim, N]
        feat_list = []
        for i in range(self.num_layers):
            y = self.act(self.layers[i](y))
            y_global = y.max(-1, keepdim=True).values
            y = torch.cat([y, y_global.expand_as(y)], dim=1)
            y = self.act(self.global_layers[i](y))
            feat_list.append(y)
        
        # 4. 合并特征并聚合
        y = torch.cat(feat_list, dim=1)
        y = self.conv_out(y).max(-1).values  # [B*K, out_channels]
        y_global = y.reshape(B, K, -1).max(1).values  # [B, out_channels]

        # 5. 处理中心点特征
        center_x = self.mlp(center_x).max(1)[0]
        center_x_feature = self.final_projection(center_x)

        return y_global, center_x_feature

        


   