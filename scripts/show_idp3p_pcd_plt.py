import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utils import gen_pcd
import torch
import torch.nn as nn

class PointNetPreprocessor(nn.Module):
    """PointNet Preprocessing Module"""
    def __init__(self, fps_num: int = 4, group_num: int = 32, radius: float = 0.2):
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
        
        return grouped_xyz, representative_points

def visualize_point_clouds(combined_points, representative_points, fps_num):
    """
    可视化点云分组结果
    :param combined_points: [B, fps_num, group_num+1, 3]
    :param representative_points: [B, fps_num, 3]
    :param fps_num: 中心点数量
    """
    # 转换为numpy并去除batch维度
    combined_np = combined_points[0].detach().cpu().numpy()  # [fps_num, group_num+1, 3]
    reps_np = representative_points[0].detach().cpu().numpy()  # [fps_num, 3]
    
    # 1. 绘制完整点云（单色）
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    all_points = combined_np.reshape(-1, 3)
    ax.scatter(all_points[:, 0], all_points[:, 1], all_points[:, 2], c='b', s=1)
    ax.set_title("Complete Point Cloud")
    plt.show()
    
    # 2. 绘制分组着色点云
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    colors = plt.cm.jet(np.linspace(0, 1, fps_num))  # 生成不同颜色
    
    for i in range(fps_num):
        group_points = combined_np[i]  # [group_num+1, 3]
        ax.scatter(group_points[:, 0], group_points[:, 1], group_points[:, 2], 
                   color=colors[i], s=1, label=f'Group {i}')
    
    # 绘制中心点（红色）
    ax.scatter(reps_np[:, 0], reps_np[:, 1], reps_np[:, 2], c='r', s=20, marker='*', label='Centroids')
    ax.set_title("Grouped Point Cloud with Colors")
    ax.legend()
    plt.show()
    
    # 3. 分别绘制每个组
    for i in range(fps_num):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        group_points = combined_np[i]  # [group_num+1, 3]
        
        # 绘制组成员点
        ax.scatter(group_points[1:, 0], group_points[1:, 1], group_points[1:, 2], 
                   color=colors[i], s=1, label=f'Group {i} Points')
        
        # 绘制中心点（红色）
        ax.scatter(group_points[0, 0], group_points[0, 1], group_points[0, 2], 
                   c='r', s=50, marker='*', label='Centroid')
        
        ax.set_title(f"Group {i} Point Cloud")
        ax.legend()
        plt.show()

def process_and_visualize(color_jpg_path, depth_png_path, fps_num=8, group_num=512, radius=0.2):
    """
    处理并可视化点云分组
    :param color_path: 彩色图片路径
    :param depth_path: 深度图片路径
    :param fps_num: 中心点数量
    :param group_num: 每组点数
    :param radius: 球查询半径
    """
    num_points_to_sample = 4096  # Number of points to sample from the point cloud
    gen_table = False  # Whether to generate table points
    threshold = 0.05  # Threshold for filtering points close to the plane

    # 1. 生成点云
    _,xyz = gen_pcd(color_jpg_path, depth_png_path, num_points_to_sample, gen_table, threshold)  # 假设返回的是[N,3] numpy数组
    
    # 2. 转换为tensor并添加batch维度
    xyz_tensor = torch.from_numpy(xyz).float().unsqueeze(0)  # [1, N, 3]
    
    # 3. 使用PointNetPreprocessor处理
    preprocessor = PointNetPreprocessor(fps_num=fps_num, group_num=group_num, radius=radius)
    combined_points, representative_points = preprocessor(xyz_tensor)
    
    # 4. 可视化结果
    visualize_point_clouds(combined_points, representative_points, fps_num)

# 使用示例
color_path = 'table_pcd/color_image.jpg';depth_path = 'table_pcd/depth_image.png'
process_and_visualize(color_path, depth_path, fps_num=4)