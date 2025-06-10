import open3d as o3d
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from utils import gen_pcd
import torch
import torch.nn as nn

class PointNetPreprocessor(nn.Module):
    """PointNet Preprocessing Module"""
    def __init__(self, fps_num: int = 4, group_num: int = 512, radius: float = 0.2):
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
    

class PointNetVisualizer:
    def __init__(self, fps_num=8, group_num=512, radius=0.2):
        self.preprocessor = PointNetPreprocessor(fps_num, group_num, radius)
        self.num_points_to_sample = 4096  # Number of points to sample from the point cloud
        self.gen_table = True  # Whether to generate table points
        self.threshold = 0.05  # Threshold for filtering points close to the plane
        
    def visualize(self, color_jpg_path, depth_png_path):
        # Generate point cloud from color and depth images
        _,xyz = gen_pcd(color_jpg_path, depth_png_path, self.num_points_to_sample, self.gen_table, self.threshold)  # Assuming this returns [N, 3]
        
        # Convert to tensor and add batch dimension
        xyz_tensor = torch.from_numpy(xyz).float().unsqueeze(0)  # [1, N, 3]
        
        # Process through PointNetPreprocessor
        combined_points, representative_points = self.preprocessor(xyz_tensor)
        
        # Convert to numpy
        combined_points_np = combined_points.squeeze(0).numpy()  # [fps_num, group_num+1, 3]
        rep_points_np = representative_points.squeeze(0).numpy()  # [fps_num, 3]
        
        # 1. Plot full point cloud
        pcd_full = o3d.geometry.PointCloud()
        pcd_full.points = o3d.utility.Vector3dVector(xyz)
        o3d.visualization.draw_geometries([pcd_full], window_name="Full Point Cloud")
        
        # 2. Plot grouped point cloud with different colors
        self._plot_grouped_point_cloud(combined_points_np, rep_points_np)
        
        # 3. Plot each group separately
        self._plot_individual_groups(combined_points_np, rep_points_np)
    
    def _plot_grouped_point_cloud(self, combined_points, rep_points):
        """Plot the full point cloud with different colors for each group"""
        # Create a colormap with enough distinct colors
        cmap = plt.get_cmap('tab20')
        colors = cmap(np.linspace(0, 1, len(combined_points)))
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        
        # Combine all points
        all_points = []
        all_colors = []
        
        for i, group in enumerate(combined_points):
            # First point is the centroid, rest are group points
            group_points = group[1:]  # [group_num, 3]
            group_color = np.tile(colors[i][:3], (len(group_points), 1))
            
            all_points.append(group_points)
            all_colors.append(group_color)
            
            # Add centroid with same color but darker
            centroid = group[0:1]  # [1, 3]
            centroid_color = np.array([colors[i][:3] * 0.7])  # Darker version
            
            all_points.append(centroid)
            all_colors.append(centroid_color)
        
        # Combine all points and colors
        all_points = np.vstack(all_points)
        all_colors = np.vstack(all_colors)
        
        pcd.points = o3d.utility.Vector3dVector(all_points)
        pcd.colors = o3d.utility.Vector3dVector(all_colors)
        
        o3d.visualization.draw_geometries([pcd], window_name="Grouped Point Cloud")
    
    def _plot_individual_groups(self, combined_points, rep_points):
        """Plot each group in a separate window with centroids in red"""
        for i, group in enumerate(combined_points):
            # Create point cloud for this group
            pcd = o3d.geometry.PointCloud()
            
            # Get points (excluding centroid)
            group_points = group[1:]  # [group_num, 3]
            
            # Create colors - make group points blue, centroid red
            group_colors = np.tile([0, 0, 1], (len(group_points), 1))  # Blue
            centroid_color = np.array([[1, 0, 0]])  # Red
            
            # Combine points and colors
            points = np.vstack([group[0:1], group_points])  # Centroid first
            colors = np.vstack([centroid_color, group_colors])
            
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            
            o3d.visualization.draw_geometries(
                [pcd], 
                window_name=f"Group {i+1} (Size: {len(group_points)})"
            )

# Example usage:
if __name__ == "__main__":
    visualizer = PointNetVisualizer(fps_num=8, group_num=512, radius=0.2)
    color_path = 'table_pcd/000000_color_0.png';depth_path = 'table_pcd/000000_depth_0.png'
    #color_path = 'table_pcd/color_image.jpg';depth_path = 'table_pcd/depth_image.png'
    visualizer.visualize(color_path, depth_path)