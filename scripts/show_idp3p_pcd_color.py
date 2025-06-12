import open3d as o3d
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from utils import gen_pcd
import torch
import torch.nn as nn

class PointNetPreprocessor(nn.Module):
    """PointNet Preprocessing Module with color support"""
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

    def forward(self, xyz: torch.Tensor, colors: torch.Tensor = None):
        """
        Input: 
            xyz: [B, N, 3] - point coordinates
            colors: [B, N, 3] - RGB colors (optional)
        Output: 
            grouped_xyz: [B, fps_num, group_num, 3]
            grouped_colors: [B, fps_num, group_num, 3] (if colors provided)
            representative_points: [B, fps_num, 3]
        """
        centroids = self.farthest_point_sampling(xyz)  # [B, fps_num]
        group_idx = self.ball_query(xyz, centroids)    # [B, fps_num, group_num]
        
        # Get grouped points
        B = xyz.shape[0]
        grouped_xyz = torch.gather(
            xyz.unsqueeze(1).expand(-1, self.fps_num, -1, -1),
            2,
            group_idx.unsqueeze(-1).expand(-1, -1, -1, 3)
        )  # [B, fps_num, group_num, 3]
        
        # Get grouped colors if available
        grouped_colors = None
        if colors is not None:
            grouped_colors = torch.gather(
                colors.unsqueeze(1).expand(-1, self.fps_num, -1, -1),
                2,
                group_idx.unsqueeze(-1).expand(-1, -1, -1, 3)
            )  # [B, fps_num, group_num, 3]
        
        # Get centroids
        centroids_xyz = torch.gather(xyz, 1, centroids.unsqueeze(-1).expand(-1, -1, 3))  # [B, fps_num, 3]
        
        # Get centroid colors if available
        centroids_colors = None
        if colors is not None:
            centroids_colors = torch.gather(colors, 1, centroids.unsqueeze(-1).expand(-1, -1, 3))
        
        return grouped_xyz, grouped_colors, centroids_xyz, centroids_colors
    

class PointNetVisualizer:
    def __init__(self, fps_num=8, group_num=512, radius=0.2):
        self.preprocessor = PointNetPreprocessor(fps_num, group_num, radius)
        self.num_points_to_sample = 4096  # Number of points to sample from the point cloud
        self.gen_table = True  # Whether to generate table points
        self.threshold = 0.005  # Threshold for filtering points close to the plane
        
    def _set_black_background(self, vis):
        """Set the background color to black and adjust point size"""
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])  # Black background
        opt.point_size = 4.0  # Increased point size for better visibility
        
    def visualize(self, color_jpg_path, depth_png_path):
        # Generate point cloud from color and depth images
        colored_pcd,_ = gen_pcd(color_jpg_path, depth_png_path, self.num_points_to_sample, self.gen_table, self.threshold)
        
        # Split into coordinates and colors
        xyz = colored_pcd[:, :3]  # [N, 3]
        colors = colored_pcd[:, 3:6]  # [N, 3], RGB values
        
        # Convert to tensor and add batch dimension
        xyz_tensor = torch.from_numpy(xyz).float().unsqueeze(0)  # [1, N, 3]
        colors_tensor = torch.from_numpy(colors).float().unsqueeze(0)  # [1, N, 3]
        
        # Process through PointNetPreprocessor
        grouped_xyz, grouped_colors, rep_points, rep_colors = self.preprocessor(xyz_tensor, colors_tensor)
        
        # Convert to numpy
        grouped_xyz_np = grouped_xyz.squeeze(0).numpy()  # [fps_num, group_num, 3]
        grouped_colors_np = grouped_colors.squeeze(0).numpy()  # [fps_num, group_num, 3]
        rep_points_np = rep_points.squeeze(0).numpy()  # [fps_num, 3]
        rep_colors_np = rep_colors.squeeze(0).numpy()  # [fps_num, 3]
        
        # 1. Plot full colored point cloud with black background
        pcd_full = o3d.geometry.PointCloud()
        pcd_full.points = o3d.utility.Vector3dVector(xyz)
        pcd_full.colors = o3d.utility.Vector3dVector(colors)
        
        # Create visualizer and set black background
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Full Colored Point Cloud")
        vis.add_geometry(pcd_full)
        self._set_black_background(vis)
        vis.run()
        vis.destroy_window()
        
        # 2. Plot grouped point cloud with different colors for each group
        self._plot_grouped_point_cloud(grouped_xyz_np, rep_points_np)
        
        # 3. Plot each group separately with original colors and black background
        self._plot_individual_groups(grouped_xyz_np, grouped_colors_np, rep_points_np, rep_colors_np)
    
    def _plot_grouped_point_cloud(self, grouped_xyz, rep_points):
        """Plot the full point cloud with different colors for each group"""
        # Create a colormap with enough distinct colors
        cmap = plt.get_cmap('tab20')
        colors = cmap(np.linspace(0, 1, len(grouped_xyz)))
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        
        # Combine all points
        all_points = []
        all_colors = []
        
        for i, group in enumerate(grouped_xyz):
            # Group points
            group_points = group  # [group_num, 3]
            group_color = np.tile(colors[i][:3], (len(group_points), 1))
            
            all_points.append(group_points)
            all_colors.append(group_color)
            
            # Add centroid with same color but darker
            centroid = rep_points[i:i+1]  # [1, 3]
            centroid_color = np.array([colors[i][:3] * 0.7])  # Darker version
            
            all_points.append(centroid)
            all_colors.append(centroid_color)
        
        # Combine all points and colors
        all_points = np.vstack(all_points)
        all_colors = np.vstack(all_colors)
        
        pcd.points = o3d.utility.Vector3dVector(all_points)
        pcd.colors = o3d.utility.Vector3dVector(all_colors)
        
        # Create visualizer and set black background
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Grouped Point Cloud (Color by Group)")
        vis.add_geometry(pcd)
        self._set_black_background(vis)
        vis.run()
        vis.destroy_window()
    
    def _plot_individual_groups(self, grouped_xyz, grouped_colors, rep_points, rep_colors):
        """Plot each group in a separate window with centroids highlighted"""
        for i in range(len(grouped_xyz)):
            # Create point cloud for this group
            pcd = o3d.geometry.PointCloud()
            
            # Get points and colors for this group
            group_points = grouped_xyz[i]  # [group_num, 3]
            group_colors = grouped_colors[i]  # [group_num, 3]
            
            # Get centroid
            centroid = rep_points[i:i+1]  # [1, 3]
            centroid_color = np.array([[1, 0, 0]])  # Make centroid red for visibility
            
            # Combine points and colors
            points = np.vstack([centroid, group_points])
            colors = np.vstack([centroid_color, group_colors])
            
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            
            # Create visualizer and set black background
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name=f"Group {i+1} (Size: {len(group_points)})")
            vis.add_geometry(pcd)
            self._set_black_background(vis)
            vis.run()
            vis.destroy_window()

# Example usage:
if __name__ == "__main__":
    visualizer = PointNetVisualizer(fps_num=8, group_num=512, radius=0.2)
    color_path = 'table_pcd/000236_color_0.png'
    depth_path = 'table_pcd/000236_depth_0.png'
    visualizer.visualize(color_path, depth_path)