import numpy as np
import zarr
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils import resize_image, gen_pcd

def read_pcd_data(zarr_path):
    # 打开 Zarr 文件
    zarr_root = zarr.open(zarr_path, mode='r')
    point_cloud = zarr_root['data/point_cloud']
    episode_ends = zarr_root['meta/episode_ends'][:]
    start_idx = 0
    end_idx = episode_ends[0]
    first_episode_point_cloud = point_cloud[start_idx + 40]

    print(f"First episode point cloud shape: {first_episode_point_cloud.shape}")
    return first_episode_point_cloud

def get_pcd_from_img():
    color_jpg_path = 'table_pcd/color_image.jpg'
    depth_png_path = 'table_pcd/depth_image.png'
    num_points_to_sample = 4096
    gen_table = True
    threshold = 0.005  # 深度阈值
    color_pcd, uncolored_pcd = gen_pcd(color_jpg_path, depth_png_path, num_points_to_sample, gen_table, threshold)
    return uncolored_pcd


def farthest_point_sampling(xyz, npoint):
    N, _ = xyz.shape
    centroids = np.zeros((npoint,), dtype=np.int64)
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, axis=-1)
        distance = np.minimum(distance, dist)
        farthest = np.argmax(distance)
    return xyz[centroids], centroids

def ball_query(radius, nsample, xyz, centroids):
    group_idx = []
    for centroid in centroids:
        center = xyz[centroid]
        dist = np.sqrt(np.sum((xyz - center) ** 2, axis=-1))
        idx = np.where(dist < radius)[0]
        if len(idx) > nsample:
            idx = idx[:nsample]
        elif len(idx) < nsample:
            # 若邻居不足，使用重复填充
            pad = np.random.choice(idx, nsample - len(idx), replace=True)
            idx = np.concatenate([idx, pad])
        group_idx.append(idx)
    return np.array(group_idx)  # [npoint, nsample]

def group_points(xyz, group_idx, centroids):
    grouped_xyz = xyz[group_idx]  # [npoint, nsample, 3]
    centroid_xyz = xyz[centroids][:, np.newaxis, :]  # [npoint, 1, 3]
    relative_xyz = grouped_xyz - centroid_xyz        # 相对坐标
    return relative_xyz, grouped_xyz

def plot_grouped_points(xyz, group_idx, centroids):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = plt.cm.get_cmap("tab20", len(centroids))  # 使用不同颜色

    # 绘制每组点云
    for i, idx in enumerate(group_idx):
        group_points = xyz[idx]  # 当前组的点云
        ax.scatter(group_points[:, 0], group_points[:, 1], group_points[:, 2], color=colors(i), label=f'Group {i}')
        ax.scatter(xyz[centroids[i], 0], xyz[centroids[i], 1], xyz[centroids[i], 2], color='black', marker='x', s=50)  # 中心点

    ax.legend()
    plt.show()

def get_processed_pcd(path,fps_num,group_num):
    #pcd = read_pcd_data(path)
    pcd = get_pcd_from_img()
    sampled_xyz, centroid_idx = farthest_point_sampling(pcd, fps_num)
    group_idx = ball_query(radius=0.2, nsample=group_num, xyz=pcd, centroids=centroid_idx)
    relative_xyz, grouped_xyz = group_points(pcd, group_idx, centroid_idx)  # 计算相对坐标
    combined_points = np.zeros((fps_num, group_num+1, 3))  # 初始化结果数组
    for i in range(fps_num):
        combined_points[i, 0] = sampled_xyz[i]  # 第一个点为采样点的绝对坐标
        combined_points[i, 1:] = relative_xyz[i]  # 后续为相对坐标
    return combined_points

if __name__ == "__main__":
    # 1. 读取点云数据
    path = '3D-Diffusion-Policy/data/g1_sim_test_expert.zarr'
    fps_num = 8;group_num = 512

    combined_points = get_processed_pcd(path, fps_num, group_num)



