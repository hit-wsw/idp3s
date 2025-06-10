import open3d as o3d
import numpy as np
import os
from hyperparameters import *
from utils import *

def gen_pcd(metadata_root):
    """
    生成点云
    """
    color_jpg_path = os.path.join(metadata_root, '000000_color_0.png')
    depth_png_path = os.path.join(metadata_root, '000000_depth_0.png')
    '''color_jpg_path = os.path.join(metadata_root, 'color_image.jpg')
    depth_png_path = os.path.join(metadata_root, 'depth_image.png')'''
    color_image_o3d = o3d.io.read_image(color_jpg_path)
    depth_image_o3d = o3d.io.read_image(depth_png_path)
    max_depth = 1000
    depth_array = np.asarray(depth_image_o3d)
    mask = depth_array > max_depth
    depth_array[mask] = 0
    filtered_depth_image = o3d.geometry.Image(depth_array)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color_image_o3d, filtered_depth_image, depth_trunc=4.0, convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, d435i_depth_intrinsic)
    pcd.transform(head_pose)
    color_pcd = np.concatenate((np.array(pcd.points), np.array(pcd.colors)), axis=-1)
    return color_pcd

def select_points(pcd):
    """
    让用户从点云中选择4个点，返回这些点的坐标
    """
    print("请按住Shift+左键选择4个点，按Q结束选择")
    
    vis = o3d.visualization.VisualizerWithEditing()
    pcd_vis = o3d.geometry.PointCloud()
    pcd_vis.points = o3d.utility.Vector3dVector(pcd[:, :3])
    pcd_vis.colors = o3d.utility.Vector3dVector(pcd[:, 3:])
    vis.create_window()
    vis.add_geometry(pcd_vis)
    vis.run()  # 用户交互选择点
    vis.destroy_window()
    
    # 获取选择的点索引
    picked_points = vis.get_picked_points()
    if len(picked_points) != 4:
        raise ValueError("请准确选择4个点！")
    
    # 返回选择的点坐标
    selected_points = pcd[picked_points, :3]
    return selected_points

def cal_plane_from_four_point(points):
    """
    从4个点计算4个可能的平面，验证它们是否接近
    """
    from itertools import combinations
    
    # 生成所有4选3的组合
    point_combinations = list(combinations(points, 3))
    
    planes = []
    for combo in point_combinations:
        # 计算平面方程 ax + by + cz + d = 0
        p1, p2, p3 = combo
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        normal = normal / np.linalg.norm(normal)  # 单位法向量
        d = -np.dot(normal, p1)
        planes.append(np.append(normal, d))
    
    # 验证平面是否接近（距离<0.5cm）
    ref_plane = planes[0]
    for plane in planes[1:]:
        # 计算两个平面之间的距离
        angle_diff = np.arccos(np.dot(ref_plane[:3], plane[:3]))
        dist_diff = np.abs(ref_plane[3] - plane[3])
        
        if angle_diff > 0.1 or dist_diff > 0.005:  # 0.1弧度≈5.7度，0.005m=0.5cm
            return None
    
    return planes[0]  # 返回第一个平面

def create_plane_mesh(plane, size=50):
    """
    创建平面网格用于可视化（单位：cm）
    修正后的版本，确保平面正确显示
    """
    a, b, c, d = plane
    
    # 创建网格平面（使用create_plane而不是create_box）
    mesh = o3d.geometry.TriangleMesh.create_plane(
        width=size,
        height=size,
        resolution=10
    )
    
    # 计算平面法向量
    normal = np.array([a, b, c])
    normal = normal / np.linalg.norm(normal)  # 确保是单位向量
    
    # 计算旋转使平面法向量对齐
    z_axis = np.array([0, 0, 1])
    
    # 处理法向量与z轴平行的情况
    if np.allclose(np.abs(np.dot(normal, z_axis)), 1.0, atol=1e-6):
        # 法向量与z轴平行，不需要旋转
        rotation_matrix = np.eye(3)
    else:
        # 计算旋转轴和角度
        rotation_axis = np.cross(z_axis, normal)
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        rotation_angle = np.arccos(np.dot(z_axis, normal))
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(
            rotation_axis * rotation_angle
        )
    
    mesh.rotate(rotation_matrix)
    
    # 计算平面上的一个点（选择平面与法线的交点）
    if np.abs(d) > 1e-6:  # 平面不通过原点
        point_on_plane = -d * normal / (a**2 + b**2 + c**2)
    else:  # 平面通过原点
        # 选择法向量上的一个点作为参考
        point_on_plane = normal * size/2
    
    mesh.translate(point_on_plane)
    
    # 设置平面颜色和透明度
    mesh.paint_uniform_color([1, 0, 0])  # 红色
    mesh.compute_vertex_normals()
    
    return mesh

def show_pcd_with_plane(pcd, plane):
    """
    显示点云和计算出的平面（单位：cm）
    修正后的版本
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # 原始点云
    pcd_vis = o3d.geometry.PointCloud()
    pcd_vis.points = o3d.utility.Vector3dVector(pcd[:, :3])
    pcd_vis.colors = o3d.utility.Vector3dVector(pcd[:, 3:])
    vis.add_geometry(pcd_vis)
    
    # 创建平面网格
    plane_mesh = create_plane_mesh(plane, size=50)  # 50cm的平面
    
    # 设置平面渲染属性
    plane_mesh.compute_vertex_normals()
    vis.add_geometry(plane_mesh)
    
    # 添加坐标系（10cm大小）
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.10,
        origin=[0, 0, 0]
    )
    vis.add_geometry(coordinate_frame)
    
    # 添加原点标记（红色小球）
    origin_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
    origin_sphere.translate([0, 0, 0])
    origin_sphere.paint_uniform_color([1, 0, 0])
    origin_sphere.compute_vertex_normals()
    vis.add_geometry(origin_sphere)
    
    # 设置渲染选项
    opt = vis.get_render_option()
    opt.mesh_show_back_face = True  # 显示平面双面
    opt.light_on = True  # 开启光照
    
    vis.run()
    vis.destroy_window()

def show_pcd(pcd, table_pcd):
    """
    显示原始点云和桌子点云（标红）
    参数：
        pcd: 原始点云，形状为(N,6)，前3列是XYZ坐标(cm)，后3列是RGB颜色
        table_pcd: 桌子点云，形状为(M,6)，格式同pcd
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # 原始点云（保持原色）
    pcd_vis = o3d.geometry.PointCloud()
    pcd_vis.points = o3d.utility.Vector3dVector(pcd[:, :3])
    pcd_vis.colors = o3d.utility.Vector3dVector(pcd[:, 3:])
    vis.add_geometry(pcd_vis)
    
    # 桌子点云（标红）
    table_pcd_vis = o3d.geometry.PointCloud()
    table_pcd_vis.points = o3d.utility.Vector3dVector(table_pcd[:, :3])
    
    # 确保桌子点云为红色（如果原本不是）
    if table_pcd.shape[1] >= 6:
        table_colors = np.ones_like(table_pcd[:, 3:]) * [1, 0, 0]  # 全部设为红色
    else:
        table_colors = np.ones((len(table_pcd), 3)) * [1, 0, 0]
    
    table_pcd_vis.colors = o3d.utility.Vector3dVector(table_colors)
    vis.add_geometry(table_pcd_vis)
    
    # 添加坐标系（10cm大小）
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.10)  # 0.10m=10cm
    vis.add_geometry(coordinate_frame)
    
    # 添加原点标记（红色小球，0.5cm半径）
    origin_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)  # 0.005m=0.5cm
    origin_sphere.translate([0, 0, 0])
    origin_sphere.paint_uniform_color([1, 0, 0])
    vis.add_geometry(origin_sphere)
    
    # 设置渲染选项
    opt = vis.get_render_option()
    opt.point_size = 2.0  # 点云大小
    
    vis.run()
    vis.destroy_window()

def filter_points_by_plane(pcd, plane, threshold=0.005):
    """
    过滤掉距离平面小于threshold的点
    """
    points = pcd[:, :3]
    a, b, c, d = plane
    
    # 计算每个点到平面的距离
    distances = np.abs(a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d) / np.sqrt(a**2 + b**2 + c**2)
    
    # 保留距离大于阈值的点
    mask = distances > threshold
    mask1 = distances <= threshold
    filtered_pcd = pcd[mask]
    filtered_pcd1 = pcd[mask1]
    
    return filtered_pcd, filtered_pcd1


def main():
    # 1. 生成点云
    metadata_root = "table_pcd"  # 修改为实际路径
    color_pcd = gen_pcd(metadata_root)
    
    while True:
        try:
            # 2. 选择4个点
            selected_points = select_points(color_pcd)
            
            # 3. 计算并验证平面
            plane = cal_plane_from_four_point(selected_points)
            
            if plane is not None:
                print(f"成功计算平面方程: {plane[0]:.3f}x + {plane[1]:.3f}y + {plane[2]:.3f}z + {plane[3]:.3f} = 0")
                break
            else:
                print("选择的点不共面，请重新选择！")
        except ValueError as e:
            print(e)
    filtered_pcd ,table_pcd = filter_points_by_plane(color_pcd, plane)
    show_pcd(filtered_pcd,table_pcd)  # 使用原有的show_pcd函数显示过滤后的点云
    # 6. 保存平面参数到文件
    plane_folder = "table_plane"
    os.makedirs(plane_folder, exist_ok=True)  # 创建文件夹，如果已存在则忽略
    
    plane_file = os.path.join(plane_folder, "table.txt")
    with open(plane_file, 'w') as f:
        f.write(f"{plane[0]:.6f} {plane[1]:.6f} {plane[2]:.6f} {plane[3]:.6f}")
    
    print(f"平面参数已保存到: {plane_file}")

if __name__ == "__main__":
    main()