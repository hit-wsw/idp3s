import numpy as np
import open3d as o3d
import copy



# the corner points of the robot table, used for remove redundant points
robot_table_corner_points = np.array([
    [-0.262721, -0.25, -0.077183],
    [0.228490, -0.23, -0.079729],
    [0.390924, -0.215, -0.783124],
    [-0.372577, -0.235, -0.781769]
])

# robot table sweep list
table_sweep_list = [0.020, 0.021, 0.022, 0.023, 0.024, 0.025]

# depth camera intrinsic
o3d_depth_intrinsic = o3d.camera.PinholeCameraIntrinsic(
    1280, 720,
    898.2010498046875,
    897.86669921875,
    657.4981079101562,
    364.30950927734375)

d435i_depth_intrinsic = o3d.camera.PinholeCameraIntrinsic(
    1280, 720,
    923.995567,
    923.492024,
    620.892371,
    384.178306)

head_pose = np.eye(4)
head_pose[:3, :3] = np.array([[1.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0],
                                [0.0, 0.0, 1.0]])
head_pose[:3, 3] = np.array([0.0, 0.0, 0.0])