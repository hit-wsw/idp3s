# usage:
#       bash scripts/vrl3_gen_demonstration_expert.sh door
import matplotlib.pyplot as plt
import argparse
import os
import torch
from utils import resize_image, gen_pcd
from termcolor import cprint
from PIL import Image
import zarr
from copy import deepcopy
import json
import cv2
from hyperparameters import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='g1_sim_pcd', help='environment to run')
    parser.add_argument('--root_dir', type=str, default='/home/wsw/D_wsw_ws/3d_dp/3D-Diffusion-Policy/3D-Diffusion-Policy/data', help='directory to save data')
    parser.add_argument('--meatdata_path', type=str, default='/home/wsw/D_wsw_ws/3d_dp/expert_data', help='path to meta data')
    parser.add_argument('--img_size', type=int, default=84, help='image size')
    parser.add_argument('--pcd_sample', type=int, default=10000, help='num points to sample pcd')
    parser.add_argument('--table_threshold', type=float, default=0.005, help='threshold to distinguish table pcd')
    parser.add_argument('--gen_table', action='store_true', help='remove table pcd (need cali)')
    parser.add_argument('--not_use_multi_view', action='store_true', help='not use multi view')
    parser.add_argument('--use_point_crop', action='store_true', help='use point crop')
    
    args = parser.parse_args()
    return args


def process_zarr_data(save_dir, meta_path, num_points_to_sample, img_size, gen_table, threshold):

    episode_dirs = sorted([
        d for d in os.listdir(meta_path)
        if d.startswith('episode_')
    ], key=lambda x: int(x.split('_')[1]))

    '''vis = o3d.visualization.Visualizer()
    vis.create_window()
    pcd_vis = o3d.geometry.PointCloud()  # Empty point cloud for starters
    firstfirst = True'''

    zarr_root = zarr.group(save_dir)
    
    # ini parameters
    total_count = 0
    img_arrays = []
    depth_arrays = []
    point_cloud_arrays = []
    waist_arrays = []
    arm_arrays = []
    hand_arrays = []
    state_arrays = []
    action_arrays = []
    episode_ends_arrays = []

    for episode in episode_dirs:
        json_path = os.path.join(meta_path, episode, 'data.json')
        # Load clip marks
        with open(json_path, 'r') as f:
            data = json.load(f)
            for item in data['data']:

                color_jpg_path = os.path.join(meta_path, episode, item['colors']['color_0'])
                depth_png_path = os.path.join(meta_path, episode, item['colors']['depth_0'])
                depth_img = cv2.imread(depth_png_path)

                # process image
                resized_color_image = resize_image(color_jpg_path, img_size)
                resized_depth_image = cv2.resize(depth_img, (img_size, img_size))
                img_arrays.append(resized_color_image)
                depth_arrays.append(resized_depth_image)

                # process pcd
                color_pcd, uncolored_pcd = gen_pcd(color_jpg_path, depth_png_path, num_points_to_sample, gen_table, threshold)
                point_cloud_arrays.append(copy.deepcopy(uncolored_pcd))

                # load pose data
                waist_data = item['states']['waist']['qpos']
                left_arm_data = item['states']['left_arm']['qpos'];right_arm_data = item['states']['right_arm']['qpos']
                left_hand_data = item['states']['left_hand']['qpos'];right_hand_data = item['states']['right_hand']['qpos']
                arm_data = np.concatenate((left_arm_data, right_arm_data), axis=-1);hand_data = np.concatenate((left_hand_data, right_hand_data), axis=-1)
                state = np.concatenate((waist_data, arm_data, hand_data), axis=-1)
                
                waist_arrays.append(np.array(waist_data));arm_arrays.append(np.array(arm_data))
                hand_arrays.append(np.array(hand_data));state_arrays.append(np.array(state))

                # load action data
                waist_ac_data = item['actions']['waist']['qpos']
                left_arm_ac_data = item['actions']['left_arm']['qpos'];right_arm_ac_data = item['actions']['right_arm']['qpos']
                left_hand_ac_data = item['actions']['left_hand']['qpos'];right_hand_ac_data = item['actions']['right_hand']['qpos']
                arm_ac_data = np.concatenate((left_arm_ac_data, right_arm_ac_data), axis=-1);hand_ac_data = np.concatenate((left_hand_ac_data, right_hand_ac_data), axis=-1)
                action = np.concatenate((waist_ac_data, arm_ac_data, hand_ac_data), axis=-1)
                
                action_arrays.append(np.array(action))

                length = item['idx'] +1

                '''# update pointcloud visualization
                pcd_vis.points = o3d.utility.Vector3dVector(color_pcd[:, :3])
                pcd_vis.colors = o3d.utility.Vector3dVector(color_pcd[:, 3:])

                if firstfirst:
                    vis.add_geometry(pcd_vis)
                    firstfirst = False
                else:
                    vis.update_geometry(pcd_vis)
                vis.poll_events()
                vis.update_renderer()'''

                # update image visualization
                cv2.imshow("resized_image", resized_color_image)
                cv2.waitKey(1)
                
            
            total_count += length
            episode_ends_arrays.append(total_count)
    

    zarr_data = zarr_root.create_group('data')
    zarr_meta = zarr_root.create_group('meta')
    # save img, state, action arrays into data, and episode ends arrays into meta
    img_arrays = np.stack(img_arrays, axis=0)
    if img_arrays.shape[1] == 3: # make channel last
        img_arrays = np.transpose(img_arrays, (0,2,3,1))
    state_arrays = np.stack(state_arrays, axis=0)
    point_cloud_arrays = np.stack(point_cloud_arrays, axis=0)
    depth_arrays = np.stack(depth_arrays, axis=0)
    action_arrays = np.stack(action_arrays, axis=0)
    episode_ends_arrays = np.array(episode_ends_arrays)

    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
    img_chunk_size = (100, img_arrays.shape[1], img_arrays.shape[2], img_arrays.shape[3])
    state_chunk_size = (100, state_arrays.shape[1])
    point_cloud_chunk_size = (100, point_cloud_arrays.shape[1], point_cloud_arrays.shape[2])
    depth_chunk_size = (100, depth_arrays.shape[1], depth_arrays.shape[2])
    action_chunk_size = (100, action_arrays.shape[1])
    zarr_data.create_dataset('img', data=img_arrays, chunks=img_chunk_size, dtype='uint8', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('state', data=state_arrays, chunks=state_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('point_cloud', data=point_cloud_arrays, chunks=point_cloud_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('depth', data=depth_arrays, chunks=depth_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('action', data=action_arrays, chunks=action_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays, dtype='int64', overwrite=True, compressor=compressor)
    
    
    # print shape
    cprint(f'img shape: {img_arrays.shape}, range: [{np.min(img_arrays)}, {np.max(img_arrays)}]', 'green')
    cprint(f'point_cloud shape: {point_cloud_arrays.shape}, range: [{np.min(point_cloud_arrays)}, {np.max(point_cloud_arrays)}]', 'green')
    cprint(f'depth shape: {depth_arrays.shape}, range: [{np.min(depth_arrays)}, {np.max(depth_arrays)}]', 'green')
    cprint(f'state shape: {state_arrays.shape}, range: [{np.min(state_arrays)}, {np.max(state_arrays)}]', 'green')
    cprint(f'action shape: {action_arrays.shape}, range: [{np.min(action_arrays)}, {np.max(action_arrays)}]', 'green')
    cprint(f'Saved zarr file to {save_dir}', 'green')
    
    cprint(f'Saved zarr file to {save_dir}', 'green')
    
    # clean up
    del img_arrays, state_arrays, point_cloud_arrays, action_arrays, episode_ends_arrays
    del zarr_root, zarr_data, zarr_meta

def main():
    args = parse_args()
    save_dir = os.path.join(args.root_dir, args.env_name+'_expert.zarr')
    if os.path.exists(save_dir):
        cprint('Data already exists at {}'.format(save_dir), 'red')
        cprint("If you want to overwrite, delete the existing directory first.", "red")
        cprint("Do you want to overwrite? (y/n)", "red")
        # user_input = input()
        user_input = 'y'
        if user_input == 'y':
            cprint('Overwriting {}'.format(save_dir), 'red')
            os.system('rm -rf {}'.format(save_dir))
        else:
            cprint('Exiting', 'red')
            return
    os.makedirs(save_dir, exist_ok=True)    

    process_zarr_data(save_dir, args.meatdata_path, args.pcd_sample, args.img_size, args.gen_table, args.table_threshold)
    
if __name__ == '__main__':
    main()