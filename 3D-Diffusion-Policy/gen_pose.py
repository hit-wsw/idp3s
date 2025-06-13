'''
This code is used for visualized diffusion
'''
if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
import copy
import random
import numpy as np
import dill
from termcolor import cprint
import zarr
from diffusion_policy_3d.policy.idp3plus import DiffusionPointcloudPolicy as idp3p
import matplotlib.pyplot as plt
import argparse
import yaml
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import numpy as np
from PIL import Image

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainDP3Workspace:
    include_keys = ['global_step', 'epoch']
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir=None):
        self.cfg = cfg
        self._output_dir = output_dir
        self._saving_thread = None
        
        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: idp3p = hydra.utils.instantiate(cfg.policy)

        self.ema_model: idp3p = None
        if cfg.training.use_ema:
            try:
                self.ema_model = copy.deepcopy(self.model)
            except: # minkowski engine could not be copied. recreate it
                self.ema_model = hydra.utils.instantiate(cfg.policy)

        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        # configure training state
        self.global_step = 0
        self.epoch = 0

    def transform_state_dict(self, state_dict):

        processed_dict = {
            key: torch.from_numpy(np.expand_dims(value, axis=0))  # 新增第0维，然后转Tensor
            for key, value in state_dict.items()
        }

        return processed_dict
    
    def get_cpkt_from_path(self, path, name):
        return pathlib.Path(path).joinpath(f'{name}.ckpt')
    
    def load_checkpoint(self, path=None, tag='latest',
            exclude_keys=None, 
            include_keys=None, 
            **kwargs):
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        else:
            path = pathlib.Path(path)
        payload = torch.load(path.open('rb'), pickle_module=dill, map_location='cpu')
        self.load_payload(payload, 
            exclude_keys=exclude_keys, 
            include_keys=include_keys)
        return payload
    
    def load_payload(self, payload, exclude_keys=None, include_keys=None, **kwargs):
        if exclude_keys is None:
            exclude_keys = tuple()
        if include_keys is None:
            include_keys = payload['pickles'].keys()

        for key, value in payload['state_dicts'].items():
            if key not in exclude_keys:
                self.__dict__[key].load_state_dict(value, **kwargs)
        for key in include_keys:
            if key in payload['pickles']:
                self.__dict__[key] = dill.loads(payload['pickles'][key])
    
    def _get_action_from_policy(self, frame_dict):
        state = self.transform_state_dict(frame_dict)
        with torch.no_grad():
            action = self.model.get_action(state)
            return action.cpu().numpy()
        
    def _get_action_from_policy_show(self, frame_dict):
        state = self.transform_state_dict(frame_dict)
        with torch.no_grad():
            action_list, time_list = self.model.get_action_show(state)
            time_list = time_list.cpu().numpy()
            ac_data = [tensor.cpu().numpy() for tensor in action_list]
            return ac_data,time_list
        
    def _get_action_from_policy_show_fix_joint(self, frame_dict):
        state = self.transform_state_dict(frame_dict)
        with torch.no_grad():
            action_list, time_list = self.model.get_action_show_fix_joint(state)
            time_list = time_list.cpu().numpy()
            ac_data = [tensor.cpu().numpy() for tensor in action_list]
            return ac_data,time_list

        
    def _load_policy(self, cpkt_path, cpkt_name):
        cpkt = self.get_cpkt_from_path(cpkt_path, cpkt_name)
        if cpkt.is_file():
            cprint(f"Resuming from checkpoint {cpkt}", 'magenta')
            self.load_checkpoint(path=cpkt)
        else:
            raise FileNotFoundError(f"Checkpoint {cpkt} not found.")
        
        policy = self.model
        self.model.set_to_gen_ac()
        self.model.eval()
        self.model.cuda()
    
    def _create_frame_dict(self, demo_pcd, demo_state, current_idx, demo_length, num=2):
        # Calculate how many frames we can take without padding
        available_frames = min(num, demo_length - current_idx)
        
        # Get the indices for the frames we want (including padding if needed)
        frame_indices = list(range(current_idx, current_idx + available_frames))
        
        # If we need padding, repeat the last available index
        if len(frame_indices) < num:
            frame_indices += [frame_indices[-1]] * (num - len(frame_indices))
        
        # Create the frame dictionary
        frame_dict = {
            'point_cloud': np.stack([demo_pcd[i] for i in frame_indices]),
            'agent_pos': np.stack([demo_state[i] for i in frame_indices]),
        }
        return frame_dict
    
    def _load_demo_data(self, zarr_path):
        root = zarr.open(zarr_path, mode='r')
        data = root['data']
        meta = root['meta']
        episode_ends = meta['episode_ends'][:]
        demo_starts = np.concatenate(([0], episode_ends[:-1]))
        demo_ends = episode_ends
        return data, demo_starts, demo_ends

    def show_diffuser(self, frame_dict):
        """
        Show denoising process with each step in a separate figure
        
        Parameters:
            action_list (list): List of denoising steps, each element is [27] array
            time_list (list): List of timesteps
        """
        # Convert to numpy arrays
        action_list,time_list = self._get_action_from_policy_show(frame_dict)
        action_array = np.array(action_list)
        time_array = np.array(time_list)
        
        T, dim = action_array.shape
        
        for t in range(T):
            # Create a new figure for each timestep
            plt.figure(figsize=(10, 6))
            
            # Create the bar plot
            bars = plt.bar(range(dim), action_array[t], color='skyblue')
            
            # Set titles and labels
            if t == 0:
                plt.title("Denoising Process: Pure Noise")
            else:
                plt.title(f"Denoising Process: t={time_array[t-1]}")
            
            plt.xlabel("Dimension")
            plt.ylabel("Value")
            plt.xticks(range(0, dim, 3))  # Show every 3rd dimension label
            plt.ylim(min(action_array.min(), -1), max(action_array.max(), 1))
            plt.grid(True, axis='y', linestyle='--', alpha=0.7)
            
            # Highlight significant changes (for steps after the first)
            if t > 0:
                changes = np.abs(action_array[t] - action_array[t-1])
                for i, change in enumerate(changes):
                    if change > 0.1:  # Threshold for significant change
                        bars[i].set_color('skyblue')
            
            plt.tight_layout()
            plt.show()

    def show_diffuser_gif(self, frame_dict, filename="denoising_process.gif"):
        """
        Create an animated GIF of the denoising process
        
        Parameters:
            action_list (list): List of denoising steps, each element is [27] array
            time_list (list): List of timesteps
            filename (str): Output GIF filename
        """
        # Convert to numpy arrays
        action_list,time_list = self._get_action_from_policy_show(frame_dict)
        action_array = np.array(action_list)
        time_array = np.array(time_list)
        
        T, dim = action_array.shape
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        def update(t):
            ax.clear()
            bars = ax.bar(range(dim), action_array[t], color='skyblue')
            
            # Set titles and labels
            if t == 0:
                ax.set_title("Denoising Process: Pure Noise")
            else:
                ax.set_title(f"Denoising Process: t={time_array[t-1]}")
            
            ax.set_xlabel("Dimension")
            ax.set_ylabel("Value")
            ax.set_xticks(range(0, dim, 3))
            ax.set_ylim(min(action_array.min(), -1), max(action_array.max(), 1))
            ax.grid(True, axis='y', linestyle='--', alpha=0.7)
            
            # Highlight significant changes
            if t > 0:
                changes = np.abs(action_array[t] - action_array[t-1])
                for i, change in enumerate(changes):
                    if change > 0.1:  # Threshold for significant change
                        bars[i].set_color('skyblue')
            
            return bars
        
        ani = animation.FuncAnimation(
            fig, update, frames=T, interval=500, blit=False
        )
        
        # Save as GIF
        writer = PillowWriter(fps=2)  
        ani.save(filename, writer=writer)
        plt.close()
        print(f"GIF saved as {filename}")

    def show_diffuser_fix_joint(self, frame_dict):
        """
        展示扩散过程中第0维度动作的变化，生成T+1张图
        
        参数:
        - action_list: 长度为T+1的列表，每个元素是[time, 27]的array
        - time_list: 长度为T的array，表示扩散去噪时的时间步
        """
        # 确保 action_list 长度为 T+1，time_list 长度为 T
        action_list,time_list = self._get_action_from_policy_show_fix_joint(frame_dict)
        assert len(action_list) == len(time_list) + 1, "action_list 长度应为 time_list 长度 +1"
        
        # 生成 T+1 张图
        for k in range(len(action_list)):
            action = action_list[k]  # 当前动作序列 [time, 27]
            time_steps = np.arange(action.shape[0])  # 横坐标: 0 到 time-1
            dim0_action = action[:, 0]  # 第0维度的动作序列
            
            plt.figure(figsize=(10, 4))
            
            if k == 0:
                plt.title(f"Action[0] - Pure Noise (Step 0)")
            else:
                plt.title(f"Action[0] - Denoised at Step {time_list[k-1]}")
            
            plt.plot(time_steps, dim0_action, 'b-', linewidth=2)
            plt.xlabel("Time Step (0 to time-1)")
            plt.ylabel("Action[0] Value")
            plt.grid(True)
            plt.show()

    def show_diffuser_gif_fixed_joint(self, frame_dict, gif_path="denoising_process_fixed.gif", fps=2):
        """
        将扩散过程中第0维度动作的变化生成GIF动画
        
        参数:
        - action_list: 长度为T+1的列表，每个元素是[time, 27]的array
        - time_list: 长度为T的array，表示扩散去噪时的时间步
        - gif_path: 输出的GIF文件路径
        - fps: 帧率 (frames per second)
        """
        action_list,time_list = self._get_action_from_policy_show_fix_joint(frame_dict)
        assert len(action_list) == len(time_list) + 1, "action_list 长度应为 time_list 长度 +1"
    
        # 临时保存所有帧的目录
        temp_dir = "temp_frames"
        os.makedirs(temp_dir, exist_ok=True)
        
        # 生成所有帧并保存为图片
        frame_paths = []
        for k in range(len(action_list)):
            action = action_list[k]
            time_steps = np.arange(action.shape[0])
            dim0_action = action[:, 0]
            
            fig = plt.figure(figsize=(10, 4))
            
            if k == 0:
                plt.title(f"Action[0] - Pure Noise (Step 0)", fontsize=12)
            else:
                plt.title(f"Action[0] - Denoised at Step {time_list[k-1]}", fontsize=12)
            
            plt.plot(time_steps, dim0_action, 'b-', linewidth=2)
            plt.xlabel("Time Step (0 to time-1)", fontsize=10)
            plt.ylabel("Action[0] Value", fontsize=10)
            plt.grid(True)
            
            # 保存帧
            frame_path = os.path.join(temp_dir, f"frame_{k:03d}.png")
            plt.savefig(frame_path, bbox_inches='tight', dpi=100)
            frame_paths.append(frame_path)
            plt.close(fig)  # 明确关闭图形
        
        # 确保所有图片尺寸一致
        images = []
        first_image = Image.open(frame_paths[0])
        for path in frame_paths:
            img = Image.open(path)
            # 转换为RGB模式并调整尺寸以匹配第一帧
            img = img.convert('RGB').resize(first_image.size)
            images.append(img)
        
        # 将帧合成为GIF
        images[0].save(
            gif_path,
            format='GIF',
            append_images=images[1:],
            save_all=True,
            duration=1000//fps,  # 每帧持续时间(ms)
            loop=0  # 无限循环
        )
        
        # 清理临时文件
        for frame in frame_paths:
            os.remove(frame)
        os.rmdir(temp_dir)
        
        print(f"GIF saved to {gif_path}")

    def get_pose_from_policy(self, cpkt_path, cpkt_name, zarr_path):
        """主函数：从策略获取姿态并比较动作"""
        self._load_policy(cpkt_path, cpkt_name)
        data, demo_starts, demo_ends = self._load_demo_data(zarr_path)
        
        # 随机选择2个demo
        selected_indices = np.random.choice(len(demo_starts), size=1, replace=False)
        
        for demo_idx, idx in enumerate(selected_indices):
            start = demo_starts[idx]
            end = demo_ends[idx]
            demo_length = end - start
            
            # 提取demo数据
            demo_pcd = data['point_cloud'][start:end]
            demo_state = data['state'][start:end]

            
            frame_dict = self._create_frame_dict(demo_pcd, demo_state, 0, demo_length, num = self.cfg.n_obs_steps)
            #self.show_diffuser_fix_joint(frame_dict)
            self.show_diffuser_gif_fixed_joint(frame_dict)
            #self.show_diffuser(frame_dict)
            #self.show_diffuser_gif(frame_dict)
                       
        return "Action group comparison plots generated."

    
def parse_args():
    parser = argparse.ArgumentParser(description='Train DP3 policy')
    parser.add_argument('--algo_name', type=str,  default='idp3plus_eval', 
                       help='Algorithm name (e.g., idp3plus_eval, idp3_eval)')
    return parser.parse_args()

def load_config_from_yaml(config_name):
    """Load YAML config file and convert to OmegaConf"""
    cfg_name = config_name + '.yaml'
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy_3d', 'config' , cfg_name))
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return OmegaConf.create(config_dict)

def main():
    args = parse_args()
    config = load_config_from_yaml(args.algo_name)
    #path = '/media/wsw/SSD1T1/data/idp/idp3/g1_sim_small_place_expert-idp3-first_seed0/checkpoints';name = 'latest'
    path = '/media/wsw/SSD1T1/data/idp/idp3p/table/g1_sim_medium_6000_expert-idp3plus-loss4-16-8_seed0/checkpoints';name = '200'
    workspace = TrainDP3Workspace(config)
    #zarr_path = "/media/wsw/SSD1T1/data/idp3_data/g1_sim_small_4096_expert.zarr"
    zarr_path = '/media/wsw/SSD1T1/data/idp3_data/g1_sim_place_expert.zarr'
    workspace.get_pose_from_policy(path,name,zarr_path)


if __name__ == "__main__":
    main()
