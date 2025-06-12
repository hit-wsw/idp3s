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
from math import ceil
from sklearn.metrics import mean_squared_error

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
    
    def _plot_action_group(self, demo_idx, group_name, dims, original_actions, predicted_actions):
        """绘制单个动作组的子图比较（含MSE指标）"""
        num_dims = len(dims)
        num_rows = ceil(num_dims / 4)  # 每行最多4个子图
        fig, axes = plt.subplots(num_rows, min(4, num_dims), 
                           figsize=(15, 3*num_rows))
        
        # 计算该组整体MSE
        group_mse = mean_squared_error(
            original_actions[:, dims], 
            predicted_actions[:, dims]
        )
        fig.suptitle(
            f'Demo {demo_idx+1} - {group_name.capitalize()} (MSE: {group_mse:.4e})',
            fontsize=14, y=1.05
        )
        
        if num_dims == 1:
            axes = np.array([axes])  # 确保1个子图也能迭代
        
        # 展平axes数组方便迭代
        axes = axes.flatten() if num_rows > 1 else axes
        
        for i, dim in enumerate(dims):
            ax = axes[i]
            original = original_actions[:, dim]
            predicted = predicted_actions[:, dim]
            diff = predicted - original
            
            # 计算当前维度MSE
            dim_mse = mean_squared_error(original, predicted)
            
            # 绘制三条曲线
            ax.plot(original, 'b-', label='Original', alpha=0.8, linewidth=1.5)
            ax.plot(predicted, 'r-', label='Predicted', alpha=0.8, linewidth=1.5)
            ax.plot(diff, 'g-', label='Difference', alpha=0.6, linewidth=1)
            
            ax.set_title(f'Dim {dim} (MSE: {dim_mse:.2e})', pad=8, fontsize=10)
            ax.set_xlabel('Frame', fontsize=8)
            ax.set_ylabel('Value', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', labelsize=8)
            
            # 只在第一个子图显示图例
            if i == 0:
                ax.legend(loc='upper right', fontsize=8, framealpha=0.5)
        
        # 隐藏多余的空子图
        for j in range(i+1, len(axes)):
            axes[j].axis('off')
        
        plt.tight_layout()
        plt.savefig(
            f'/home/wsw/D_wsw_ws/3d_dp/idp3_fig/action_comparison_demo{demo_idx+1}_{group_name}.png', 
            dpi=150, 
            bbox_inches='tight'
        )
        cprint(f"Saved action comparison plot for Demo {demo_idx+1} - {group_name}", 'green')
        plt.close()
    
    def _plot_action_differences(self, demo_idx, original_actions, predicted_actions):
        """绘制所有动作组的比较图"""
        action_groups = [
            ('waist', [0]),
            ('leftarm', range(1, 8)),
            ('rightarm', range(8, 15)),
            ('lefthand', range(15, 21)),
            ('righthand', range(21, 27))
        ]
        
        for group_name, dims in action_groups:
            self._plot_action_group(demo_idx, group_name, dims, 
                                  original_actions, predicted_actions)
    
    def get_pose_from_policy(self, cpkt_path, cpkt_name, zarr_path):
        """主函数：从策略获取姿态并比较动作"""
        self._load_policy(cpkt_path, cpkt_name)
        data, demo_starts, demo_ends = self._load_demo_data(zarr_path)
        
        # 随机选择2个demo
        selected_indices = np.random.choice(len(demo_starts), size=2, replace=False)
        
        for demo_idx, idx in enumerate(selected_indices):
            start = demo_starts[idx]
            end = demo_ends[idx]
            demo_length = end - start
            
            # 提取demo数据
            demo_pcd = data['point_cloud'][start:end]
            demo_state = data['state'][start:end]
            original_actions = data['action'][start:end]
            
            predicted_actions = []
            for i in range(demo_length):
                frame_dict = self._create_frame_dict(demo_pcd, demo_state, i, demo_length, num = self.cfg.n_obs_steps)
                action = self._get_action_from_policy(frame_dict)
                predicted_actions.append(action)
            
            predicted_actions = np.array(predicted_actions)
            
            # 确保形状一致
            min_length = min(predicted_actions.shape[0], original_actions.shape[0])
            predicted_actions = predicted_actions[:min_length]
            original_actions = original_actions[:min_length]
            
            # 绘制分组动作比较图
            self._plot_action_differences(demo_idx, original_actions, predicted_actions)
        
        return "Action group comparison plots generated."

    
def parse_args():
    parser = argparse.ArgumentParser(description='Train DP3 policy')
    parser.add_argument('--algo_name', type=str,  default='idp3_eval', 
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
    path = '/media/wsw/SSD1T1/data/idp/idp3/g1_sim_small_place_expert-idp3-first_seed0/checkpoints';name = 'latest'
    #path = '/media/wsw/SSD1T1/data/idp/idp3p/table/g1_sim_medium_6000_expert-idp3plus-loss4-16-8_seed0/checkpoints';name = '200'
    workspace = TrainDP3Workspace(config)
    #zarr_path = "/media/wsw/SSD1T1/data/idp3_data/g1_sim_small_4096_expert.zarr"
    zarr_path = '/media/wsw/SSD1T1/data/idp3_data/g1_sim_place_expert.zarr'
    workspace.get_pose_from_policy(path,name,zarr_path)


if __name__ == "__main__":
    main()
