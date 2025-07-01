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
from diffusion_policy_3d.policy.idp3plus import DiffusionPointcloudPolicy as iDP3p
import argparse
import yaml

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
        self.model: iDP3p = hydra.utils.instantiate(cfg.policy)

        self.ema_model: iDP3p = None
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
    
    def _create_frame_dict(self, demo_pcd, demo_state, current_idx, demo_length):
        if current_idx == demo_length - 1:  # 处理最后一帧
            frame_dict = {
                'point_cloud': np.stack([demo_pcd[current_idx], demo_pcd[current_idx]]),
                'agent_pos': np.stack([demo_state[current_idx], demo_state[current_idx]]),
            }
        else:
            frame_dict = {
                'point_cloud': np.stack([demo_pcd[current_idx], demo_pcd[current_idx+1]]),
                'agent_pos': np.stack([demo_state[current_idx], demo_state[current_idx+1]]),
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
    
    def get_pose_from_policy(self, cpkt_path, cpkt_name, zarr_path):
        """主函数：从策略获取姿态"""
        self._load_policy(cpkt_path, cpkt_name)
        data, demo_starts, demo_ends = self._load_demo_data(zarr_path)
        
        # 随机选择2个demo
        selected_indices = np.random.choice(len(demo_starts), size=2, replace=False)
        
        results = []
        for idx in selected_indices:
            start = demo_starts[idx]
            end = demo_ends[idx]
            demo_length = end - start
            
            # 提取demo数据
            demo_pcd = data['point_cloud'][start:end]
            demo_state = data['state'][start:end]
            
            for i in range(demo_length):
                frame_dict = self._create_frame_dict(demo_pcd, demo_state, i, demo_length)
                action = self._get_action_from_policy(frame_dict)
                print(action)

    def get_action_from_state(self, cpkt_path, cpkt_name,frame_dict):
        self._load_policy(cpkt_path, cpkt_name)
        state = self.transform_state_dict(frame_dict)
        with torch.no_grad():
            action = self.model.get_action(state)
            print(action)
            return action.cpu().numpy()

    

def parse_args():
    parser = argparse.ArgumentParser(description='Train DP3 policy')
    parser.add_argument('--algo_name', type=str,  default='idp3plus_eval', 
                       help='Algorithm name (e.g., dp3, simple_dp3)')
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
    path = '/media/wsw/SSD1T1/data/idp/idp3p/table/g1_sim_medium_6000_expert-idp3plus-loss4-16-8_seed0/checkpoints';name = '200'
    original_dict = {
        'point_cloud': np.random.rand(2, 10000, 3),
        'agent_pos': np.random.rand(2, 27),
        }

    workspace = TrainDP3Workspace(config)
    workspace.get_action_from_state(path,name,original_dict)


if __name__ == "__main__":
    main()
