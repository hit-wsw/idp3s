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
import dill
from omegaconf import OmegaConf
import pathlib
from train import TrainDP3Workspace
import numpy as np

OmegaConf.register_new_resolver("eval", eval, replace=True)
    

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy_3d', 'config'))
)
def main(cfg):
    path = '/home/wsw'
    name = '210'
    original_dict = {
        'point_cloud': np.random.rand(2, 10000, 3),  # 示例形状 [T=10, N=5, D=3]
        'agent_pos': np.random.rand(2, 27),
        }

    workspace = TrainDP3Workspace(cfg)
    action = workspace.get_pose_from_policy(path,name,original_dict)
    while action is not None:
        print(action[0])
        action = action[1:]
        if len(action) == 0:
            break

if __name__ == "__main__":
    main()
