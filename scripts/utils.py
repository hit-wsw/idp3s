# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import random
import re
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal
from hyperparameters import *


class eval_mode:
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False
    
def gen_pcd(color_jpg_path, depth_png_path, num_points_to_sample, gen_table, threshold):
    # process pointcloud
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

    if gen_table:
        plane_file = "table_plane/table.txt"
        with open(plane_file, 'r') as f:
            line = f.readline().strip()
            plane = np.array([float(x) for x in line.split()])
        color_pcd, table_pcd = filter_points_by_plane(color_pcd, plane, threshold)
        
        num_table_points = int(0.05 * len(table_pcd))
        if num_table_points > 0:
            table_indices = np.random.choice(len(table_pcd), num_table_points, replace=False)
            color_pcd = np.concatenate((color_pcd, table_pcd[table_indices]))
    uncolored_pcd = color_pcd[:, :3] 

    if len(color_pcd) > num_points_to_sample:
        indices = np.random.choice(len(color_pcd), num_points_to_sample, replace=False)
        color_pcd = color_pcd[indices]
        uncolored_pcd = uncolored_pcd[indices]
    
    return color_pcd, uncolored_pcd

def resize_image(image_path, size):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image, (size, size))
    return image_resized

def filter_points_by_plane(pcd, plane, threshold):
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
    color_pcd = pcd[mask]
    table_pcd = pcd[mask1]
    
    return color_pcd, table_pcd

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)


def to_torch(xs, device):
    return tuple(torch.as_tensor(x, device=device) for x in xs)


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


class Until:
    def __init__(self, until, action_repeat=1):
        self._until = until
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._until is None:
            return True
        until = self._until // self._action_repeat
        return step < until


class Every:
    def __init__(self, every, action_repeat=1):
        self._every = every
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._every is None:
            return False
        every = self._every // self._action_repeat
        if step % every == 0:
            return True
        return False


class Timer:
    def __init__(self):
        self._start_time = time.time()
        self._last_time = time.time()

    def reset(self):
        elapsed_time = time.time() - self._last_time
        self._last_time = time.time()
        total_time = time.time() - self._start_time
        return elapsed_time, total_time

    def total_time(self):
        return time.time() - self._start_time


class TruncatedNormal(pyd.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape,
                               dtype=self.loc.dtype,
                               device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)


def schedule(schdl, step):
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r'linear\((.+),(.+),(.+)\)', schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
        match = re.match(r'step_linear\((.+),(.+),(.+),(.+),(.+)\)', schdl)
        if match:
            init, final1, duration1, final2, duration2 = [
                float(g) for g in match.groups()
            ]
            if step <= duration1:
                mix = np.clip(step / duration1, 0.0, 1.0)
                return (1.0 - mix) * init + mix * final1
            else:
                mix = np.clip((step - duration1) / duration2, 0.0, 1.0)
                return (1.0 - mix) * final1 + mix * final2
    raise NotImplementedError(schdl)
