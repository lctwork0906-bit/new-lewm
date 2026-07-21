import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from ..base import DataFormatStrategy
from ..transforms.voxelize import VoxelizeTransform


class PNGDepthDataset(Dataset):
    def __init__(self, root_dir, num_steps=1, transform=None):
        self.root_dir = root_dir
        self.num_steps = num_steps
        self.transform = transform
        self._load_metadata()

    def _load_metadata(self):
        self.tasks = []
        for scene in os.listdir(self.root_dir):
            scene_path = os.path.join(self.root_dir, scene)
            if os.path.isdir(scene_path):
                for task in os.listdir(scene_path):
                    task_path = os.path.join(scene_path, task)
                    if os.path.isdir(task_path):
                        if os.path.exists(os.path.join(task_path, 'step_0_front.png')):
                            self.tasks.append(task_path)
        print(f'Found {len(self.tasks)} tasks with RGB-D data')

    def __len__(self):
        return len(self.tasks)

    def _compute_collision_risk(self, task_path):
        """从深度图计算碰撞风险：深度越小，风险越高"""
        depth_path = os.path.join(task_path, 'step_0_front_depth.png')
        if os.path.exists(depth_path):
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth is not None:
                # 深度图归一化：假设 max_depth=20m
                mean_depth = np.mean(depth) / 1000.0  # mm -> m
                # 风险 = 1 - min(深度/安全距离, 1)
                # 安全距离设为 5m
                risk = 1.0 - min(mean_depth / 5.0, 1.0)
                return torch.tensor(risk).float()
        return torch.tensor(0.0).float()

    def __getitem__(self, idx):
        task_path = self.tasks[idx]
        angles = ['front', 'left', 'right', 'down']
        
        all_pixels = []
        all_depths = []
        for angle in angles:
            rgb_path = os.path.join(task_path, f'step_0_{angle}.png')
            depth_path = os.path.join(task_path, f'step_0_{angle}_depth.png')
            
            rgb = cv2.imread(rgb_path)
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            
            if rgb is not None and depth is not None:
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                all_pixels.append(rgb)
                all_depths.append(depth)
        
        pixels = np.stack(all_pixels)
        depths = np.stack(all_depths)
        
        pixels = torch.from_numpy(pixels).permute(0, 3, 1, 2).float()
        depths = torch.from_numpy(depths).float()
        
        # 计算碰撞风险
        collision_risk = self._compute_collision_risk(task_path)
        
        data = {
            'pixels': pixels,
            'depth': depths,
            'action': torch.zeros(1, 2),
            'collision_risk': collision_risk,  # 加入碰撞风险
        }
        
        if self.transform:
            data = self.transform(data)
        return data


class PNGDepthStrategy(DataFormatStrategy):
    def detect(self, path: str) -> bool:
        return os.path.isdir(path) and os.path.exists(os.path.join(path, 'BrushifyUrban'))

    def load(self, path: str, **kwargs) -> Dataset:
        transform = kwargs.get('transform', None)
        return PNGDepthDataset(path, transform=transform)

    def get_input_channels(self) -> int:
        return 3

    def get_transform(self, img_size):
        return VoxelizeTransform()

    def get_column_names(self) -> list:
        return ['pixels', 'depth', 'action', 'collision_risk']

    def get_name(self) -> str:
        return "PNG-RGBD"
