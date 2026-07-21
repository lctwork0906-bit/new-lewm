# core/voxel.py
"""
整合老师的 3D 体素模块
- RGBDVoxelizer: RGB-D → 3D 体素
- VoxelJEPA: 3D 体素编码 + 预测
- 碰撞检测
"""

import math
import torch
import torch.nn as nn
import numpy as np

# ============ 1. 体素规格 ============
class VoxelSpec:
    def __init__(self, z_cells=8, y_cells=24, x_cells=24, voxel_size=1.0, max_depth=20.0):
        self.z_cells = z_cells
        self.y_cells = y_cells
        self.x_cells = x_cells
        self.voxel_size = voxel_size
        self.max_depth = max_depth


# ============ 2. RGB-D → 体素转换器 ============
class RGBDVoxelizer:
    def __init__(self, spec):
        self.spec = spec
        self.z_min = -4.0  # 无人机高度范围

    def _index(self, point):
        x, y, z = point
        ix = int(math.floor(x / self.spec.voxel_size + self.spec.x_cells / 2))
        iy = int(math.floor(y / self.spec.voxel_size + self.spec.y_cells / 2))
        iz = int(math.floor((z - self.z_min) / self.spec.voxel_size))
        if not (0 <= ix < self.spec.x_cells and 0 <= iy < self.spec.y_cells and 0 <= iz < self.spec.z_cells):
            return None
        return iz, iy, ix

    def build(self, rgb, depth):
        """
        输入: rgb (H, W, 3), depth (H, W)
        输出: voxel (3, Z, Y, X)
              - 通道0: 障碍物端点
              - 通道1: 自由空间
              - 通道2: 已观察区域
        """
        grid = np.zeros((3, self.spec.z_cells, self.spec.y_cells, self.spec.x_cells), dtype=np.float32)
        endpoints = []
        
        # 采样深度图
        height, width = depth.shape
        samples = 12
        ys = np.linspace(0, height-1, samples).astype(int)
        xs = np.linspace(0, width-1, samples).astype(int)
        
        for iy in ys:
            v = (iy + 0.5 - height/2) / max(height/2, 1.0)
            for ix in xs:
                val = float(depth[iy, ix])
                if not math.isfinite(val) or val < 0.4:
                    continue
                u = (ix + 0.5 - width/2) / max(width/2, 1.0)
                # 假设水平相机
                ray = np.array([1.0, u, v], dtype=np.float32)
                ray = ray / max(np.linalg.norm(ray), 1e-6)
                visible = min(val, self.spec.max_depth)
                
                # 标记自由空间 (沿射线)
                for frac in (0.25, 0.50, 0.75):
                    idx = self._index(ray * visible * frac)
                    if idx is not None:
                        grid[1, idx[0], idx[1], idx[2]] = 1.0
                        grid[2, idx[0], idx[1], idx[2]] = 1.0
                
                # 标记障碍物端点
                if val <= self.spec.max_depth:
                    endpoint = ray * val
                    idx = self._index(endpoint)
                    if idx is not None:
                        grid[0, idx[0], idx[1], idx[2]] = 1.0
                        grid[2, idx[0], idx[1], idx[2]] = 1.0
                        endpoints.append(endpoint)
        
        # 标记无人机自身位置
        center = self._index((0.0, 0.0, 0.0))
        if center is not None:
            grid[1, center[0], center[1], center[2]] = 1.0
            grid[2, center[0], center[1], center[2]] = 1.0
        
        return torch.from_numpy(grid).float(), endpoints

    def collision_risk(self, occupied, pose, radius=1.0):
        """
        检查给定位置是否有碰撞风险
        pose: (x, y, z)
        """
        center = self._index(pose[:3])
        if center is None:
            return 0.0
        hits = 0
        r = int(radius)
        for dx in range(-r, r+1):
            for dy in range(-r, r+1):
                for dz in range(-r, r+1):
                    idx = (center[0]+dz, center[1]+dy, center[2]+dx)
                    if idx[0] < 0 or idx[0] >= self.spec.z_cells:
                        continue
                    if idx[1] < 0 or idx[1] >= self.spec.y_cells:
                        continue
                    if idx[2] < 0 or idx[2] >= self.spec.x_cells:
                        continue
                    if occupied[idx] > 0:
                        hits += 1
        return min(1.0, hits / 4.0)


# ============ 3. 3D 体素 JEPA 编码器 ============
class VoxelJEPAEncoder(nn.Module):
    def __init__(self, spec, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(3, 12, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv3d(12, 24, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool3d((2, 3, 3)),
            nn.Flatten(),
            nn.Linear(24 * 2 * 3 * 3, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, voxel):
        return self.encoder(voxel.float())