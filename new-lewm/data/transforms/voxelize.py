# data/transforms/voxelize.py
from core.voxel import RGBDVoxelizer, VoxelSpec

class VoxelizeTransform:
    def __init__(self, spec=None):
        self.spec = spec or VoxelSpec()
        self.voxelizer = RGBDVoxelizer(self.spec)
    
    def __call__(self, data):
        # 输入: data 包含 'rgb' 和 'depth'
        # 输出: data 包含 'voxel'
        voxel, _ = self.voxelizer.build(data['rgb'], data['depth'])
        data['voxel'] = voxel
        return data