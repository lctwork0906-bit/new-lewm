"""
3D体素可视化：展示RGB-D → 体素转换 + 模型预测
"""
import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from core.voxel import VoxelSpec, RGBDVoxelizer, VoxelJEPAEncoder
from core.model_3d import JEPA3D
from legacy.module import ARPredictor, Embedder, MLP


def load_model(ckpt_path):
    """加载训练好的模型"""
    spec = VoxelSpec()
    encoder = VoxelJEPAEncoder(spec, latent_dim=64)
    
    predictor = ARPredictor(
        num_frames=3,
        input_dim=64,
        hidden_dim=64,
        output_dim=64,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dim_head=64,
        dropout=0.1,
        emb_dropout=0.0,
    )
    
    action_encoder = Embedder(input_dim=2, smoothed_dim=10, emb_dim=64, mlp_scale=4)
    projector = MLP(input_dim=64, output_dim=64, hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d)
    pred_proj = MLP(input_dim=64, output_dim=64, hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d)
    
    model = JEPA3D(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=pred_proj,
    )
    
    state_dict = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, spec


def load_sample(scene='BrushifyUrban', task='0'):
    """加载一个样本的RGB-D数据"""
    base_path = f'/DATA/DATANAS2/jzq26/DATA/collected_vla/{scene}/{task}/'
    
    rgb_path = os.path.join(base_path, 'step_0_front.png')
    depth_path = os.path.join(base_path, 'step_0_front_depth.png')
    
    rgb = cv2.imread(rgb_path)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    
    if rgb is None or depth is None:
        return None, None
    
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    return rgb, depth


def visualize_voxel_3d(voxel, ax=None, title="3D Voxel Grid"):
    """可视化3D体素网格"""
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    
    # 获取占据体素的位置
    occupied = torch.where(voxel[0] > 0.5)
    free = torch.where(voxel[1] > 0.5)
    
    # 绘制占据体素（红色）
    if len(occupied[0]) > 0:
        ax.scatter(occupied[2].cpu().numpy(), 
                   occupied[1].cpu().numpy(), 
                   occupied[0].cpu().numpy(), 
                   c='red', marker='s', s=15, alpha=0.8, label='Occupied')
    
    # 绘制自由体素（绿色，半透明）
    if len(free[0]) > 0:
        ax.scatter(free[2].cpu().numpy(), 
                   free[1].cpu().numpy(), 
                   free[0].cpu().numpy(), 
                   c='green', marker='s', s=8, alpha=0.3, label='Free')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    return ax


def visualize_multi_scene(scenes, model):
    """可视化多个场景的体素转换结果"""
    fig = plt.figure(figsize=(18, 12))
    
    for idx, scene in enumerate(scenes[:6]):
        rgb, depth = load_sample(scene, '0')
        if rgb is None:
            continue
        
        # 体素化
        spec = VoxelSpec()
        voxelizer = RGBDVoxelizer(spec)
        voxel, endpoints = voxelizer.build(rgb, depth)
        
        # 模型预测
        voxel_tensor = voxel.unsqueeze(0)
        action = torch.randn(1, 1, 2)
        batch = {'voxel': voxel_tensor, 'action': action}
        
        with torch.no_grad():
            output = model(batch)
            pred_loss = output['pred_loss'].item()
        
        # 子图：RGB图像
        ax1 = fig.add_subplot(3, 6, idx + 1)
        ax1.imshow(rgb)
        ax1.set_title(f'{scene}')
        ax1.axis('off')
        
        # 子图：体素3D
        ax2 = fig.add_subplot(3, 6, idx + 7, projection='3d')
        visualize_voxel_3d(voxel, ax2, title=f'Voxel, loss={pred_loss:.3f}')
    
    plt.tight_layout()
    plt.savefig('/villa/lct25-srt/voxel_visualization.png', dpi=150, bbox_inches='tight')
    print('✅ Saved to /villa/lct25-srt/voxel_visualization.png')


def main():
    # 加载模型
    ckpt_path = '/villa/lct25-srt/.stable-wm/checkpoints/lewm_voxel/weights_epoch_46.pt'
    model, spec = load_model(ckpt_path)
    print(f'✅ Model loaded')
    
    scenes = ['BrushifyUrban', 'CabinLake', 'CityPark', 'DownTown', 'Neighborhood', 'Slum']
    
    visualize_multi_scene(scenes, model)
    
    # 单场景详细可视化
    rgb, depth = load_sample('BrushifyUrban', '0')
    if rgb is not None:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        axes[0, 0].imshow(rgb)
        axes[0, 0].set_title('RGB Image')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(depth, cmap='plasma')
        axes[0, 1].set_title('Depth Map')
        axes[0, 1].axis('off')
        
        spec = VoxelSpec()
        voxelizer = RGBDVoxelizer(spec)
        voxel, endpoints = voxelizer.build(rgb, depth)
        
        ax = axes[0, 2]
        visualize_voxel_3d(voxel, ax, title='3D Voxel Grid')
        
        for c in range(3):
            ax = axes[1, c]
            slice_z = voxel.shape[1] // 2
            im = ax.imshow(voxel[c, slice_z, :, :].numpy(), cmap='gray')
            ax.set_title(f'Channel {c}, Z={slice_z}')
            ax.axis('off')
            plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        plt.savefig('/villa/lct25-srt/voxel_detailed.png', dpi=150, bbox_inches='tight')
        print('✅ Saved to /villa/lct25-srt/voxel_detailed.png')


if __name__ == '__main__':
    main()
