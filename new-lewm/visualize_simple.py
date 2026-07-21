"""
简单体素可视化：显示体素的 2D 切片和深度图
"""
import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from core.voxel import VoxelSpec, RGBDVoxelizer
from core.model_3d import JEPA3D
from core.voxel import VoxelJEPAEncoder
from legacy.module import ARPredictor, Embedder, MLP


def load_model(ckpt_path):
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
    return model


def load_sample(scene='BrushifyUrban', task='0'):
    base_path = f'/DATA/DATANAS2/jzq26/DATA/collected_vla/{scene}/{task}/'
    rgb_path = os.path.join(base_path, 'step_0_front.png')
    depth_path = os.path.join(base_path, 'step_0_front_depth.png')
    
    rgb = cv2.imread(rgb_path)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    
    if rgb is None or depth is None:
        return None, None
    
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    return rgb, depth


def main():
    # 加载模型
    ckpt_path = '/villa/lct25-srt/.stable-wm/checkpoints/lewm_voxel/weights_epoch_46.pt'
    model = load_model(ckpt_path)
    print('✅ Model loaded')
    
    # 加载数据
    rgb, depth = load_sample('BrushifyUrban', '0')
    if rgb is None:
        print('❌ Failed to load sample')
        return
    
    # 体素化
    spec = VoxelSpec()
    voxelizer = RGBDVoxelizer(spec)
    voxel, endpoints = voxelizer.build(rgb, depth)
    print(f'✅ Voxel shape: {voxel.shape}')
    
    # 创建可视化
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # 1. RGB
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title('RGB Image')
    axes[0, 0].axis('off')
    
    # 2. Depth
    axes[0, 1].imshow(depth, cmap='plasma')
    axes[0, 1].set_title('Depth Map')
    axes[0, 1].axis('off')
    
    # 3-4. 体素通道 0 和 1 的切片（中间 Z 层）
    slice_z = voxel.shape[1] // 2
    
    ax = axes[0, 2]
    im = ax.imshow(voxel[0, slice_z, :, :].numpy(), cmap='Reds', vmin=0, vmax=1)
    ax.set_title(f'Occupied (Z={slice_z})')
    ax.axis('off')
    plt.colorbar(im, ax=ax)
    
    ax = axes[0, 3]
    im = ax.imshow(voxel[1, slice_z, :, :].numpy(), cmap='Greens', vmin=0, vmax=1)
    ax.set_title(f'Free (Z={slice_z})')
    ax.axis('off')
    plt.colorbar(im, ax=ax)
    
    # 第二行：预测结果
    voxel_tensor = voxel.unsqueeze(0)
    action = torch.randn(1, 1, 2)
    batch = {'voxel': voxel_tensor, 'action': action}
    
    with torch.no_grad():
        output = model(batch)
        pred_loss = output['pred_loss'].item()
    
    # 5. 预测损失
    axes[1, 0].text(0.5, 0.5, f'Prediction Loss: {pred_loss:.4f}', 
                    ha='center', va='center', fontsize=14)
    axes[1, 0].set_title('Model Prediction')
    axes[1, 0].axis('off')
    
    # 6. 端点数
    axes[1, 1].text(0.5, 0.5, f'Endpoints: {len(endpoints)}', 
                    ha='center', va='center', fontsize=14)
    axes[1, 1].set_title('Voxel Statistics')
    axes[1, 1].axis('off')
    
    # 7. 体素占用统计
    occupied = (voxel[0] > 0.5).sum().item()
    free = (voxel[1] > 0.5).sum().item()
    observed = (voxel[2] > 0.5).sum().item()
    axes[1, 2].bar(['Occupied', 'Free', 'Observed'], [occupied, free, observed], 
                   color=['red', 'green', 'blue'])
    axes[1, 2].set_title('Voxel Statistics')
    axes[1, 2].set_ylabel('Count')
    axes[1, 2].tick_params(axis='x', rotation=15)
    
    # 8. 嵌入形状
    axes[1, 3].text(0.5, 0.5, f'Emb Shape: {output["emb"].shape}', 
                    ha='center', va='center', fontsize=14)
    axes[1, 3].set_title('Latent Embedding')
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig('/villa/lct25-srt/voxel_simple.png', dpi=150, bbox_inches='tight')
    print('✅ Saved to /villa/lct25-srt/voxel_simple.png')
    plt.show()


if __name__ == '__main__':
    main()
