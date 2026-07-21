"""
体素可视化：显示体素网格和预测结果
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from core.voxel import VoxelSpec, RGBDVoxelizer, VoxelJEPAEncoder
from core.model_3d import JEPA3D
import cv2
import os


def visualize_voxel_grid(voxel, title="Voxel Grid"):
    """可视化 3D 体素网格"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 获取占据的体素位置
    occupied = torch.where(voxel[0] > 0.5)
    free = torch.where(voxel[1] > 0.5)
    
    # 绘制占据体素（红色）
    if len(occupied[0]) > 0:
        ax.scatter(occupied[2], occupied[1], occupied[0], 
                   c='red', marker='s', s=20, alpha=0.6, label='Occupied')
    
    # 绘制自由体素（绿色）
    if len(free[0]) > 0:
        ax.scatter(free[2], free[1], free[0], 
                   c='green', marker='s', s=10, alpha=0.3, label='Free')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    plt.show()


def visualize_prediction(model, voxel, action, save_path=None):
    """可视化预测结果"""
    model.eval()
    with torch.no_grad():
        # 编码
        batch = {'voxel': voxel.unsqueeze(0), 'action': action.unsqueeze(0)}
        output = model(batch)
        
        emb = output['emb'][0]  # (T, D)
        pred_loss = output['pred_loss']
        
        # 可视化 embedding 轨迹
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 1. 显示 embedding 的 PCA 投影
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        emb_2d = pca.fit_transform(emb.cpu().numpy())
        axes[0].plot(emb_2d[:, 0], emb_2d[:, 1], 'b-o')
        axes[0].scatter(emb_2d[0, 0], emb_2d[0, 1], c='g', s=100, label='Start')
        axes[0].scatter(emb_2d[-1, 0], emb_2d[-1, 1], c='r', s=100, label='End')
        axes[0].set_title(f'Latent Trajectory (PCA), loss={pred_loss.item():.4f}')
        axes[0].legend()
        
        # 2. 显示损失曲线
        # 简化：显示每个时间步的损失
        axes[1].bar(['Prediction'], [pred_loss.item()])
        axes[1].set_ylabel('MSE Loss')
        axes[1].set_title('Prediction Loss')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
        else:
            plt.show()


def main():
    # 加载一个样本
    data_root = "/DATA/DATANAS2/jzq26/DATA/collected_vla/"
    sample_path = os.path.join(data_root, "BrushifyUrban", "0")
    
    # 读取 RGB 和 Depth
    rgb = cv2.imread(os.path.join(sample_path, 'step_0_front.png'))
    depth = cv2.imread(os.path.join(sample_path, 'step_0_front_depth.png'), cv2.IMREAD_UNCHANGED)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    
    # 体素化
    spec = VoxelSpec()
    voxelizer = RGBDVoxelizer(spec)
    voxel, endpoints = voxelizer.build(rgb, depth)
    
    print(f"Voxel shape: {voxel.shape}")
    print(f"Endpoints: {len(endpoints)}")
    
    # 可视化体素
    visualize_voxel_grid(voxel, "RGB-D to Voxel Conversion")
    
    # 加载模型并预测
    # TODO: 加载 checkpoint
    # model = ...
    # visualize_prediction(model, voxel, action)


if __name__ == "__main__":
    main()
