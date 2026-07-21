"""
3D 体素 JEPA 模型
"""
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange


class JEPA3D(nn.Module):
    def __init__(self, encoder, predictor, action_encoder, projector=None, pred_proj=None):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()

    def encode(self, info):
        voxel = info['voxel'].float()
        # 确保有通道维度
        if len(voxel.shape) == 5:
            voxel = voxel.unsqueeze(2)
        B, T, C, D, H, W = voxel.shape

        # 展平时间和 batch
        voxel_flat = rearrange(voxel, "b t c d h w -> (b t) c d h w")
        emb = self.encoder(voxel_flat)
        emb = self.projector(emb)
        emb = rearrange(emb, "(b t) d -> b t d", b=B, t=T)
        info["emb"] = emb

        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])
        return info

    def predict(self, emb, act_emb):
        return self.predictor(emb, act_emb)

    def forward(self, batch):
        output = self.encode(batch)
        emb = output["emb"]
        act_emb = output["act_emb"]
        T = emb.shape[1]
        ctx_emb = emb[:, :T-1]
        ctx_act = act_emb[:, :T-1]
        tgt_emb = emb[:, 1:]
        pred_emb = self.predict(ctx_emb, ctx_act)
        pred_loss = F.mse_loss(pred_emb, tgt_emb)
        return {"pred_loss": pred_loss, "emb": emb}