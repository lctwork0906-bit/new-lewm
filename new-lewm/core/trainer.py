"""
训练器（独立实现）
"""
import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from omegaconf import OmegaConf

from data import DataStrategyRegistry
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.logging import setup_logger


class Trainer:
    def __init__(self, cfg, strategy):
        self.cfg = cfg
        self.strategy = strategy
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = setup_logger()

        self._load_data()
        self._build_model()
        self._setup_optimizer()

    def _load_data(self):
        dataset_cfg = OmegaConf.to_container(self.cfg.data.dataset, resolve=True)
        data_path = dataset_cfg.pop("name")

        self.dataset = self.strategy.load(
            data_path,
            keys_to_load=dataset_cfg.get("keys_to_load", ['pixels', 'action']),
            transform=self.strategy.get_transform(self.cfg.img_size),
            num_steps=self.cfg.data.dataset.num_steps
        )

        action_dim = self.dataset[0]["action"].shape[-1]
        self.cfg.model.action_encoder.input_dim = action_dim
        print(f"[Train] Action dimension: {action_dim}")

        train_size = int(0.9 * len(self.dataset))
        val_size = len(self.dataset) - train_size
        self.train_set, self.val_set = torch.utils.data.random_split(
            self.dataset, [train_size, val_size]
        )

        self.train_loader = DataLoader(
            self.train_set,
            batch_size=self.cfg.loader.batch_size,
            shuffle=True,
            num_workers=self.cfg.loader.num_workers,
            drop_last=True
        )
        self.val_loader = DataLoader(
            self.val_set,
            batch_size=self.cfg.loader.batch_size,
            shuffle=False,
            num_workers=self.cfg.loader.num_workers,
        )
        self.logger.info(f"Train samples: {len(self.train_set)}, Val samples: {len(self.val_set)}")

    def _build_model(self):
        model_cfg = OmegaConf.to_container(self.cfg.model, resolve=True)
        model_cfg["encoder"]["in_chans"] = self.strategy.get_input_channels()

        import hydra
        self.model = hydra.utils.instantiate(model_cfg)
        self.model = self.model.to(self.device)

        from legacy.module import SIGReg
        self.sigreg = SIGReg(**self.cfg.loss.sigreg.kwargs).to(self.device)

        self.logger.info(f"Model built with {sum(p.numel() for p in self.model.parameters())} params")

    def _setup_optimizer(self):
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.optimizer.lr,
            weight_decay=self.cfg.optimizer.weight_decay
        )

    def run(self):
        self.logger.info("Starting training...")
        for epoch in range(self.cfg.trainer.max_epochs):
            self._train_epoch(epoch)
            self._validate_epoch(epoch)

            if (epoch + 1) % 1 == 0:
                save_checkpoint(
                    self.model,
                    self.cfg.output_model_name,
                    epoch + 1,
                    self.cfg
                )
        self.logger.info("Training complete!")

    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v
                    for k, v in batch.items()}

            output = self.model(batch)
            emb = output["emb"]
            pred_loss = output["pred_loss"]

            sigreg_loss = self.sigreg(emb.transpose(0, 1))
            loss = pred_loss + self.cfg.loss.sigreg.weight * sigreg_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item(), "pred": pred_loss.item()})

        avg_loss = total_loss / len(self.train_loader)
        self.logger.info(f"Epoch {epoch+1} train loss: {avg_loss:.4f}")

    def _validate_epoch(self, epoch):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in self.val_loader:
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v
                        for k, v in batch.items()}
                output = self.model(batch)
                emb = output["emb"]
                pred_loss = output["pred_loss"]
                sigreg_loss = self.sigreg(emb.transpose(0, 1))
                loss = pred_loss + self.cfg.loss.sigreg.weight * sigreg_loss
                total_loss += loss.item()
        avg_loss = total_loss / len(self.val_loader)
        self.logger.info(f"Epoch {epoch+1} val loss: {avg_loss:.4f}")
