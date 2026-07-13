"""
Wide & Deep 模型训练脚本
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 将 dataset_loader 目录加入 path
_HERE = Path(__file__).parent
_PROJ_ROOT = _HERE.parent.parent.parent  # weather/
sys.path.insert(0, str(_PROJ_ROOT / "data" / "data_engineer" / "hourly_data"))
sys.path.insert(0, str(_HERE.parent))  # model_3_deep_learning/

from dataset_loader import (
    get_dataloader,
    get_dataloaders,
    config as dl_config,
)
from hourly_train.model import WideAndDeep, create_model, count_parameters


# ==================== 训练配置 ====================
OUTPUT_DIR = _HERE / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

LOG_DIR = OUTPUT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)


# ==================== 训练引擎 ====================
class Trainer:
    """Wide & Deep 训练器"""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        patience: int = 10,
        grad_clip: float = 5.0,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.criterion = nn.MSELoss()
        self.target_columns = dl_config.TARGET_COLUMNS
        self.num_targets = len(self.target_columns)
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        self.patience = patience
        self.grad_clip = grad_clip

        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.epochs_no_improve = 0
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in self.train_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()
                     if isinstance(v, torch.Tensor)}

            self.optimizer.zero_grad()
            pred = self.model(batch)          # (B, num_targets)
            target = batch['target']          # (B, num_targets)
            if target.dim() == 1:
                target = target.unsqueeze(1)

            loss = self.criterion(pred, target)
            loss.backward()

            # 梯度裁剪
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    @torch.no_grad()
    def validate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()
                     if isinstance(v, torch.Tensor)}

            pred = self.model(batch)
            target = batch['target']
            if target.dim() == 1:
                target = target.unsqueeze(1)

            loss = self.criterion(pred, target)
            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def fit(self, epochs: int = 50, log_interval: int = 2):
        print(f"\n{'='*60}")
        print(f"开始训练 — {epochs} epochs, device={self.device}")
        print(f"目标变量 ({self.num_targets}): {', '.join(self.target_columns)}")
        print(f"参数量: {count_parameters(self.model):,}")
        print(f"{'='*60}\n")

        start_time = time.time()

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch()
            val_loss = self.validate()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            self.scheduler.step(val_loss)

            if epoch % log_interval == 0:
                elapsed = time.time() - start_time
                lr = self.optimizer.param_groups[0]['lr']
                print(
                    f"Epoch {epoch:3d}/{epochs} | "
                    f"Train MSE: {train_loss:.6f} | "
                    f"Val MSE: {val_loss:.6f} | "
                    f"LR: {lr:.2e} | "
                    f"Time: {elapsed:.0f}s"
                )

            # Early stopping + checkpoint
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.epochs_no_improve = 0
                self.save_checkpoint(epoch, val_loss)
            else:
                self.epochs_no_improve += 1

            if self.epochs_no_improve >= self.patience:
                print(f"\nEarly stopping at epoch {epoch} (best: {self.best_val_loss:.6f} @ epoch {self.best_epoch})")
                break

        elapsed = time.time() - start_time
        print(f"\n训练完成. 总耗时: {elapsed:.0f}s, "
              f"Best Val MSE: {self.best_val_loss:.6f} (epoch {self.best_epoch})")

        self.save_history()

    def save_checkpoint(self, epoch: int, val_loss: float):
        path = CHECKPOINT_DIR / f"best_model.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
        }, path)
        print(f"  -> 保存 checkpoint: {path}")

    def save_history(self):
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
        }
        path = LOG_DIR / "training_history.json"
        with open(path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"训练历史已保存: {path}")


# ==================== 主入口 ====================
def main():
    parser = argparse.ArgumentParser(description="Wide & Deep 天气预测训练")
    parser.add_argument('--batch_size', type=int, default=512, help='批次大小')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--device', type=str, default='auto', help='设备 (cpu/cuda/auto)')
    parser.add_argument('--num_workers', type=int, default=0, help='数据加载线程（CPU 建议 0）')
    parser.add_argument('--prefetch_factor', type=int, default=2, help='预取倍数')
    parser.add_argument('--seq_length', type=int, default=24, help='序列长度(小时)')
    parser.add_argument('--max_samples', type=int, default=None, 
                        help='最大样本数（调试用，如 100000）')
    parser.add_argument('--no_cache', action='store_true', help='禁用缓存')
    parser.add_argument('--group_by_city', action='store_true', default=False,
                        help='按城市分组（默认为 False，提速）')
    args = parser.parse_args()

    # 设备
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # 加载数据
    dataset_kwargs = dict(
        seq_length=args.seq_length,
        use_cache=not args.no_cache,
        group_by_city=args.group_by_city,
        max_samples=args.max_samples,
    )

    loaders = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=args.prefetch_factor,
        **dataset_kwargs,
    )

    train_loader = loaders['train']
    val_loader = loaders['test']
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # 模型
    model = create_model(device=device)
    print(f"模型参数量: {count_parameters(model):,}")

    # 训练
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=args.lr,
    )
    trainer.fit(epochs=args.epochs)


if __name__ == '__main__':
    main()
