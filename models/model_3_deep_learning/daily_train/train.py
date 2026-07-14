"""
Wide & Deep 模型训练脚本
"""
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# 添加相对路径到dataset_loader
dataset_loader_path = Path(__file__).resolve().parent.parent.parent.parent / 'data' / 'data_engineer' / 'daily_data'
if str(dataset_loader_path) not in sys.path:
    sys.path.insert(0, str(dataset_loader_path))

from dataset_loader import get_dataloaders

# 导入本地模块
from model import WideDeepModel
from train_config import CONFIG
from utils import (
    set_seed, calculate_metrics, plot_training_curves,
    plot_prediction_analysis, save_checkpoint, print_metrics, print_final_summary
)


class WeightedMSELoss(nn.Module):
    """加权MSE损失，对不同目标使用不同权重"""
    def __init__(self):
        super().__init__()
        # 目标权重：温度(5) | 降水(3) | 风速(1)
        # 降水和风速权重更高，因为它们更难预测
        self.weights = torch.tensor([
            1.0, 1.0, 1.0, 1.0, 1.0,  # 温度类（5个）
            3.0, 3.0, 3.0,             # 降水类（3个）- 提高权重
            2.0                        # 风速类（1个）- 提高权重
        ])
    
    def forward(self, pred, target):
        """
        Args:
            pred: (B, 9) 预测值
            target: (B, 9) 目标值
        Returns:
            loss: 加权MSE损失
        """
        weights = self.weights.to(pred.device)
        squared_error = (pred - target) ** 2  # (B, 9)
        weighted_error = squared_error * weights.unsqueeze(0)  # (B, 9)
        return weighted_error.mean()


def train_one_epoch(model, train_loader, optimizer, criterion, device, log_interval):
    """训练一个epoch"""
    model.train()
    epoch_loss = 0.0
    epoch_mae = [0.0] * 9  # 9个目标
    
    for batch_idx, batch in enumerate(train_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        target = batch['target'].float()  # (B, 9)
        
        optimizer.zero_grad()
        pred = model(batch)  # (B, 9)
        loss = criterion(pred, target)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CONFIG['grad_clip_norm'])
        optimizer.step()
        
        epoch_loss += loss.item()
        with torch.no_grad():
            for i in range(9):
                epoch_mae[i] += torch.abs(pred[:, i] - target[:, i]).mean().item()
        
        if batch_idx > 0 and batch_idx % log_interval == 0:
            avg_mae = [m/(batch_idx+1) for m in epoch_mae]
            mae_str = f"T:{avg_mae[2]:.3f} P:{avg_mae[5]:.3f} W:{avg_mae[8]:.3f}"
            print(f"  [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f} | {mae_str}")
    
    return {
        'loss': epoch_loss / len(train_loader),
        'mae': [m / len(train_loader) for m in epoch_mae]
    }


def validate(model, test_loader, criterion, device):
    """验证模型"""
    model.eval()
    all_preds = []
    all_targets = []
    val_loss = 0.0
    
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            target = batch['target'].float()  # (B, 9)
            
            pred = model(batch)  # (B, 9)
            loss = criterion(pred, target)
            
            val_loss += loss.item()
            all_preds.append(pred.cpu().numpy())
            all_targets.append(target.cpu().numpy())
    
    all_preds = np.concatenate(all_preds)  # (N, 9)
    all_targets = np.concatenate(all_targets)  # (N, 9)
    
    metrics = calculate_metrics(all_preds, all_targets)
    metrics['loss'] = val_loss / len(test_loader)
    metrics['predictions'] = all_preds
    metrics['targets'] = all_targets
    
    return metrics


def train_model(config):
    """主训练流程"""
    # 设置随机种子
    set_seed(config['seed'])
    
    device = config['device']
    print(f"\n使用设备: {device}")
    
    # 加载数据
    print("\n加载数据...")
    loaders = get_dataloaders(batch_size=config['batch_size'])
    train_loader = loaders['train']
    test_loader = loaders['test']
    print(f"训练样本: {len(train_loader.dataset):,}, 测试样本: {len(test_loader.dataset):,}")
    
    # 创建模型
    print("\n初始化模型...")
    model = WideDeepModel(config).to(device)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数: {total_params:,}, 可训练: {trainable_params:,}")
    
    # 优化器
    criterion = WeightedMSELoss()  # 使用加权损失函数
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # 学习率调度器
    if config['use_scheduler']:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['T_max']
        )
    
    # 训练历史
    history = {
        'train_loss': [],
        'val_loss': [], 
        'val_mae': [[] for _ in range(9)],   # 9个目标
        'val_rmse': [[] for _ in range(9)],  # 9个目标
        'val_r2': [[] for _ in range(9)]     # 9个目标
    }
    
    best_rmse = float('inf')
    best_epoch = 0
    patience_counter = 0
    
    print("\n" + "="*60)
    print("开始训练".center(60))
    print("="*60)
    
    # 训练循环
    for epoch in range(1, config['epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['epochs']}")
        print("-" * 40)
        
        # 训练
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, config['log_interval']
        )
        
        # 验证
        val_metrics = validate(model, test_loader, criterion, device)
        
        # 学习率调度
        if config['use_scheduler']:
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
        else:
            current_lr = config['learning_rate']
        
        # 记录历史
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        for i in range(9):
            history['val_mae'][i].append(val_metrics['mae'][i])
            history['val_rmse'][i].append(val_metrics['rmse'][i])
            history['val_r2'][i].append(val_metrics['r2'][i])
        
        # 打印结果
        print(f"\nTrain: Loss: {train_metrics['loss']:.4f}")
        print(f"Val:   ", end="")
        print_metrics(val_metrics)
        print(f"LR: {current_lr:.6f}")
        
        # 保存最佳模型 (使用平均温度的RMSE作为主要指标)
        if val_metrics['rmse'][2] < best_rmse:  # temperature_2m_mean是第3个目标
            best_rmse = val_metrics['rmse'][2]
            best_epoch = epoch
            patience_counter = 0
            
            save_checkpoint(
                model, optimizer, epoch, best_rmse, config, config['checkpoint_path']
            )
            print(f"✓ 最佳模型 (T_mean RMSE: {best_rmse:.4f})")
        else:
            patience_counter += 1
        
        # 早停
        if config['early_stopping'] and patience_counter >= config['patience']:
            print(f"\n早停触发! 最佳: Epoch {best_epoch}, T_mean RMSE {best_rmse:.4f}")
            break
    
    print("\n" + "="*60)
    print("训练完成".center(60))
    print("="*60)
    print(f"最佳模型: Epoch {best_epoch}, T_mean RMSE {best_rmse:.4f}")
    
    # 加载最佳模型 (PyTorch 2.6需要weights_only=False)
    checkpoint = torch.load(config['checkpoint_path'], map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 最终评估
    print("\n最终评估...")
    final_metrics = validate(model, test_loader, criterion, device)
    print_final_summary(final_metrics)
    
    # 可视化
    print("\n生成可视化...")
    plot_training_curves(history, config['plot_dir'] / 'training_curves.png')
    print(f"✓ 训练曲线: {config['plot_dir']}/training_curves.png")
    
    plot_prediction_analysis(
        final_metrics['predictions'],
        final_metrics['targets'],
        config['plot_dir'] / 'prediction_analysis.png'
    )
    print(f"✓ 预测分析: {config['plot_dir']}/prediction_analysis.png")
    
    return model, history, final_metrics


def main():
    """主函数"""
    try:
        model, history, metrics = train_model(CONFIG)
        print("\n训练成功!")
    except KeyboardInterrupt:
        print("\n训练被中断")
    except Exception as e:
        print(f"\n训练出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
