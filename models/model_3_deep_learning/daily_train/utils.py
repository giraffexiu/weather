"""
工具函数：评估指标与可视化
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from pathlib import Path
from scipy import stats

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


def set_seed(seed):
    """设置随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_metrics(predictions, targets):
    """
    计算多目标回归评估指标
    
    Args:
        predictions: numpy array (N, 3) - [temperature, precipitation, wind_speed]
        targets: numpy array (N, 3)
    
    Returns:
        dict: 评估指标 (每个目标一个列表)
    """
    num_targets = predictions.shape[1]
    mae_list = []
    mse_list = []
    rmse_list = []
    r2_list = []
    
    for i in range(num_targets):
        pred_i = predictions[:, i]
        target_i = targets[:, i]
        
        mae = np.abs(pred_i - target_i).mean()
        mse = ((pred_i - target_i) ** 2).mean()
        rmse = np.sqrt(mse)
        
        ss_res = ((target_i - pred_i) ** 2).sum()
        ss_tot = ((target_i - target_i.mean()) ** 2).sum()
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        mae_list.append(mae)
        mse_list.append(mse)
        rmse_list.append(rmse)
        r2_list.append(r2)
    
    return {
        'mae': mae_list,
        'mse': mse_list,
        'rmse': rmse_list,
        'r2': r2_list
    }


def plot_training_curves(history, save_path):
    """绘制9目标训练曲线（仅显示关键目标）"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], label='Train', linewidth=2, color='#3498db')
    ax.plot(epochs, history['val_loss'], label='Val', linewidth=2, color='#e74c3c')
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel('MSE Loss', fontsize=10)
    ax.set_title('Loss', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    
    # 关键目标的RMSE和R²
    key_targets = [(2, 'T_mean', '#e74c3c'), (5, 'Precip', '#3498db'), (8, 'Wind', '#2ecc71')]
    
    for idx, (i, name, color) in enumerate(key_targets):
        row = (idx + 1) // 2
        col = (idx + 1) % 2
        ax = axes[row, col]
        
        ax.plot(epochs, history['val_rmse'][i], linewidth=2, color=color, label='RMSE')
        ax2 = ax.twinx()
        ax2.plot(epochs, history['val_r2'][i], linewidth=2, color='#f39c12', linestyle='--', label='R²')
        
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel('RMSE', fontsize=10, color=color)
        ax2.set_ylabel('R²', fontsize=10, color='#f39c12')
        ax.set_title(f'{name}', fontsize=11, fontweight='bold')
        ax.tick_params(axis='y', labelcolor=color)
        ax2.tick_params(axis='y', labelcolor='#f39c12')
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_prediction_analysis(predictions, targets, save_path):
    """绘制9目标预测分析图（仅显示关键目标）"""
    target_names = ['T_mean', 'Precip', 'Wind']
    key_indices = [2, 5, 8]  # temperature_2m_mean, precipitation_sum, wind_speed_10m_max
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 13))
    
    for plot_idx, (i, name, color) in enumerate(zip(key_indices, target_names, colors)):
        pred_i = predictions[:, i]
        target_i = targets[:, i]
        residuals = pred_i - target_i
        
        # 预测vs真实
        ax = axes[plot_idx, 0]
        ax.scatter(target_i, pred_i, alpha=0.3, s=4, color=color)
        min_val = min(target_i.min(), pred_i.min())
        max_val = max(target_i.max(), pred_i.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5, label='y=x')
        ax.set_xlabel(f'True {name}', fontsize=9)
        ax.set_ylabel(f'Pred {name}', fontsize=9)
        ax.set_title(f'{name}: Pred vs True', fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        
        # 残差分布
        ax = axes[plot_idx, 1]
        ax.hist(residuals, bins=50, edgecolor='black', alpha=0.7, color=color)
        ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5)
        ax.set_xlabel('Residual', fontsize=9)
        ax.set_ylabel('Frequency', fontsize=9)
        ax.set_title(f'{name}: Residual (μ={residuals.mean():.3f})', fontsize=10, fontweight='bold')
        ax.grid(alpha=0.3)
        
        # 残差vs预测值
        ax = axes[plot_idx, 2]
        ax.scatter(pred_i, residuals, alpha=0.3, s=4, color=color)
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
        ax.set_xlabel(f'Pred {name}', fontsize=9)
        ax.set_ylabel('Residual', fontsize=9)
        ax.set_title(f'{name}: Residual Plot', fontsize=10, fontweight='bold')
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_checkpoint(model, optimizer, epoch, best_rmse, config, path):
    """保存模型检查点"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_rmse': best_rmse,
        'config': config
    }, path)


def load_checkpoint(model, optimizer, path, device):
    """加载模型检查点"""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['best_rmse']


def print_metrics(metrics):
    """打印9个目标的评估指标（精简版）"""
    target_names = ['T_max', 'T_min', 'T_mean', 'T_range', 'T_feel', 'Precip', 'Rain', 'Snow', 'Wind']
    
    # 只显示关键目标的指标
    key_indices = [2, 5, 8]  # T_mean, Precip, Wind
    key_names = [target_names[i] for i in key_indices]
    
    mae_str = " | ".join([f"{key_names[j]} MAE:{metrics['mae'][key_indices[j]]:.3f}" for j in range(3)])
    r2_str = " | ".join([f"{key_names[j]} R²:{metrics['r2'][key_indices[j]]:.3f}" for j in range(3)])
    print(f"{mae_str}\n       {r2_str}")


def print_final_summary(metrics):
    """打印9个目标的最终评估总结"""
    target_names = [
        'Temperature Max (°C)',
        'Temperature Min (°C)', 
        'Temperature Mean (°C)',
        'Temperature Range (°C)',
        'Feels Like (°C)',
        'Precipitation (mm)',
        'Rain (mm)',
        'Snow (mm)',
        'Wind Speed (m/s)'
    ]
    
    print("\n" + "="*70)
    print("最终评估结果 (9目标预测)".center(70))
    print("="*70)
    
    for i, name in enumerate(target_names):
        status = "✓" if metrics['r2'][i] >= 0.85 else "⚠"
        print(f"{status} {name:25s} MAE:{metrics['mae'][i]:6.3f} | RMSE:{metrics['rmse'][i]:6.3f} | R²:{metrics['r2'][i]:6.3f}")
    
    print("="*70)
