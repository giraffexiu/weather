"""
生成可视化图表：训练曲线 + 预测分析
"""
import sys
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

_HERE = Path(__file__).parent
_OUTPUT = _HERE / "output"
_PLOTS = _OUTPUT / "plots"
_PLOTS.mkdir(parents=True, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
# 避免中文 warning
import warnings
warnings.filterwarnings('ignore')
sns.set_style("whitegrid")


def plot_training_curves(history_path: Path, save_path: Path):
    with open(history_path) as f:
        h = json.load(f)
    train = h['train_losses']
    val = h['val_losses']
    epochs = range(1, len(train) + 1)
    best_epoch = h.get('best_epoch', len(val))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(epochs, train, label='Train MSE', linewidth=2, color='#3498db')
    ax.plot(epochs, val, label='Val MSE', linewidth=2, color='#e74c3c')
    ax.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7, label=f'Best epoch={best_epoch}')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('MSE Loss', fontsize=12)
    ax.set_title('Wide & Deep 训练曲线 (5目标)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"训练曲线: {save_path}")


def plot_prediction_analysis(pred_csv: Path, save_path: Path):
    df = pd.read_csv(pred_csv)
    targets = ['temperature_2m', 'precipitation', 'wind_speed_10m',
               'apparent_temperature', 'relative_humidity_2m']
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

    # 只取有预测的行
    df = df.dropna(subset=[f'pred_{t}' for t in targets]).copy()

    fig, axes = plt.subplots(3, 5, figsize=(22, 12))

    for i, (col, color) in enumerate(zip(targets, colors)):
        y_true = df[col].values
        y_pred = df[f'pred_{col}'].values
        res = y_pred - y_true
        mse = float(np.mean(res ** 2))
        mae = float(np.mean(np.abs(res)))
        ss_res = np.sum(res ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = float(1 - ss_res / ss_tot if ss_tot > 0 else 0)

        short = col.replace('_', '\n')
        sample = slice(0, min(20000, len(y_true)))

        # Row 1: Pred vs True
        ax = axes[0, i]
        ax.scatter(y_true[sample], y_pred[sample], alpha=0.15, s=3, color=color)
        mn = min(y_true[sample].min(), y_pred[sample].min())
        mx = max(y_true[sample].max(), y_pred[sample].max())
        ax.plot([mn, mx], [mn, mx], 'r--', linewidth=1)
        ax.set_xlabel('True')
        ax.set_ylabel('Pred')
        ax.set_title(f'{short}', fontsize=10, fontweight='bold')
        ax.grid(alpha=0.3)

        # Row 2: Residual distribution
        ax = axes[1, i]
        q99 = np.percentile(np.abs(res), 99)
        res_clip = np.clip(res, -q99, q99)
        ax.hist(res_clip, bins=60, edgecolor='grey', alpha=0.7, color=color)
        ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5)
        ax.set_xlabel('Residual')
        ax.set_title(f'μ={res.mean():.3f}, σ={res.std():.3f}', fontsize=9)
        ax.grid(alpha=0.3)

        # Row 3: Time series sample
        ax = axes[2, i]
        ax.plot(y_true[sample], alpha=0.7, linewidth=0.6, color='#555555', label='True')
        ax.plot(y_pred[sample], alpha=0.6, linewidth=0.6, color=color, label='Pred')
        ax.set_xlabel('Sample')
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(alpha=0.3)

    plt.suptitle(f'Wide & Deep 预测分析 (N={len(df):,}, Overall MAE={mae:.4f})',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"预测分析: {save_path}")


if __name__ == '__main__':
    history_path = _OUTPUT / "logs" / "training_history.json"
    pred_csv = _OUTPUT / "predictions.csv"

    if history_path.exists():
        plot_training_curves(history_path, _PLOTS / "training_curves.png")
    else:
        print(f"跳过训练曲线: {history_path} 不存在")

    if pred_csv.exists():
        plot_prediction_analysis(pred_csv, _PLOTS / "prediction_analysis.png")
    else:
        print(f"跳过预测分析: {pred_csv} 不存在")
