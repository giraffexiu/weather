"""
模型评估脚本
"""
import sys
import torch
import numpy as np
from pathlib import Path

# 添加dataset_loader路径
dataset_loader_path = Path(__file__).resolve().parent.parent.parent.parent / 'data' / 'data_engineer' / 'daily_data'
if str(dataset_loader_path) not in sys.path:
    sys.path.insert(0, str(dataset_loader_path))

from dataset_loader import get_dataloaders
from model import WideDeepModel
from train_config import CONFIG
from utils import calculate_metrics, print_final_summary, plot_prediction_analysis, set_seed

def evaluate_model():
    """评估已保存的最佳模型"""
    
    set_seed(CONFIG['seed'])
    device = CONFIG['device']
    
    print("\n" + "="*60)
    print("模型评估".center(60))
    print("="*60)
    
    # 加载数据
    print("\n加载测试数据...")
    loaders = get_dataloaders(batch_size=CONFIG['batch_size'])
    test_loader = loaders['test']
    print(f"测试样本: {len(test_loader.dataset):,}")
    
    # 加载模型
    print("\n加载最佳模型...")
    model = WideDeepModel(CONFIG).to(device)
    checkpoint = torch.load(CONFIG['checkpoint_path'], map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ 加载Epoch {checkpoint['epoch']}的模型")
    print(f"训练时最佳RMSE: {checkpoint['best_rmse']:.4f}°C")
    
    # 评估
    print("\n在测试集上评估...")
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            target = batch['target'].float()  # (B, 9)
            
            pred = model(batch)  # (B, 9)
            
            all_preds.append(pred.cpu().numpy())
            all_targets.append(target.cpu().numpy())
    
    all_preds = np.concatenate(all_preds)  # (N, 9)
    all_targets = np.concatenate(all_targets)  # (N, 9)
    
    # 计算指标
    metrics = calculate_metrics(all_preds, all_targets)
    metrics['predictions'] = all_preds
    metrics['targets'] = all_targets
    
    # 打印结果
    print_final_summary(metrics)
    
    # 生成分析图
    print("\n生成预测分析图...")
    plot_prediction_analysis(
        metrics['predictions'],
        metrics['targets'],
        CONFIG['plot_dir'] / 'prediction_analysis.png'
    )
    print(f"✓ 已保存: {CONFIG['plot_dir']}/prediction_analysis.png")
    
    return metrics

if __name__ == '__main__':
    try:
        metrics = evaluate_model()
        print("\n✅ 评估完成!")
    except Exception as e:
        print(f"\n❌ 评估出错: {e}")
        import traceback
        traceback.print_exc()
