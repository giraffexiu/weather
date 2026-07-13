"""
集成模型评估脚本
评估单模型和集成模型的性能
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, mean_absolute_error,
    mean_squared_error, r2_score
)

# 添加路径
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    TEST_DATA_PATH,
    RESULTS_DIR,
    DEVICE,
    setup_paths,
    validate_config,
    PROBABILITY_CONVERSION_CONFIG,
    MODEL1_CLASSIFICATION_TASKS,
    MODEL1_REGRESSION_TASKS,
    ENSEMBLE_TASKS
)
from model_wrapper import Model1Wrapper, Model3Wrapper
from probability_converter import ProbabilityConverter
from soft_voting_ensemble import SoftVotingEnsemble


def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """计算回归指标"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # MAPE (避免除零)
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape
    }


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray
) -> dict:
    """计算分类指标"""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_prob)
    }


def evaluate_models():
    """评估所有模型"""
    print("\n" + "="*70)
    print("Daily Ensemble 模型评估")
    print("="*70)
    
    # 设置路径和验证配置
    setup_paths()
    validate_config()
    
    print(f"\n使用设备: {DEVICE}")
    
    # 加载测试数据
    print("\n加载测试数据...")
    test_df = pd.read_csv(TEST_DATA_PATH)
    print(f"  测试集大小: {len(test_df):,} 样本")
    
    # 初始化模型
    print("\n初始化模型...")
    model1 = Model1Wrapper()
    
    converter = ProbabilityConverter(PROBABILITY_CONVERSION_CONFIG)
    model3 = Model3Wrapper(probability_converter=converter)
    
    # 初始化集成器
    ensemble = SoftVotingEnsemble(
        model1_wrapper=model1,
        model3_wrapper=model3,
        weight_method='performance_based',
        verbose=True
    )
    
    # 准备 Model 1 的 DataLoader
    print("\n准备 Model 1 DataLoader...")
    from dataset_loader import get_dataloaders
    loaders_model1 = get_dataloaders(batch_size=128)
    test_loader_model1 = loaders_model1['test']
    print(f"  测试批次数: {len(test_loader_model1)}")
    
    # 准备 Model 3 的 DataLoader
    print("\n准备 Model 3 DataLoader...")
    loaders_model3 = get_dataloaders(batch_size=128)
    test_loader_model3 = loaders_model3['test']
    print(f"  测试批次数: {len(test_loader_model3)}")
    
    # 执行预测
    print("\n" + "="*70)
    print("执行预测")
    print("="*70)
    
    predictions = ensemble.predict(test_loader_model1, test_loader_model3)
    
    # 评估结果
    print("\n" + "="*70)
    print("评估结果")
    print("="*70)
    
    results = {
        'regression': {},
        'classification': {}
    }
    
    # 评估回归任务
    print("\n【回归任务评估】")
    print("-"*70)
    
    for task_name, task_config in ENSEMBLE_TASKS['regression'].items():
        model1_key = task_config['model1_key']
        target_column = MODEL1_REGRESSION_TASKS[model1_key]['target_column']
        
        if target_column not in test_df.columns:
            print(f"  ⚠ 跳过 {task_name}: 目标列 {target_column} 不存在")
            continue
        
        y_true = test_df[target_column].values
        
        # Model 1 预测
        y_pred_model1 = predictions['individual']['model1']['regression'][model1_key]
        metrics_model1 = calculate_regression_metrics(y_true, y_pred_model1)
        
        # Model 3 预测
        model3_target = list(MODEL3Wrapper(converter).model.preprocessor.num_idx.keys())[
            task_config['model3_index']
        ] if hasattr(Model3Wrapper(converter).model, 'preprocessor') else None
        
        # 简化：直接从预测结果获取
        model3_targets = list(predictions['individual']['model3']['regression'].keys())
        y_pred_model3 = predictions['individual']['model3']['regression'][
            model3_targets[task_config['model3_index']]
        ]
        metrics_model3 = calculate_regression_metrics(y_true, y_pred_model3)
        
        # 集成预测
        y_pred_ensemble = predictions['regression'][task_name]
        metrics_ensemble = calculate_regression_metrics(y_true, y_pred_ensemble)
        
        # 保存结果
        results['regression'][task_name] = {
            'model1': metrics_model1,
            'model3': metrics_model3,
            'ensemble': metrics_ensemble
        }
        
        # 打印结果
        print(f"\n{task_name.upper()}")
        print(f"  Model 1    | MAE: {metrics_model1['mae']:.4f} | "
              f"RMSE: {metrics_model1['rmse']:.4f} | R²: {metrics_model1['r2']:.4f}")
        print(f"  Model 3    | MAE: {metrics_model3['mae']:.4f} | "
              f"RMSE: {metrics_model3['rmse']:.4f} | R²: {metrics_model3['r2']:.4f}")
        print(f"  Ensemble   | MAE: {metrics_ensemble['mae']:.4f} | "
              f"RMSE: {metrics_ensemble['rmse']:.4f} | R²: {metrics_ensemble['r2']:.4f}")
        
        # 计算改进
        improvement_r2 = metrics_ensemble['r2'] - max(metrics_model1['r2'], metrics_model3['r2'])
        improvement_mae = min(metrics_model1['mae'], metrics_model3['mae']) - metrics_ensemble['mae']
        
        if improvement_r2 > 0 or improvement_mae > 0:
            print(f"  ✓ 改进    | R²: {improvement_r2:+.4f} | MAE: {improvement_mae:+.4f}")
        else:
            print(f"  - 改进    | R²: {improvement_r2:+.4f} | MAE: {improvement_mae:+.4f}")
    
    # 评估分类任务
    print("\n" + "-"*70)
    print("\n【分类任务评估】")
    print("-"*70)
    
    for task_name, task_config in ENSEMBLE_TASKS['classification'].items():
        model1_key = task_config['model1_key']
        target_column = MODEL1_CLASSIFICATION_TASKS[model1_key]['target_column']
        
        if target_column not in test_df.columns:
            print(f"  ⚠ 跳过 {task_name}: 目标列 {target_column} 不存在")
            continue
        
        y_true = test_df[target_column].values
        
        # Model 1 预测
        y_pred_model1 = predictions['individual']['model1']['classification'][model1_key]['prediction']
        y_prob_model1 = predictions['individual']['model1']['classification'][model1_key]['probability']
        metrics_model1 = calculate_classification_metrics(y_true, y_pred_model1, y_prob_model1)
        
        # Model 3 预测
        y_prob_model3 = predictions['individual']['model3']['classification'][task_name]['probability']
        y_pred_model3 = (y_prob_model3 >= 0.5).astype(int)
        metrics_model3 = calculate_classification_metrics(y_true, y_pred_model3, y_prob_model3)
        
        # 集成预测
        y_pred_ensemble = predictions['classification'][task_name]['prediction']
        y_prob_ensemble = predictions['classification'][task_name]['probability']
        metrics_ensemble = calculate_classification_metrics(y_true, y_pred_ensemble, y_prob_ensemble)
        
        # 保存结果
        results['classification'][task_name] = {
            'model1': metrics_model1,
            'model3': metrics_model3,
            'ensemble': metrics_ensemble
        }
        
        # 打印结果
        print(f"\n{task_name.upper()}")
        print(f"  Model 1    | F1: {metrics_model1['f1']:.4f} | "
              f"AUC: {metrics_model1['roc_auc']:.4f} | "
              f"Acc: {metrics_model1['accuracy']:.4f}")
        print(f"  Model 3    | F1: {metrics_model3['f1']:.4f} | "
              f"AUC: {metrics_model3['roc_auc']:.4f} | "
              f"Acc: {metrics_model3['accuracy']:.4f}")
        print(f"  Ensemble   | F1: {metrics_ensemble['f1']:.4f} | "
              f"AUC: {metrics_ensemble['roc_auc']:.4f} | "
              f"Acc: {metrics_ensemble['accuracy']:.4f}")
        
        # 计算改进
        improvement_f1 = metrics_ensemble['f1'] - max(metrics_model1['f1'], metrics_model3['f1'])
        improvement_auc = metrics_ensemble['roc_auc'] - max(
            metrics_model1['roc_auc'], metrics_model3['roc_auc']
        )
        
        if improvement_f1 > 0 or improvement_auc > 0:
            print(f"  ✓ 改进    | F1: {improvement_f1:+.4f} | AUC: {improvement_auc:+.4f}")
        else:
            print(f"  - 改进    | F1: {improvement_f1:+.4f} | AUC: {improvement_auc:+.4f}")
    
    # 保存结果
    save_evaluation_results(results)
    
    print("\n" + "="*70)
    print("评估完成！")
    print("="*70)
    
    return results


def save_evaluation_results(results: dict):
    """保存评估结果到文件"""
    output_file = RESULTS_DIR / "ensemble_evaluation.txt"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("Daily Ensemble 评估结果\n")
        f.write("="*70 + "\n\n")
        
        # 回归任务
        f.write("【回归任务】\n")
        f.write("-"*70 + "\n")
        for task_name, metrics in results['regression'].items():
            f.write(f"\n{task_name.upper()}\n")
            f.write(f"  Model 1  : R²={metrics['model1']['r2']:.4f}, "
                   f"MAE={metrics['model1']['mae']:.4f}\n")
            f.write(f"  Model 3  : R²={metrics['model3']['r2']:.4f}, "
                   f"MAE={metrics['model3']['mae']:.4f}\n")
            f.write(f"  Ensemble : R²={metrics['ensemble']['r2']:.4f}, "
                   f"MAE={metrics['ensemble']['mae']:.4f}\n")
        
        # 分类任务
        f.write("\n" + "-"*70 + "\n")
        f.write("\n【分类任务】\n")
        f.write("-"*70 + "\n")
        for task_name, metrics in results['classification'].items():
            f.write(f"\n{task_name.upper()}\n")
            f.write(f"  Model 1  : F1={metrics['model1']['f1']:.4f}, "
                   f"AUC={metrics['model1']['roc_auc']:.4f}\n")
            f.write(f"  Model 3  : F1={metrics['model3']['f1']:.4f}, "
                   f"AUC={metrics['model3']['roc_auc']:.4f}\n")
            f.write(f"  Ensemble : F1={metrics['ensemble']['f1']:.4f}, "
                   f"AUC={metrics['ensemble']['roc_auc']:.4f}\n")
    
    print(f"\n结果已保存到: {output_file}")


if __name__ == "__main__":
    try:
        results = evaluate_models()
    except KeyboardInterrupt:
        print("\n\n评估被中断")
    except Exception as e:
        print(f"\n\n评估出错: {e}")
        import traceback
        traceback.print_exc()
