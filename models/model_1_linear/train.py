"""
模型训练脚本 - Linear Baseline
训练 Logistic Regression (分类) 和 Ridge Regression (回归) 模型
"""
import sys
import json
import pickle
import argparse
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, classification_report,
    confusion_matrix, mean_absolute_error, 
    mean_squared_error, r2_score
)

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))
import config

warnings.filterwarnings('ignore')


def load_data():
    """加载训练和测试数据"""
    print("\n" + "="*60)
    print("加载数据")
    print("="*60)
    
    print(f"数据类型: {config.DATA_TYPE}")
    print(f"训练集: {config.TRAIN_DATA_PATH}")
    print(f"测试集: {config.TEST_DATA_PATH}")
    
    if not config.TRAIN_DATA_PATH.exists():
        raise FileNotFoundError(f"训练集文件不存在: {config.TRAIN_DATA_PATH}")
    if not config.TEST_DATA_PATH.exists():
        raise FileNotFoundError(f"测试集文件不存在: {config.TEST_DATA_PATH}")
    
    train_df = pd.read_csv(config.TRAIN_DATA_PATH)
    test_df = pd.read_csv(config.TEST_DATA_PATH)
    
    print(f"\n训练集形状: {train_df.shape}")
    print(f"测试集形状: {test_df.shape}")
    
    return train_df, test_df


def prepare_features(train_df, test_df, target_column):
    """准备特征和目标变量"""
    print("\n" + "="*60)
    print("准备特征")
    print("="*60)
    
    if target_column not in train_df.columns:
        raise ValueError(f"目标列 '{target_column}' 不存在于数据中")
    
    excluded = set(config.EXCLUDED_COLUMNS + config.TARGET_COLUMNS)
    feature_columns = [
        col for col in train_df.columns 
        if col not in excluded and train_df[col].dtype in ['int64', 'float64']
    ]
    
    print(f"特征数量: {len(feature_columns)}")
    print(f"目标列: {target_column}")
    
    X_train = train_df[feature_columns].values
    y_train = train_df[target_column].values
    X_test = test_df[feature_columns].values
    y_test = test_df[target_column].values
    
    # 处理缺失值
    if np.isnan(X_train).any():
        print(f"警告: 训练集存在 {np.isnan(X_train).sum()} 个缺失值")
        X_train = np.nan_to_num(X_train, 0)
    if np.isnan(X_test).any():
        print(f"警告: 测试集存在 {np.isnan(X_test).sum()} 个缺失值")
        X_test = np.nan_to_num(X_test, 0)
    
    return X_train, X_test, y_train, y_test, feature_columns


def train_classification(X_train, y_train, use_sampling=False, sample_fraction=0.1):
    """训练分类模型"""
    print("\n" + "="*60)
    print("训练 Logistic Regression 模型")
    print("="*60)
    
    if use_sampling and len(X_train) > 100000:
        print(f"使用 {sample_fraction*100}% 数据采样")
        n_samples = int(len(X_train) * sample_fraction)
        indices = np.random.choice(len(X_train), n_samples, replace=False)
        X_train = X_train[indices]
        y_train = y_train[indices]
        print(f"采样后样本数: {len(X_train):,}")
    
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"\n训练集类别分布:")
    for label, count in zip(unique, counts):
        print(f"  类别 {label}: {count:,} ({count/len(y_train)*100:.2f}%)")
    
    model = LogisticRegression(**config.LOGISTIC_PARAMS)
    
    print(f"\n开始训练...")
    start_time = datetime.now()
    model.fit(X_train, y_train)
    train_time = (datetime.now() - start_time).total_seconds()
    print(f"训练完成！耗时: {train_time:.2f} 秒")
    
    return model


def train_regression(X_train, y_train, use_sampling=False, sample_fraction=0.1):
    """训练回归模型"""
    print("\n" + "="*60)
    print("训练 Ridge Regression 模型")
    print("="*60)
    
    if use_sampling and len(X_train) > 100000:
        print(f"使用 {sample_fraction*100}% 数据采样")
        n_samples = int(len(X_train) * sample_fraction)
        indices = np.random.choice(len(X_train), n_samples, replace=False)
        X_train = X_train[indices]
        y_train = y_train[indices]
        print(f"采样后样本数: {len(X_train):,}")
    
    print(f"\n目标变量统计:")
    print(f"  均值: {y_train.mean():.4f}")
    print(f"  标准差: {y_train.std():.4f}")
    print(f"  最小值: {y_train.min():.4f}")
    print(f"  最大值: {y_train.max():.4f}")
    
    model = Ridge(**config.RIDGE_PARAMS)
    
    print(f"\n开始训练...")
    start_time = datetime.now()
    model.fit(X_train, y_train)
    train_time = (datetime.now() - start_time).total_seconds()
    print(f"训练完成！耗时: {train_time:.2f} 秒")
    
    return model


def evaluate_classification(model, X_test, y_test):
    """评估分类模型"""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_proba),
    }
    
    print("\n" + "="*60)
    print("测试集性能")
    print("="*60)
    for metric, value in metrics.items():
        print(f"  {metric.upper()}: {value:.4f}")
    
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    cm = confusion_matrix(y_test, y_pred)
    print("混淆矩阵:")
    print(cm)
    
    return metrics


def evaluate_regression(model, X_test, y_test):
    """评估回归模型"""
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    mask = y_test != 0
    mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100 if mask.sum() > 0 else np.nan
    
    metrics = {'mae': mae, 'rmse': rmse, 'r2': r2, 'mape': mape}
    
    print("\n" + "="*60)
    print("测试集性能")
    print("="*60)
    print(f"  MAE:  {metrics['mae']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  R²:   {metrics['r2']:.4f}")
    if not np.isnan(metrics['mape']):
        print(f"  MAPE: {metrics['mape']:.2f}%")
    
    return metrics


def save_model(model, feature_columns, task_type, target_name):
    """保存模型和特征"""
    model_path = config.get_model_path(task_type, target_name)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n模型已保存: {model_path}")
    
    feature_names_path = config.get_feature_names_path()
    with open(feature_names_path, 'w') as f:
        json.dump(feature_columns, f, indent=2)
    print(f"特征名称已保存: {feature_names_path}")


def train_all_tasks(sample_fraction=None, skip_classification=False, skip_regression=False):
    """训练所有任务"""
    print("\n" + "="*70)
    print(" "*15 + "批量训练所有预测任务")
    print("="*70)
    print(f"数据类型: {config.DATA_TYPE}")
    print(f"采样比例: {sample_fraction if sample_fraction else '无（使用全部数据）'}")
    
    # 加载数据（只加载一次）
    train_df, test_df = load_data()
    
    results_summary = {'classification': {}, 'regression': {}}
    start_time = datetime.now()
    
    # 训练分类任务
    if not skip_classification:
        print("\n" + "="*70)
        print(" "*20 + "分类任务训练")
        print("="*70)
        
        for task_name, target_column in config.CLASSIFICATION_TARGETS.items():
            print(f"\n{'='*70}")
            print(f"训练分类任务: {task_name} ({target_column})")
            print(f"{'='*70}")
            
            try:
                X_train, X_test, y_train, y_test, feature_columns = prepare_features(
                    train_df, test_df, target_column
                )
                
                use_sampling = sample_fraction is not None
                model = train_classification(X_train, y_train, use_sampling, sample_fraction or 0.1)
                metrics = evaluate_classification(model, X_test, y_test)
                save_model(model, feature_columns, 'classification', task_name)
                
                results_summary['classification'][task_name] = {
                    'target': target_column,
                    'accuracy': metrics['accuracy'],
                    'f1': metrics['f1'],
                    'roc_auc': metrics['roc_auc']
                }
                
                print(f"\n✅ {task_name} 完成！F1-Score: {metrics['f1']:.4f}")
                
            except Exception as e:
                print(f"\n❌ {task_name} 失败: {str(e)}")
                results_summary['classification'][task_name] = {'error': str(e)}
    
    # 训练回归任务
    if not skip_regression:
        print("\n" + "="*70)
        print(" "*20 + "回归任务训练")
        print("="*70)
        
        for task_name, target_column in config.REGRESSION_TARGETS.items():
            print(f"\n{'='*70}")
            print(f"训练回归任务: {task_name} ({target_column})")
            print(f"{'='*70}")
            
            try:
                X_train, X_test, y_train, y_test, feature_columns = prepare_features(
                    train_df, test_df, target_column
                )
                
                use_sampling = sample_fraction is not None
                model = train_regression(X_train, y_train, use_sampling, sample_fraction or 0.1)
                metrics = evaluate_regression(model, X_test, y_test)
                save_model(model, feature_columns, 'regression', task_name)
                
                results_summary['regression'][task_name] = {
                    'target': target_column,
                    'r2': metrics['r2'],
                    'mae': metrics['mae'],
                    'rmse': metrics['rmse']
                }
                
                print(f"\n✅ {task_name} 完成！R²: {metrics['r2']:.4f}")
                
            except Exception as e:
                print(f"\n❌ {task_name} 失败: {str(e)}")
                results_summary['regression'][task_name] = {'error': str(e)}
    
    # 总结
    total_time = (datetime.now() - start_time).total_seconds()
    
    print("\n" + "="*70)
    print(" "*20 + "训练完成总结")
    print("="*70)
    print(f"总耗时: {total_time:.2f} 秒")
    
    if not skip_classification:
        print(f"\n分类任务:")
        for task_name, result in results_summary['classification'].items():
            if 'error' not in result:
                print(f"  ✅ {task_name}: F1={result['f1']:.4f}, AUC={result['roc_auc']:.4f}")
            else:
                print(f"  ❌ {task_name}: {result['error']}")
    
    if not skip_regression:
        print(f"\n回归任务:")
        for task_name, result in results_summary['regression'].items():
            if 'error' not in result:
                print(f"  ✅ {task_name}: R²={result['r2']:.4f}, MAE={result['mae']:.4f}")
            else:
                print(f"  ❌ {task_name}: {result['error']}")
    
    # 保存总结
    save_summary_report(results_summary, total_time)
    
    print("\n" + "="*70)
    print("所有任务训练完成！")
    print("="*70)
    
    return results_summary


def save_summary_report(results, total_time):
    """保存训练总结报告"""
    report_path = config.RESULTS_DIR / "training_summary.txt"
    
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write(" "*20 + "Model 1 训练总结\n")
        f.write("="*70 + "\n")
        f.write(f"训练时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"数据类型: {config.DATA_TYPE}\n")
        f.write(f"总耗时: {total_time:.2f} 秒\n")
        f.write("="*70 + "\n\n")
        
        f.write("分类任务:\n")
        f.write("-"*70 + "\n")
        for task_name, result in results['classification'].items():
            if 'error' not in result:
                f.write(f"{task_name:20s} | ")
                f.write(f"Accuracy: {result['accuracy']:.4f} | ")
                f.write(f"F1: {result['f1']:.4f} | ")
                f.write(f"AUC: {result['roc_auc']:.4f}\n")
        
        f.write("\n回归任务:\n")
        f.write("-"*70 + "\n")
        for task_name, result in results['regression'].items():
            if 'error' not in result:
                f.write(f"{task_name:20s} | ")
                f.write(f"R²: {result['r2']:.4f} | ")
                f.write(f"MAE: {result['mae']:.4f} | ")
                f.write(f"RMSE: {result['rmse']:.4f}\n")
    
    print(f"\n训练总结已保存: {report_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='训练线性基准模型（默认训练所有任务）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python train.py                      # 训练所有任务
  python train.py --skip-regression     # 只训练分类任务
  python train.py --skip-classification # 只训练回归任务
  python train.py --sample 0.1          # 使用10%数据采样
  python train.py --task classification --target rain  # 训练单个任务（旧模式）
        """
    )
    
    # 批量训练参数
    parser.add_argument('--skip-classification', action='store_true',
                        help='跳过分类任务')
    parser.add_argument('--skip-regression', action='store_true',
                        help='跳过回归任务')
    parser.add_argument('--sample', type=float, default=None,
                        help='数据采样比例 (0-1)')
    
    # 单任务训练参数（向后兼容）
    parser.add_argument('--task', type=str,
                        choices=['classification', 'regression'],
                        help='任务类型（单任务模式）')
    parser.add_argument('--target', type=str,
                        help='目标名称（单任务模式）')
    
    args = parser.parse_args()
    
    # 单任务模式（向后兼容）
    if args.task and args.target:
        print("\n" + "="*60)
        print("Linear Baseline 模型训练 (单任务模式)")
        print("="*60)
        print(f"任务类型: {args.task}")
        print(f"目标: {args.target}")
        
        # 获取目标列名
        if args.task == 'classification':
            target_column = config.CLASSIFICATION_TARGETS.get(args.target, args.target)
        else:
            target_column = config.REGRESSION_TARGETS.get(args.target, args.target)
        
        # 1. 加载数据
        train_df, test_df = load_data()
        
        # 2. 准备特征
        X_train, X_test, y_train, y_test, feature_columns = prepare_features(
            train_df, test_df, target_column
        )
        
        # 3. 训练模型
        use_sampling = args.sample is not None
        sample_fraction = args.sample if use_sampling else config.SAMPLE_FRACTION
        
        if args.task == 'classification':
            model = train_classification(X_train, y_train, use_sampling, sample_fraction)
            metrics = evaluate_classification(model, X_test, y_test)
        else:
            model = train_regression(X_train, y_train, use_sampling, sample_fraction)
            metrics = evaluate_regression(model, X_test, y_test)
        
        # 4. 保存模型
        save_model(model, feature_columns, args.task, args.target)
        
        print("\n" + "="*60)
        print("训练完成！")
        print("="*60)
        
        return model, metrics
    
    # 批量训练模式（默认）
    else:
        results = train_all_tasks(
            sample_fraction=args.sample,
            skip_classification=args.skip_classification,
            skip_regression=args.skip_regression
        )
        return results


if __name__ == "__main__":
    results = main()
