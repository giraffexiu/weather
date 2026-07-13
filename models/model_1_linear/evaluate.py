"""
模型评估脚本
评估已训练模型在测试集上的性能
"""
import sys
import json
import pickle
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, classification_report,
    confusion_matrix, mean_absolute_error, 
    mean_squared_error, r2_score
)

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))
import config


def load_model(model_path):
    """加载训练好的模型"""
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print(f"模型已加载: {model_path}")
    return model


def load_feature_names():
    """加载特征名称"""
    feature_names_path = config.get_feature_names_path()
    
    if not feature_names_path.exists():
        raise FileNotFoundError(f"特征名称文件不存在: {feature_names_path}")
    
    with open(feature_names_path, 'r') as f:
        feature_names = json.load(f)
    
    return feature_names


def load_test_data(feature_names):
    """加载测试数据"""
    print("\n" + "="*60)
    print("加载测试数据")
    print("="*60)
    
    if not config.TEST_DATA_PATH.exists():
        raise FileNotFoundError(f"测试集文件不存在: {config.TEST_DATA_PATH}")
    
    test_df = pd.read_csv(config.TEST_DATA_PATH)
    print(f"测试集形状: {test_df.shape}")
    
    # 提取特征
    X_test = test_df[feature_names].values
    
    # 检查缺失值
    if np.isnan(X_test).any():
        print(f"警告: 测试集存在 {np.isnan(X_test).sum()} 个缺失值，将填充为0")
        X_test = np.nan_to_num(X_test, 0)
    
    return test_df, X_test


def evaluate_classification(model, X_test, y_test, task_name):
    """评估分类模型"""
    print("\n" + "="*60)
    print(f"评估分类模型: {task_name}")
    print("="*60)
    
    # 预测
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # 计算指标
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_proba),
    }
    
    # 打印结果
    print("\n性能指标:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"  Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"  F1-Score:  {metrics['f1']:.4f} ({metrics['f1']*100:.2f}%)")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f} ({metrics['roc_auc']*100:.2f}%)")
    
    # 详细分类报告
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    print("混淆矩阵:")
    print(cm)
    
    # 类别分布
    unique, counts = np.unique(y_test, return_counts=True)
    print("\n测试集类别分布:")
    for label, count in zip(unique, counts):
        print(f"  类别 {label}: {count:,} ({count/len(y_test)*100:.2f}%)")
    
    return metrics, y_pred, y_proba


def evaluate_regression(model, X_test, y_test, task_name):
    """评估回归模型"""
    print("\n" + "="*60)
    print(f"评估回归模型: {task_name}")
    print("="*60)
    
    # 预测
    y_pred = model.predict(X_test)
    
    # 计算指标
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # 计算 MAPE（避免除以零）
    mask = y_test != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
    else:
        mape = np.nan
    
    metrics = {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape,
    }
    
    # 打印结果
    print("\n性能指标:")
    print(f"  MAE (Mean Absolute Error):      {metrics['mae']:.4f}")
    print(f"  RMSE (Root Mean Squared Error): {metrics['rmse']:.4f}")
    print(f"  R² (R-squared):                 {metrics['r2']:.4f} ({metrics['r2']*100:.2f}%)")
    if not np.isnan(metrics['mape']):
        print(f"  MAPE (Mean Absolute % Error):   {metrics['mape']:.2f}%")
    
    # 统计信息
    print("\n目标变量统计:")
    print(f"  真实值 - 均值: {y_test.mean():.4f}, 标准差: {y_test.std():.4f}")
    print(f"  预测值 - 均值: {y_pred.mean():.4f}, 标准差: {y_pred.std():.4f}")
    
    # 误差统计
    errors = y_test - y_pred
    print("\n预测误差统计:")
    print(f"  误差均值: {errors.mean():.4f}")
    print(f"  误差标准差: {errors.std():.4f}")
    print(f"  误差最大值: {errors.max():.4f}")
    print(f"  误差最小值: {errors.min():.4f}")
    
    return metrics, y_pred


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='评估训练好的模型')
    parser.add_argument('--task', type=str, required=True,
                        help='任务类型: classification 或 regression')
    parser.add_argument('--target', type=str, required=True,
                        help='目标名称 (如: rain, temp_mean)')
    parser.add_argument('--save-predictions', action='store_true',
                        help='是否保存预测结果')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print(f"模型评估 - {args.task.upper()}")
    print("="*60)
    print(f"目标任务: {args.target}")
    print(f"数据类型: {config.DATA_TYPE}")
    
    # 1. 加载模型
    model_path = config.get_model_path(args.task, args.target)
    model = load_model(model_path)
    
    # 2. 加载特征名称
    feature_names = load_feature_names()
    print(f"特征数量: {len(feature_names)}")
    
    # 3. 加载测试数据
    test_df, X_test = load_test_data(feature_names)
    
    # 4. 获取目标变量
    if args.task == 'classification':
        if args.target in config.CLASSIFICATION_TARGETS:
            target_column = config.CLASSIFICATION_TARGETS[args.target]
        else:
            target_column = args.target
    elif args.task == 'regression':
        if args.target in config.REGRESSION_TARGETS:
            target_column = config.REGRESSION_TARGETS[args.target]
        else:
            target_column = args.target
    else:
        raise ValueError(f"task 必须是 'classification' 或 'regression'")
    
    if target_column not in test_df.columns:
        raise ValueError(f"目标列 '{target_column}' 不存在于测试数据中")
    
    y_test = test_df[target_column].values
    
    # 5. 评估模型
    if args.task == 'classification':
        metrics, y_pred, y_proba = evaluate_classification(
            model, X_test, y_test, args.target
        )
        
        # 保存预测结果
        if args.save_predictions:
            pred_df = test_df[['time', 'city', 'country', target_column]].copy()
            pred_df['predicted'] = y_pred
            pred_df['probability'] = y_proba
            
            save_path = config.PREDICTIONS_DIR / f"predictions_{args.task}_{args.target}.csv"
            pred_df.to_csv(save_path, index=False)
            print(f"\n预测结果已保存: {save_path}")
    
    elif args.task == 'regression':
        metrics, y_pred = evaluate_regression(
            model, X_test, y_test, args.target
        )
        
        # 保存预测结果
        if args.save_predictions:
            pred_df = test_df[['time', 'city', 'country', target_column]].copy()
            pred_df['predicted'] = y_pred
            pred_df['error'] = y_test - y_pred
            
            save_path = config.PREDICTIONS_DIR / f"predictions_{args.task}_{args.target}.csv"
            pred_df.to_csv(save_path, index=False)
            print(f"\n预测结果已保存: {save_path}")
    
    # 6. 总结
    print("\n" + "="*60)
    print("评估完成！")
    print("="*60)
    
    if args.task == 'classification':
        print(f"\n关键指标:")
        print(f"  F1-Score: {metrics['f1']:.4f}")
        print(f"  ROC-AUC:  {metrics['roc_auc']:.4f}")
    else:
        print(f"\n关键指标:")
        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  R²:  {metrics['r2']:.4f}")
    
    return metrics


if __name__ == "__main__":
    metrics = main()
