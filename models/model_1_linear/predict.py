"""
模型预测脚本
使用训练好的模型对新数据进行预测
"""
import sys
import json
import pickle
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))
import config


class WeatherPredictor:
    """天气预测器类"""
    
    def __init__(self, task_type, target_name):
        """
        初始化预测器
        
        Args:
            task_type: 'classification' 或 'regression'
            target_name: 目标名称 (如 'rain', 'temp_mean')
        """
        self.task_type = task_type
        self.target_name = target_name
        
        # 加载模型
        model_path = config.get_model_path(task_type, target_name)
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        print(f"✓ 模型已加载: {model_path.name}")
        
        # 加载特征名称
        feature_names_path = config.get_feature_names_path()
        if not feature_names_path.exists():
            raise FileNotFoundError(f"特征名称文件不存在: {feature_names_path}")
        
        with open(feature_names_path, 'r') as f:
            self.feature_names = json.load(f)
        
        print(f"✓ 特征数量: {len(self.feature_names)}")
    
    def load_data(self, data_path):
        """
        加载输入数据
        
        Args:
            data_path: CSV 文件路径
            
        Returns:
            DataFrame
        """
        if isinstance(data_path, str):
            data_path = Path(data_path)
        
        if not data_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {data_path}")
        
        df = pd.read_csv(data_path)
        print(f"✓ 数据已加载: {df.shape}")
        
        # 检查特征是否存在
        missing_features = [f for f in self.feature_names if f not in df.columns]
        if missing_features:
            raise ValueError(f"缺少特征列: {missing_features[:5]}...")
        
        return df
    
    def prepare_features(self, df):
        """
        准备特征矩阵
        
        Args:
            df: DataFrame
            
        Returns:
            numpy array
        """
        X = df[self.feature_names].values
        
        # 处理缺失值
        if np.isnan(X).any():
            n_missing = np.isnan(X).sum()
            print(f"⚠ 检测到 {n_missing} 个缺失值，将填充为0")
            X = np.nan_to_num(X, 0)
        
        return X
    
    def predict(self, data_path, save_output=True):
        """
        执行预测
        
        Args:
            data_path: 输入数据路径
            save_output: 是否保存预测结果
            
        Returns:
            DataFrame with predictions
        """
        print("\n" + "="*60)
        print(f"开始预测 - {self.task_type.upper()}")
        print("="*60)
        
        # 1. 加载数据
        df = self.load_data(data_path)
        
        # 2. 准备特征
        X = self.prepare_features(df)
        
        # 3. 预测
        print(f"\n正在预测 {len(X):,} 个样本...")
        
        if self.task_type == 'classification':
            predictions = self.model.predict(X)
            probabilities = self.model.predict_proba(X)[:, 1]
            
            # 添加预测结果到 DataFrame
            result_df = df.copy()
            result_df['predicted'] = predictions
            result_df['probability'] = probabilities
            
            # 统计
            unique, counts = np.unique(predictions, return_counts=True)
            print(f"\n预测结果分布:")
            for label, count in zip(unique, counts):
                print(f"  类别 {label}: {count:,} ({count/len(predictions)*100:.2f}%)")
            
            print(f"\n平均预测概率: {probabilities.mean():.4f}")
            
        else:  # regression
            predictions = self.model.predict(X)
            
            # 添加预测结果到 DataFrame
            result_df = df.copy()
            result_df['predicted'] = predictions
            
            # 统计
            print(f"\n预测结果统计:")
            print(f"  均值: {predictions.mean():.4f}")
            print(f"  标准差: {predictions.std():.4f}")
            print(f"  最小值: {predictions.min():.4f}")
            print(f"  最大值: {predictions.max():.4f}")
        
        # 4. 保存结果
        if save_output:
            output_path = config.PREDICTIONS_DIR / f"predictions_{self.task_type}_{self.target_name}.csv"
            result_df.to_csv(output_path, index=False)
            print(f"\n✓ 预测结果已保存: {output_path}")
        
        return result_df
    
    def predict_single(self, features_dict):
        """
        预测单个样本
        
        Args:
            features_dict: 特征字典 {feature_name: value}
            
        Returns:
            prediction (and probability for classification)
        """
        # 构造特征向量
        X = np.array([[features_dict.get(f, 0) for f in self.feature_names]])
        
        if self.task_type == 'classification':
            prediction = self.model.predict(X)[0]
            probability = self.model.predict_proba(X)[0, 1]
            return prediction, probability
        else:
            prediction = self.model.predict(X)[0]
            return prediction


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='使用训练好的模型进行预测')
    parser.add_argument('--task', type=str, required=True,
                        choices=['classification', 'regression'],
                        help='任务类型')
    parser.add_argument('--target', type=str, required=True,
                        help='目标名称 (如: rain, temp_mean)')
    parser.add_argument('--input', type=str, required=True,
                        help='输入数据文件路径 (CSV)')
    parser.add_argument('--no-save', action='store_true',
                        help='不保存预测结果')
    
    args = parser.parse_args()
    
    # 创建预测器
    predictor = WeatherPredictor(args.task, args.target)
    
    # 执行预测
    result_df = predictor.predict(args.input, save_output=not args.no_save)
    
    print("\n" + "="*60)
    print("预测完成！")
    print("="*60)
    
    # 显示前几行预测结果
    if args.task == 'classification':
        print("\n预测结果示例 (前10行):")
        print(result_df[['time', 'city', 'predicted', 'probability']].head(10).to_string(index=False))
    else:
        print("\n预测结果示例 (前10行):")
        print(result_df[['time', 'city', 'predicted']].head(10).to_string(index=False))
    
    return result_df


if __name__ == "__main__":
    result_df = main()
