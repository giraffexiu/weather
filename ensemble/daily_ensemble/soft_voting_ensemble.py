"""
软投票集成：加权平均 Model 1 和 Model 3 的预测结果
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

from config import (
    MODEL1_CLASSIFICATION_TASKS,
    MODEL1_REGRESSION_TASKS,
    ENSEMBLE_TASKS,
    WEIGHT_METHOD,
    REGRESSION_METRIC_FOR_WEIGHT,
    CLASSIFICATION_METRIC_FOR_WEIGHT,
    MODEL3_PERFORMANCE
)


class SoftVotingEnsemble:
    """
    软投票集成器
    
    对于回归任务：加权平均连续值
    对于分类任务：加权平均概率值
    """
    
    def __init__(
        self,
        model1_wrapper,
        model3_wrapper,
        weight_method: str = 'performance_based',
        verbose: bool = True
    ):
        """
        初始化软投票集成器
        
        Args:
            model1_wrapper: Model 1 包装器
            model3_wrapper: Model 3 包装器
            weight_method: 权重计算方法 ('equal', 'performance_based')
            verbose: 是否打印详细信息
        """
        self.model1 = model1_wrapper
        self.model3 = model3_wrapper
        self.weight_method = weight_method
        self.verbose = verbose
        
        # 计算权重
        self.weights = self._calculate_weights()
        
        if self.verbose:
            self._print_weights()
    
    def _calculate_weights(self) -> Dict[str, Dict[str, float]]:
        """
        计算每个任务的模型权重
        
        Returns:
            权重字典: {
                'regression': {
                    'temp_mean': {'model1': 0.5, 'model3': 0.5},
                    ...
                },
                'classification': {
                    'rain': {'model1': 0.5, 'model3': 0.5},
                    ...
                }
            }
        """
        weights = {
            'regression': {},
            'classification': {}
        }
        
        if self.weight_method == 'equal':
            # 等权重
            for task in ENSEMBLE_TASKS['regression'].keys():
                weights['regression'][task] = {'model1': 0.5, 'model3': 0.5}
            
            for task in ENSEMBLE_TASKS['classification'].keys():
                weights['classification'][task] = {'model1': 0.5, 'model3': 0.5}
        
        elif self.weight_method == 'performance_based':
            # 基于性能的权重
            
            # 回归任务权重
            for task_name, task_config in ENSEMBLE_TASKS['regression'].items():
                model1_key = task_config['model1_key']
                
                # 获取性能指标
                model1_metric = MODEL1_REGRESSION_TASKS[model1_key].get(
                    REGRESSION_METRIC_FOR_WEIGHT, 0.5
                )
                
                # Model 3 的性能（如果未评估，使用默认值 0.5）
                model3_target = list(MODEL3_PERFORMANCE.keys())[
                    task_config['model3_index']
                ]
                model3_metric = MODEL3_PERFORMANCE[model3_target].get(
                    REGRESSION_METRIC_FOR_WEIGHT
                )
                
                if model3_metric is None:
                    # 未评估 Model 3，使用等权重
                    w1, w3 = 0.5, 0.5
                else:
                    # 归一化权重
                    if REGRESSION_METRIC_FOR_WEIGHT in ['r2']:
                        # R²越大越好
                        total = model1_metric + model3_metric
                        if total > 0:
                            w1 = model1_metric / total
                            w3 = model3_metric / total
                        else:
                            w1, w3 = 0.5, 0.5
                    else:
                        # MAE/RMSE 越小越好，使用倒数
                        inv1 = 1.0 / (model1_metric + 1e-6)
                        inv3 = 1.0 / (model3_metric + 1e-6)
                        total = inv1 + inv3
                        w1 = inv1 / total
                        w3 = inv3 / total
                
                weights['regression'][task_name] = {'model1': w1, 'model3': w3}
            
            # 分类任务权重
            for task_name, task_config in ENSEMBLE_TASKS['classification'].items():
                model1_key = task_config['model1_key']
                
                # 获取 F1-Score
                model1_f1 = MODEL1_CLASSIFICATION_TASKS[model1_key].get('f1_score', 0.5)
                
                # Model 3 的分类性能需要通过评估获得
                # 这里暂时使用等权重或给 Model 1 更高权重（因为是专门训练的分类器）
                model3_f1 = 0.5  # 占位符
                
                # 归一化
                total = model1_f1 + model3_f1
                if total > 0:
                    w1 = model1_f1 / total
                    w3 = model3_f1 / total
                else:
                    w1, w3 = 0.5, 0.5
                
                weights['classification'][task_name] = {'model1': w1, 'model3': w3}
        
        else:
            raise ValueError(f"Unknown weight method: {self.weight_method}")
        
        return weights
    
    def _print_weights(self):
        """打印权重信息"""
        print("\n" + "="*70)
        print("集成权重配置")
        print("="*70)
        print(f"权重方法: {self.weight_method}")
        
        print("\n回归任务权重:")
        print("-"*70)
        for task, weight in self.weights['regression'].items():
            print(f"  {task:20s} | Model1: {weight['model1']:.3f} | "
                  f"Model3: {weight['model3']:.3f}")
        
        print("\n分类任务权重:")
        print("-"*70)
        for task, weight in self.weights['classification'].items():
            print(f"  {task:20s} | Model1: {weight['model1']:.3f} | "
                  f"Model3: {weight['model3']:.3f}")
        print("="*70)
    
    def ensemble_regression(
        self,
        model1_predictions: Dict[str, np.ndarray],
        model3_predictions: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        集成回归任务的预测
        
        Args:
            model1_predictions: Model 1 的回归预测
            model3_predictions: Model 3 的回归预测
            
        Returns:
            集成后的预测结果
        """
        ensemble_results = {}
        
        for task_name, task_config in ENSEMBLE_TASKS['regression'].items():
            model1_key = task_config['model1_key']
            model3_index = task_config['model3_index']
            
            # 获取两个模型的预测
            model1_pred = model1_predictions[model1_key]
            
            # 从 Model 3 找到对应的目标
            model3_target = list(MODEL3_PERFORMANCE.keys())[model3_index]
            model3_pred = model3_predictions[model3_target]
            
            # 获取权重
            w1 = self.weights['regression'][task_name]['model1']
            w3 = self.weights['regression'][task_name]['model3']
            
            # 加权平均
            ensemble_pred = w1 * model1_pred + w3 * model3_pred
            
            ensemble_results[task_name] = ensemble_pred
        
        return ensemble_results
    
    def ensemble_classification(
        self,
        model1_predictions: Dict[str, Dict[str, np.ndarray]],
        model3_predictions: Dict[str, Dict[str, np.ndarray]]
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        集成分类任务的预测
        
        Args:
            model1_predictions: Model 1 的分类预测（包含概率）
            model3_predictions: Model 3 的分类预测（包含概率）
            
        Returns:
            集成后的预测结果 {'rain': {'probability': array, 'prediction': array}}
        """
        ensemble_results = {}
        
        for task_name in ENSEMBLE_TASKS['classification'].keys():
            # 获取两个模型的概率
            model1_prob = model1_predictions[task_name]['probability']
            model3_prob = model3_predictions[task_name]['probability']
            
            # 获取权重
            w1 = self.weights['classification'][task_name]['model1']
            w3 = self.weights['classification'][task_name]['model3']
            
            # 加权平均概率
            ensemble_prob = w1 * model1_prob + w3 * model3_prob
            
            # 基于概率阈值生成预测
            ensemble_pred = (ensemble_prob >= 0.5).astype(int)
            
            ensemble_results[task_name] = {
                'probability': ensemble_prob,
                'prediction': ensemble_pred
            }
        
        return ensemble_results
    
    def predict(
        self,
        X_model1: np.ndarray,
        data_loader_model3
    ) -> Dict[str, Any]:
        """
        执行集成预测
        
        Args:
            X_model1: Model 1 的输入特征矩阵 (N, num_features)
            data_loader_model3: Model 3 的 DataLoader
            
        Returns:
            集成预测结果:
            {
                'regression': {
                    'temp_mean': array,
                    'temp_max': array,
                    ...
                },
                'classification': {
                    'rain': {'probability': array, 'prediction': array},
                    'snow': {...},
                    'severe': {...}
                },
                'individual': {
                    'model1': {...},
                    'model3': {...}
                }
            }
        """
        if self.verbose:
            print("\n执行集成预测...")
        
        # Model 1 预测
        if self.verbose:
            print("  Model 1 预测中...")
        model1_results = self.model1.predict(X_model1)
        
        # Model 3 预测
        if self.verbose:
            print("  Model 3 预测中...")
        model3_results = self.model3.predict(data_loader_model3)
        
        # 集成回归任务
        if self.verbose:
            print("  集成回归任务...")
        ensemble_regression = self.ensemble_regression(
            model1_results['regression'],
            model3_results['regression']
        )
        
        # 集成分类任务
        if self.verbose:
            print("  集成分类任务...")
        ensemble_classification = self.ensemble_classification(
            model1_results['classification'],
            model3_results['classification']
        )
        
        if self.verbose:
            print("  ✓ 集成完成")
        
        return {
            'regression': ensemble_regression,
            'classification': ensemble_classification,
            'individual': {
                'model1': model1_results,
                'model3': model3_results
            }
        }
    
    def predict_with_dataframe(
        self,
        test_df: pd.DataFrame,
        data_loader_model3
    ) -> pd.DataFrame:
        """
        使用 DataFrame 执行预测并返回结果
        
        Args:
            test_df: 测试数据 DataFrame
            data_loader_model3: Model 3 的 DataLoader
            
        Returns:
            包含预测结果的 DataFrame
        """
        # 准备 Model 1 的输入
        X_model1 = self.model1.prepare_features(test_df)
        
        # 执行预测
        predictions = self.predict(X_model1, data_loader_model3)
        
        # 构建结果 DataFrame
        result_df = test_df[['time', 'city', 'country']].copy()
        
        # 添加回归预测
        for task_name, values in predictions['regression'].items():
            result_df[f'ensemble_{task_name}'] = values
        
        # 添加分类预测
        for task_name, pred_dict in predictions['classification'].items():
            result_df[f'ensemble_{task_name}_prob'] = pred_dict['probability']
            result_df[f'ensemble_{task_name}_pred'] = pred_dict['prediction']
        
        return result_df


if __name__ == "__main__":
    print("软投票集成模块已加载")
    print("请使用 predict_ensemble.py 进行实际预测")
