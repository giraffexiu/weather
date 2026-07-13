"""
模型包装器：为 Model 1 和 Model 3 提供统一的预测接口
"""
import pickle
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from config import (
    MODEL1_CLASSIFICATION_TASKS,
    MODEL1_REGRESSION_TASKS,
    MODEL1_FEATURE_NAMES,
    MODEL3_CHECKPOINT,
    MODEL3_DIR,
    MODEL3_TARGET_INDICES,
    DEVICE,
    DATA_DIR
)
from probability_converter import ProbabilityConverter


class BaseModelWrapper(ABC):
    """模型包装器基类"""
    
    @abstractmethod
    def predict(self, X: Any) -> Dict[str, np.ndarray]:
        """
        预测接口
        
        Args:
            X: 输入数据
            
        Returns:
            预测结果字典
        """
        pass


class Model1Wrapper(BaseModelWrapper):
    """
    Model 1 (Linear Models) 包装器
    
    加载所有训练好的 sklearn 模型，提供统一预测接口
    """
    
    def __init__(self):
        """初始化 Model 1 包装器"""
        self.classification_models = {}
        self.regression_models = {}
        self.feature_names = None
        
        self._load_models()
        self._load_feature_names()
    
    def _load_models(self):
        """加载所有模型"""
        print("加载 Model 1 (Linear) 模型...")
        
        # 加载分类模型
        for task_name, task_config in MODEL1_CLASSIFICATION_TASKS.items():
            model_path = task_config['model_path']
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")
            
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            self.classification_models[task_name] = {
                'model': model,
                'target_column': task_config['target_column']
            }
            print(f"  ✓ {task_name} (分类)")
        
        # 加载回归模型
        for task_name, task_config in MODEL1_REGRESSION_TASKS.items():
            model_path = task_config['model_path']
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")
            
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            self.regression_models[task_name] = {
                'model': model,
                'target_column': task_config['target_column']
            }
            print(f"  ✓ {task_name} (回归)")
        
        print(f"Model 1 加载完成: {len(self.classification_models)} 个分类模型, "
              f"{len(self.regression_models)} 个回归模型")
    
    def _load_feature_names(self):
        """加载特征名称"""
        if not MODEL1_FEATURE_NAMES.exists():
            raise FileNotFoundError(f"Feature names not found: {MODEL1_FEATURE_NAMES}")
        
        with open(MODEL1_FEATURE_NAMES, 'r') as f:
            self.feature_names = json.load(f)
        
        print(f"  ✓ 特征数量: {len(self.feature_names)}")
    
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        准备特征矩阵
        
        Args:
            df: 输入 DataFrame
            
        Returns:
            特征矩阵 (N, num_features)
        """
        # 检查特征是否存在
        missing_features = [f for f in self.feature_names if f not in df.columns]
        if missing_features:
            raise ValueError(
                f"Missing features in input data: {missing_features[:5]}..."
            )
        
        # 提取特征
        X = df[self.feature_names].values
        
        # 处理缺失值
        if np.isnan(X).any():
            n_missing = np.isnan(X).sum()
            print(f"  ⚠ 检测到 {n_missing} 个缺失值，填充为0")
            X = np.nan_to_num(X, 0.0)
        
        return X
    
    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        执行预测
        
        Args:
            X: 特征矩阵 (N, num_features)
            
        Returns:
            预测结果字典:
            {
                'classification': {
                    'rain': {'probability': array, 'prediction': array},
                    'snow': {...},
                    'severe': {...}
                },
                'regression': {
                    'temp_mean': array,
                    'temp_max': array,
                    ...
                }
            }
        """
        results = {
            'classification': {},
            'regression': {}
        }
        
        # 分类任务预测
        for task_name, task_info in self.classification_models.items():
            model = task_info['model']
            
            # 预测概率
            probabilities = model.predict_proba(X)[:, 1]  # 正类概率
            predictions = model.predict(X)
            
            results['classification'][task_name] = {
                'probability': probabilities,
                'prediction': predictions
            }
        
        # 回归任务预测
        for task_name, task_info in self.regression_models.items():
            model = task_info['model']
            predictions = model.predict(X)
            
            results['regression'][task_name] = predictions
        
        return results


class Model3Wrapper(BaseModelWrapper):
    """
    Model 3 (Wide & Deep) 包装器
    
    加载 PyTorch 模型，提供统一预测接口
    """
    
    def __init__(self, probability_converter: Optional[ProbabilityConverter] = None):
        """
        初始化 Model 3 包装器
        
        Args:
            probability_converter: 概率转换器（用于分类任务）
        """
        self.model = None
        self.device = DEVICE
        self.probability_converter = probability_converter
        
        # 添加必要的路径
        if str(MODEL3_DIR) not in sys.path:
            sys.path.insert(0, str(MODEL3_DIR))
        if str(DATA_DIR) not in sys.path:
            sys.path.insert(0, str(DATA_DIR))
        
        self._load_model()
    
    def _load_model(self):
        """加载 Model 3"""
        print("\n加载 Model 3 (Wide & Deep) 模型...")
        
        if not MODEL3_CHECKPOINT.exists():
            raise FileNotFoundError(f"Checkpoint not found: {MODEL3_CHECKPOINT}")
        
        try:
            # 导入模型定义
            from model import WideDeepModel
            from train_config import CONFIG
            
            # 创建模型
            self.model = WideDeepModel(CONFIG).to(self.device)
            
            # 加载权重
            checkpoint = torch.load(
                MODEL3_CHECKPOINT,
                map_location=self.device,
                weights_only=False
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            print(f"  ✓ 加载自 Epoch {checkpoint['epoch']}")
            print(f"  ✓ 训练时最佳 RMSE: {checkpoint['best_rmse']:.4f}")
            print(f"  ✓ 设备: {self.device}")
            
        except ImportError as e:
            raise ImportError(
                f"Failed to import Model 3 modules. "
                f"Make sure model.py and train_config.py exist in {MODEL3_DIR}\n"
                f"Error: {e}"
            )
    
    def predict(self, data_loader) -> Dict[str, np.ndarray]:
        """
        执行预测
        
        Args:
            data_loader: PyTorch DataLoader
            
        Returns:
            预测结果字典:
            {
                'regression': {
                    'temperature_2m_max': array,
                    'temperature_2m_min': array,
                    ...  # 9个目标
                },
                'classification': {  # 如果提供了 probability_converter
                    'rain': {'probability': array},
                    'snow': {'probability': array},
                    'severe': {'probability': array}
                }
            }
        """
        self.model.eval()
        all_predictions = []
        
        with torch.no_grad():
            for batch in data_loader:
                # 将数据移到设备
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # 前向传播
                predictions = self.model(batch)  # (B, 9)
                
                all_predictions.append(predictions.cpu().numpy())
        
        # 合并所有批次
        all_predictions = np.concatenate(all_predictions, axis=0)  # (N, 9)
        
        # 构建回归结果
        results = {'regression': {}}
        
        for target_name, index in MODEL3_TARGET_INDICES.items():
            results['regression'][target_name] = all_predictions[:, index]
        
        # 如果提供了概率转换器，生成分类概率
        if self.probability_converter is not None:
            classification_probs = self.probability_converter.convert_all(
                all_predictions
            )
            
            results['classification'] = {}
            for task_name, probabilities in classification_probs.items():
                results['classification'][task_name] = {
                    'probability': probabilities
                }
        
        return results


def test_model_wrappers():
    """测试模型包装器"""
    print("="*70)
    print("测试 Model 1 包装器")
    print("="*70)
    
    # 测试 Model 1
    try:
        model1 = Model1Wrapper()
        print("\n✅ Model 1 包装器测试通过")
    except Exception as e:
        print(f"\n❌ Model 1 包装器测试失败: {e}")
        return
    
    print("\n" + "="*70)
    print("测试 Model 3 包装器")
    print("="*70)
    
    # 测试 Model 3
    try:
        from config import PROBABILITY_CONVERSION_CONFIG
        converter = ProbabilityConverter(PROBABILITY_CONVERSION_CONFIG)
        model3 = Model3Wrapper(probability_converter=converter)
        print("\n✅ Model 3 包装器测试通过")
    except Exception as e:
        print(f"\n❌ Model 3 包装器测试失败: {e}")
        return
    
    print("\n" + "="*70)
    print("✅ 所有包装器测试通过")
    print("="*70)


if __name__ == "__main__":
    test_model_wrappers()
