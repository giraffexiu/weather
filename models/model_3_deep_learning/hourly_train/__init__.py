"""
hourly_train - Wide & Deep 小时天气预测模块

模型 3：高维泛化专家 —— 深度学习神经网络 (Wide & Deep)

Wide 侧：直接输入气压、温度以及风向×月份的交叉特征（捕捉强规则）
Deep 侧：Embedding + 标准化数值特征 → 3层 DNN（Dropout + BatchNorm）

用法:
    # 训练
    python -m hourly_train.train --batch_size 512 --epochs 50

    # 推理
    python -m hourly_train.inference --batch_size 1024
"""

from .model import WideAndDeep, create_model, count_parameters
from .inference import Predictor

__all__ = [
    'WideAndDeep',
    'create_model',
    'count_parameters',
    'Predictor',
]
