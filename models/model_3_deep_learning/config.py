"""
模型三：Wide & Deep 神经网络配置文件
"""
import os
import sys
import torch
from pathlib import Path

# ==================== 路径配置 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))

# 添加dataset_loader到Python路径
DATASET_LOADER_PATH = os.path.join(PROJECT_ROOT, "data/data_engineer/daily_data/dataset_loader")
sys.path.insert(0, DATASET_LOADER_PATH)

# 从dataset_loader导入配置和工具
# 这样可以直接使用已有的数据加载、特征分组、缓存机制
from config import (
    TRAIN_DATA_PATH, TEST_DATA_PATH, CACHE_DIR,
    SEQ_LENGTH, PRED_HORIZON, TARGET_COLUMNS, TARGET_TYPE,
    BATCH_SIZE, NUM_WORKERS, PIN_MEMORY
)
from feature_config import get_feature_groups, get_feature_dims

# 模型保存路径
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# 模型保存路径
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# 创建必要目录
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==================== 设备配置 ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = True  # 混合精度训练（需要CUDA）

# ==================== 任务配置 ====================
# 从dataset_loader导入的配置
# TARGET_TYPE: 'regression' 或 'classification'
# TARGET_COLUMNS: 目标变量列名
# NUM_CLASSES: 分类任务的类别数

# 如果是分类任务，需要根据实际数据确定类别数
NUM_CLASSES = 2  # 示例：二分类任务

# ==================== 使用dataset_loader的数据 ====================
# 方案：使用dataset_loader的PyTorch Dataset和DataLoader
# 优势：
# 1. 自动处理时序序列（滑动窗口）
# 2. 特征已经分组（categorical, numerical, cyclical, binary）
# 3. 支持缓存机制（pkl文件）
# 4. 按城市分组，保证时序完整性

# 导入示例（在训练脚本中使用）：
# from data_loader import get_train_dataloader, get_test_dataloader
# train_loader = get_train_dataloader(batch_size=BATCH_SIZE)
# test_loader = get_test_dataloader(batch_size=BATCH_SIZE)

# ==================== Wide & Deep 架构配置 ====================

# Wide 侧配置
WIDE_CONFIG = {
    "features": [  # Wide侧使用的原始特征
        "pressure",  # 当前气压
        "temperature",  # 当前温度
        "humidity",  # 湿度
        "wind_speed",  # 风速
    ],
    "cross_features": [  # 交叉特征（需要在数据预处理时生成）
        "wind_direction_x_month",  # 风向×月份
        "temperature_x_humidity",  # 温度×湿度
        "pressure_x_pressure_change",  # 气压×气压变化率
        "wind_speed_x_wind_direction",  # 风速×风向
    ]
}

# Deep 侧配置
DEEP_CONFIG = {
    "numerical_features": [  # 数值特征（需要标准化）
        "pressure", "temperature", "humidity", "wind_speed",
        "pressure_diff_3h", "temperature_diff_3h",
        "dew_point", "visibility"
    ],
    "categorical_features": {  # 类别特征及其Embedding维度
        "wind_direction": {"vocab_size": 16, "embed_dim": 8},
        "month": {"vocab_size": 12, "embed_dim": 4},
        "hour": {"vocab_size": 24, "embed_dim": 6},
    },
    "hidden_layers": [128, 64, 32],  # DNN隐藏层维度
    "dropout_rates": [0.3, 0.2, 0.1],  # 每层的Dropout率
    "use_batch_norm": True  # 是否使用BatchNorm
}

# 模型架构参数
MODEL_CONFIG = {
    "wide_dim": len(WIDE_CONFIG["features"]) + len(WIDE_CONFIG["cross_features"]),
    "deep_dim": DEEP_CONFIG["hidden_layers"][-1],  # Deep侧最后一层的维度
    "output_dim": NUM_CLASSES if TASK_TYPE == "classification" else 1
}


# ==================== 训练配置 ====================
TRAIN_CONFIG = {
    "batch_size": 256,
    "num_epochs": 100,
    "learning_rate": 0.001,
    "weight_decay": 1e-5,  # L2正则化
    "gradient_clip": 1.0,  # 梯度裁剪
    "early_stopping_patience": 10,  # 早停耐心值
    "val_split": 0.2,  # 验证集比例
    "random_state": 42,
    "num_workers": 4,  # DataLoader工作进程数
    "pin_memory": True  # 加速数据传输到GPU
}

# 优化器配置
OPTIMIZER_CONFIG = {
    "type": "adam",  # 'adam' 或 'adamw'
    "lr": TRAIN_CONFIG["learning_rate"],
    "betas": (0.9, 0.999),
    "eps": 1e-8,
    "weight_decay": TRAIN_CONFIG["weight_decay"]
}

# 学习率调度器配置
SCHEDULER_CONFIG = {
    "type": "reduce_on_plateau",  # 'reduce_on_plateau', 'cosine', 'step'
    "mode": "min",
    "factor": 0.5,  # 学习率衰减因子
    "patience": 5,  # 等待epoch数
    "min_lr": 1e-6
}

# 损失函数配置
LOSS_CONFIG = {
    "classification": "cross_entropy",  # 'bce' 或 'cross_entropy'
    "regression": "mse",  # 'mse' 或 'mae'
    "class_weights": None  # 类别权重（处理不平衡）
}


# ==================== 特征工程配置 ====================
FEATURE_CONFIG = {
    "exclude_columns": ["date", "city", "country"],
    "normalize_numerical": True,  # 标准化数值特征
    "create_cross_features": True,  # 创建交叉特征
    "handle_missing": "mean"  # 缺失值处理策略
}

# ==================== 日志与监控配置 ====================
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_file": os.path.join(RESULTS_DIR, "training.log"),
    "tensorboard_dir": os.path.join(RESULTS_DIR, "tensorboard"),
    "log_interval": 10  # 每多少个batch打印一次日志
}

# ==================== 评估配置 ====================
EVAL_CONFIG = {
    "metrics": ["accuracy", "precision", "recall", "f1"],  # 分类指标
    "save_confusion_matrix": True,
    "save_predictions": True,
    "predictions_file": os.path.join(RESULTS_DIR, "predictions.csv")
}

# ==================== 模型保存配置 ====================
SAVE_CONFIG = {
    "save_best_only": True,  # 只保存最佳模型
    "monitor": "val_loss",  # 监控指标
    "mode": "min",  # 'min' 或 'max'
    "save_checkpoint_every": 5  # 每N个epoch保存一次检查点
}
