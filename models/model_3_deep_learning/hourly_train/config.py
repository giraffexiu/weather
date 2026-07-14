"""
Wide & Deep 小时天气预测 — 模型/训练配置
"""
from pathlib import Path

# ==================== 路径 ====================
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"
CHECKPOINT_PATH = CHECKPOINT_DIR / "best_model.pt"

# ==================== 模型超参数 ====================
MODEL_CONFIG = {
    'deep_hidden_dims': [512, 256, 128],
    'dropout': 0.3,
    'num_targets': 5,          # temperature_2m, precipitation, wind_speed_10m, apparent_temperature, relative_humidity_2m
}

# ==================== 训练超参数 ====================
TRAIN_CONFIG = {
    'batch_size': 512,
    'epochs': 50,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'grad_clip': 5.0,
    'patience': 10,            # early stopping
    'scheduler_factor': 0.5,
    'scheduler_patience': 5,
    'num_workers': 0,          # CPU 建议 0
    'prefetch_factor': 2,
    'log_interval': 2,         # 每 N 个 epoch 打印一次
}

# ==================== 数据配置 ====================
DATA_CONFIG = {
    'seq_length': 24,          # 过去24小时
    'group_by_city': False,    # 关闭以提速
    'use_cache': True,
}
