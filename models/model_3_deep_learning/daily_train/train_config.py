"""
Wide & Deep 模型配置
"""
import torch
from pathlib import Path

# ==================== 路径配置 ====================
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

PLOT_DIR = OUTPUT_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)

LOG_DIR = OUTPUT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

CHECKPOINT_PATH = CHECKPOINT_DIR / "best_model.pth"

# ==================== 模型配置 ====================
MODEL_CONFIG = {
    # Embedding维度
    'city_embed_dim': 8,
    'country_embed_dim': 8,
    'season_embed_dim': 4,
    
    # Wide侧维度 (实际14维)
    'wide_dim': 14,
    
    # Deep侧维度
    'deep_dim': 123,  # 20(embed) + 88(num) + 6(cyc) + 9(bin)
    'hidden_dims': [128, 64, 32],
    'dropout': 0.3,
    'use_batch_norm': True,
    
    # 输出目标数量（9个天气变量）
    'num_targets': 9,  # temperature(5) + precipitation(3) + wind(1)
}

# ==================== 训练配置 ====================
TRAIN_CONFIG = {
    'batch_size': 64,
    'epochs': 80,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'grad_clip_norm': 1.0,
    
    # 学习率调度
    'use_scheduler': True,
    'scheduler_type': 'cosine',  # 'cosine' or 'step'
    'T_max': 80,
    
    # 早停
    'early_stopping': True,
    'patience': 15,
    
    # 日志
    'log_interval': 50,  # 每50个batch打印一次
    'eval_interval': 1,  # 每1个epoch评估一次
}

# ==================== 数据配置 ====================
DATA_CONFIG = {
    'batch_size': 64,
    'num_workers': 4,
    'pin_memory': True,
}

# ==================== 特征索引 ====================
# numerical特征在第二维的索引位置
NUMERICAL_FEATURE_INDEX = {
    'latitude': 0,
    'longitude': 1,
    'temperature_2m_max': 2,
    'temperature_2m_min': 3,
    'temperature_2m_mean': 4,
    'temperature_range': 5,
    'feels_like_temperature': 6,
    'precipitation_sum': 7,
    'rain_sum': 8,
    'snow_sum': 9,
    'wind_speed_10m_max': 10,
    'shortwave_radiation_sum': 11,
    'month': 12,
    'year': 13,
    'day': 14,
    'day_of_year': 15,
    'day_of_week': 16,
    'quarter': 17,
    'week_of_year': 18,
}

# binary特征索引
BINARY_FEATURE_INDEX = {
    'is_freezing': 0,
    'is_hot_day': 1,
    'is_rainy': 2,
    'is_heavy_rain': 3,
    'is_snowy': 4,
    'is_windy': 5,
    'is_sunny': 6,
    'is_severe_weather': 7,
    'is_high_latitude': 8,
}

# ==================== 设备配置 ====================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42

# ==================== 完整配置 ====================
CONFIG = {
    **MODEL_CONFIG,
    **TRAIN_CONFIG,
    **DATA_CONFIG,
    'device': DEVICE,
    'seed': SEED,
    'checkpoint_path': CHECKPOINT_PATH,
    'plot_dir': PLOT_DIR,
    'log_dir': LOG_DIR,
}
