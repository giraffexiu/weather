"""
Hourly Ensemble 配置文件
基于现有 hourly Wide & Deep 模型的集成配置
"""
from pathlib import Path
import sys

# ==================== 路径配置 ====================
BASE_DIR = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent.parent

# Model 3 Wide & Deep (hourly) 路径
MODEL3_DIR = PROJECT_ROOT / "models" / "model_3_deep_learning" / "hourly_train"
MODEL3_CHECKPOINT = MODEL3_DIR / "output" / "checkpoints" / "best_model.pt"

# 数据路径
DATA_DIR = PROJECT_ROOT / "data" / "data_engineer" / "hourly_data"
PROCESSED_DATA_DIR = DATA_DIR / "processed_data"
TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train_features.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test_features.csv"

# 输出路径
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

RESULTS_DIR = OUTPUT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
PREDICTIONS_DIR.mkdir(exist_ok=True)

PLOTS_DIR = OUTPUT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# ==================== 目标变量（小时数据 5 目标） ====================
HOUR_TARGET_COLUMNS = [
    'temperature_2m',
    'precipitation',
    'wind_speed_10m',
    'apparent_temperature',
    'relative_humidity_2m',
]

# Model 3 输出的 5 目标索引
MODEL3_TARGET_INDICES = {col: i for i, col in enumerate(HOUR_TARGET_COLUMNS)}

# Model 3 性能（训练结果）
MODEL3_PERFORMANCE = {
    'temperature_2m':         {'mse': 0.009, 'mae': 0.071, 'r2': 0.991},
    'precipitation':          {'mse': 0.940, 'mae': 0.255, 'r2': 0.394},
    'wind_speed_10m':         {'mse': 0.093, 'mae': 0.224, 'r2': 0.911},
    'apparent_temperature':   {'mse': 0.010, 'mae': 0.075, 'r2': 0.990},
    'relative_humidity_2m':   {'mse': 0.043, 'mae': 0.151, 'r2': 0.953},
}

# ==================== 分类概率转换配置 ====================
# 从回归输出推导分类概率
PROBABILITY_CONVERSION_CONFIG = {
    'rain': {
        'method': 'threshold_based',
        'threshold': 0.1,
        'scale': 10.0,
        'source_target': 'precipitation',    # 映射自降水量
    },
    'freezing': {
        'method': 'threshold_based',
        'threshold': 0.0,
        'scale': 5.0,
        'source_target': 'temperature_2m',   # 映射自温度
    },
    'windy': {
        'method': 'threshold_based',
        'threshold': 5.0,
        'scale': 15.0,
        'source_target': 'wind_speed_10m',   # 映射自风速
    },
}

# ==================== 设备配置 ====================
import torch
DEVICE = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)

# ==================== 随机种子 ====================
RANDOM_SEED = 42

# ==================== 评估配置 ====================
EVALUATION_METRICS = {
    'regression': ['mse', 'mae', 'r2'],
}

# ==================== 路径添加 ====================
def setup_paths():
    """添加必要的模块路径"""
    paths_to_add = [
        str(DATA_DIR),
        str(MODEL3_DIR),
        str(MODEL3_DIR.parent),  # model_3_deep_learning/
    ]
    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)

# ==================== 验证配置 ====================
def validate_config():
    """验证配置的完整性"""
    errors = []
    if not MODEL3_CHECKPOINT.exists():
        errors.append(f"Model 3 checkpoint not found: {MODEL3_CHECKPOINT}")
    if not TEST_DATA_PATH.exists():
        errors.append(f"Test data not found: {TEST_DATA_PATH}")
    if errors:
        raise FileNotFoundError("\n".join(errors))
    return True


if __name__ == "__main__":
    try:
        validate_config()
        print("Hourly Ensemble 配置验证通过！")
        print(f"\n设备: {DEVICE}")
        print(f"Model 3 检查点: {MODEL3_CHECKPOINT}")
        print(f"测试数据: {TEST_DATA_PATH}")
        print(f"目标变量: {HOUR_TARGET_COLUMNS}")
    except Exception as e:
        print(f"配置验证失败:\n{e}")
