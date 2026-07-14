"""
Daily Ensemble 配置文件
软投票集成配置：Model 1 (Linear) + Model 3 (Wide & Deep)
"""
from pathlib import Path
import sys

# ==================== 路径配置 ====================
BASE_DIR = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent.parent

# Model 1 (Linear) 路径
MODEL1_DIR = PROJECT_ROOT / "models" / "model_1_linear"
MODEL1_MODELS_DIR = MODEL1_DIR / "models"
MODEL1_FEATURE_NAMES = MODEL1_MODELS_DIR / "feature_names.json"

# Model 3 (Wide & Deep) 路径
MODEL3_DIR = PROJECT_ROOT / "models" / "model_3_deep_learning" / "daily_train"
MODEL3_CHECKPOINT = MODEL3_DIR / "outputs" / "checkpoints" / "best_model.pth"

# 数据路径
DATA_DIR = PROJECT_ROOT / "data" / "data_engineer" / "daily_data"
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

# ==================== Model 1 配置 ====================
# 分类任务模型（输出概率）
MODEL1_CLASSIFICATION_TASKS = {
    'rain': {
        'model_path': MODEL1_MODELS_DIR / "logistic_rain.pkl",
        'target_column': 'is_rainy',
        'f1_score': 0.7737,  # 从训练结果
        'auc': 0.8489
    },
    'snow': {
        'model_path': MODEL1_MODELS_DIR / "logistic_snow.pkl",
        'target_column': 'is_snowy',
        'f1_score': 0.3490,
        'auc': 0.9159
    },
    'severe': {
        'model_path': MODEL1_MODELS_DIR / "logistic_severe.pkl",
        'target_column': 'is_severe_weather',
        'f1_score': 0.8449,
        'auc': 0.9614
    }
}

# 回归任务模型（输出连续值）
MODEL1_REGRESSION_TASKS = {
    'temp_mean': {
        'model_path': MODEL1_MODELS_DIR / "ridge_temp_mean.pkl",
        'target_column': 'temperature_2m_mean',
        'r2': 0.8637,
        'mae': 0.2863,
        'rmse': 0.3642
    },
    'temp_max': {
        'model_path': MODEL1_MODELS_DIR / "ridge_temp_max.pkl",
        'target_column': 'temperature_2m_max',
        'r2': 0.8620,
        'mae': 0.2907,
        'rmse': 0.3681
    },
    'temp_min': {
        'model_path': MODEL1_MODELS_DIR / "ridge_temp_min.pkl",
        'target_column': 'temperature_2m_min',
        'r2': 0.8225,
        'mae': 0.3245,
        'rmse': 0.4123
    },
    'temp_range': {
        'model_path': MODEL1_MODELS_DIR / "ridge_temp_range.pkl",
        'target_column': 'temperature_range',
        'r2': 0.5487,
        'mae': 0.5170,
        'rmse': 0.6536
    },
    'temp_feels': {
        'model_path': MODEL1_MODELS_DIR / "ridge_temp_feels.pkl",
        'target_column': 'feels_like_temperature',
        'r2': 0.8650,
        'mae': 0.2844,
        'rmse': 0.3616
    },
    'precipitation': {
        'model_path': MODEL1_MODELS_DIR / "ridge_precipitation.pkl",
        'target_column': 'precipitation_sum',
        'r2': 0.2679,
        'mae': 0.5441,
        'rmse': 1.0030
    },
    'wind': {
        'model_path': MODEL1_MODELS_DIR / "ridge_wind.pkl",
        'target_column': 'wind_speed_10m_max',
        'r2': 0.8979,
        'mae': 0.2696,
        'rmse': 0.3274
    }
}

# ==================== Model 3 配置 ====================
# Model 3 输出的9个目标索引
MODEL3_TARGET_INDICES = {
    'temperature_2m_max': 0,
    'temperature_2m_min': 1,
    'temperature_2m_mean': 2,
    'temperature_range': 3,
    'feels_like_temperature': 4,
    'precipitation_sum': 5,
    'rain_sum': 6,
    'snow_sum': 7,
    'wind_speed_10m_max': 8
}

# Model 3 性能（需要通过评估获得，这里使用占位符）
MODEL3_PERFORMANCE = {
    'temperature_2m_max': {'r2': None, 'mae': None, 'rmse': None},
    'temperature_2m_min': {'r2': None, 'mae': None, 'rmse': None},
    'temperature_2m_mean': {'r2': None, 'mae': None, 'rmse': None},
    'temperature_range': {'r2': None, 'mae': None, 'rmse': None},
    'feels_like_temperature': {'r2': None, 'mae': None, 'rmse': None},
    'precipitation_sum': {'r2': None, 'mae': None, 'rmse': None},
    'rain_sum': {'r2': None, 'mae': None, 'rmse': None},
    'snow_sum': {'r2': None, 'mae': None, 'rmse': None},
    'wind_speed_10m_max': {'r2': None, 'mae': None, 'rmse': None}
}

# ==================== 集成配置 ====================
# 权重计算方法
WEIGHT_METHOD = 'performance_based'  # 'equal', 'performance_based'

# 回归任务使用的性能指标
REGRESSION_METRIC_FOR_WEIGHT = 'r2'  # 'r2', 'mae', 'rmse'

# 分类任务使用的性能指标
CLASSIFICATION_METRIC_FOR_WEIGHT = 'f1'  # 'f1', 'auc'

# 概率转换配置（Model 3 回归值 -> 分类概率）
PROBABILITY_CONVERSION_CONFIG = {
    'rain': {
        'method': 'threshold_based',  # 'threshold_based', 'sigmoid'
        'threshold': 0.1,  # rain_sum > 0.1mm 认为有雨
        'scale': 10.0      # 用于sigmoid的缩放因子
    },
    'snow': {
        'method': 'threshold_based',
        'threshold': 0.1,  # snow_sum > 0.1mm 认为有雪
        'scale': 10.0
    },
    'severe': {
        'method': 'composite',  # 综合多个指标
        'thresholds': {
            'temp_range': 15.0,  # 温度变化大
            'wind_speed': 10.0,   # 强风
            'precipitation': 5.0  # 强降水
        },
        'weights': {
            'temp_range': 0.3,
            'wind_speed': 0.4,
            'precipitation': 0.3
        }
    }
}

# ==================== 任务对齐配置 ====================
# 定义哪些任务可以在两个模型之间集成
ENSEMBLE_TASKS = {
    # 回归任务（直接加权平均）
    'regression': {
        'temp_mean': {
            'model1_key': 'temp_mean',
            'model3_index': MODEL3_TARGET_INDICES['temperature_2m_mean']
        },
        'temp_max': {
            'model1_key': 'temp_max',
            'model3_index': MODEL3_TARGET_INDICES['temperature_2m_max']
        },
        'temp_min': {
            'model1_key': 'temp_min',
            'model3_index': MODEL3_TARGET_INDICES['temperature_2m_min']
        },
        'temp_range': {
            'model1_key': 'temp_range',
            'model3_index': MODEL3_TARGET_INDICES['temperature_range']
        },
        'temp_feels': {
            'model1_key': 'temp_feels',
            'model3_index': MODEL3_TARGET_INDICES['feels_like_temperature']
        },
        'precipitation': {
            'model1_key': 'precipitation',
            'model3_index': MODEL3_TARGET_INDICES['precipitation_sum']
        },
        'wind': {
            'model1_key': 'wind',
            'model3_index': MODEL3_TARGET_INDICES['wind_speed_10m_max']
        }
    },
    # 分类任务（概率加权平均）
    'classification': {
        'rain': {
            'model1_key': 'rain',
            'model3_indices': {
                'rain_sum': MODEL3_TARGET_INDICES['rain_sum']
            }
        },
        'snow': {
            'model1_key': 'snow',
            'model3_indices': {
                'snow_sum': MODEL3_TARGET_INDICES['snow_sum']
            }
        },
        'severe': {
            'model1_key': 'severe',
            'model3_indices': {
                'temp_range': MODEL3_TARGET_INDICES['temperature_range'],
                'wind_speed': MODEL3_TARGET_INDICES['wind_speed_10m_max'],
                'precipitation': MODEL3_TARGET_INDICES['precipitation_sum']
            }
        }
    }
}

# ==================== 设备配置 ====================
import torch
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 
                     'cuda' if torch.cuda.is_available() else 'cpu')

# ==================== 随机种子 ====================
RANDOM_SEED = 42

# ==================== 评估配置 ====================
EVALUATION_METRICS = {
    'classification': ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
    'regression': ['mae', 'rmse', 'r2', 'mape']
}

# ==================== 路径添加 ====================
# 添加必要的路径到 sys.path
def setup_paths():
    """添加必要的模块路径"""
    paths_to_add = [
        str(MODEL1_DIR),  # Model 1 配置
        str(MODEL3_DIR),  # Model 3 模块
        str(DATA_DIR),    # DataLoader
    ]
    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)

# ==================== 验证配置 ====================
def validate_config():
    """验证配置的完整性"""
    errors = []
    
    # 检查 Model 1 模型文件
    for task_name, task_config in MODEL1_CLASSIFICATION_TASKS.items():
        if not task_config['model_path'].exists():
            errors.append(f"Model 1 classification model not found: {task_config['model_path']}")
    
    for task_name, task_config in MODEL1_REGRESSION_TASKS.items():
        if not task_config['model_path'].exists():
            errors.append(f"Model 1 regression model not found: {task_config['model_path']}")
    
    # 检查 Model 1 特征名称
    if not MODEL1_FEATURE_NAMES.exists():
        errors.append(f"Model 1 feature names not found: {MODEL1_FEATURE_NAMES}")
    
    # 检查 Model 3 模型文件
    if not MODEL3_CHECKPOINT.exists():
        errors.append(f"Model 3 checkpoint not found: {MODEL3_CHECKPOINT}")
    
    # 检查数据文件
    if not TEST_DATA_PATH.exists():
        errors.append(f"Test data not found: {TEST_DATA_PATH}")
    
    if errors:
        raise FileNotFoundError("\n".join(errors))
    
    return True


if __name__ == "__main__":
    # 验证配置
    try:
        validate_config()
        print("✅ 配置验证通过！")
        print(f"\n设备: {DEVICE}")
        print(f"Model 1 目录: {MODEL1_DIR}")
        print(f"Model 3 检查点: {MODEL3_CHECKPOINT}")
        print(f"测试数据: {TEST_DATA_PATH}")
    except Exception as e:
        print(f"❌ 配置验证失败:\n{e}")
