"""
Linear Baseline 模型配置文件
"""
from pathlib import Path

# ==================== 路径配置 ====================
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "data"

# 选择使用 daily 或 hourly 数据
DATA_TYPE = 'daily'  # 'daily' 或 'hourly'

# 数据路径
if DATA_TYPE == 'daily':
    TRAIN_DATA_PATH = DATA_ROOT / "data_engineer" / "daily_data" / "processed_data" / "train_features.csv"
    TEST_DATA_PATH = DATA_ROOT / "data_engineer" / "daily_data" / "processed_data" / "test_features.csv"
elif DATA_TYPE == 'hourly':
    TRAIN_DATA_PATH = DATA_ROOT / "data_engineer" / "hourly_data" / "processed_data" / "train_features.csv"
    TEST_DATA_PATH = DATA_ROOT / "data_engineer" / "hourly_data" / "processed_data" / "test_features.csv"
else:
    raise ValueError(f"DATA_TYPE 必须是 'daily' 或 'hourly'，得到: {DATA_TYPE}")

# 输出路径
MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ==================== 任务配置 ====================

# 分类任务目标列
CLASSIFICATION_TARGETS = {
    'rain': 'is_rainy',      # 预测是否下雨
    'snow': 'is_snowy',      # 预测是否下雪
    'severe': 'is_severe_weather',  # 预测是否恶劣天气
}

# 默认分类目标
DEFAULT_CLASSIFICATION_TARGET = 'is_rainy'

# 回归任务目标列
REGRESSION_TARGETS = {
    'temp_mean': 'temperature_2m_mean',
    'temp_max': 'temperature_2m_max',
    'temp_min': 'temperature_2m_min',
    'temp_range': 'temperature_range',           # 温度范围 (max - min)
    'temp_feels': 'feels_like_temperature',      # 体感温度
    'precipitation': 'precipitation_sum',
    'wind': 'wind_speed_10m_max',
}

# 默认回归目标
DEFAULT_REGRESSION_TARGET = 'temperature_2m_mean'

# ==================== 特征配置 ====================

# 需要排除的列（不作为特征）
EXCLUDED_COLUMNS = [
    'time',           # 时间戳
    'city',           # 原始城市名
    'country',        # 原始国家名
    'weather_code',   # 原始天气代码（使用 weather_code_id）
    # Hourly/Daily data 中会导致数据泄露的列
    'rain',           # 降雨量（与 is_rainy 直接相关）
    'rain_sum',       # 降雨总量
    'snowfall',       # 降雪量（与 is_snowy 直接相关）
    'snow_sum',       # 降雪总量
    'precipitation',  # 总降水（与 is_rainy 直接相关）
    'precipitation_sum',  # 总降水
    'precipitation_intensity',  # 降水强度（派生自 precipitation）
    'solid_precip',   # 固态降水（派生自 rain/snowfall）
    'is_snow',        # 降雪标记（派生自 snowfall）
    'is_heavy_rain',  # 暴雨标记（派生自 rain）
    'precipitation_level',  # 降水等级（派生自 precipitation）
]

# 目标列（会在训练时自动排除）
TARGET_COLUMNS = list(CLASSIFICATION_TARGETS.values()) + list(REGRESSION_TARGETS.values())

# ==================== 模型参数 ====================

# Logistic Regression 参数
LOGISTIC_PARAMS = {
    'C': 1.0,                    # 正则化强度的倒数（越小正则化越强）
    'max_iter': 1000,            # 最大迭代次数
    'random_state': 42,
    'solver': 'lbfgs',           # 优化算法
    'n_jobs': -1,                # 使用所有CPU核心
    'class_weight': 'balanced',  # 自动平衡类别权重
}

# Ridge Regression 参数
RIDGE_PARAMS = {
    'alpha': 1.0,                # 正则化强度（越大正则化越强）
    'random_state': 42,
    'solver': 'auto',
}

# ==================== 训练配置 ====================

# 是否使用数据采样（用于 hourly 大数据集）
USE_SAMPLING = False
SAMPLE_FRACTION = 0.1  # 采样比例（如果 USE_SAMPLING=True）

# 随机种子
RANDOM_STATE = 42

# 是否保存特征重要性
SAVE_FEATURE_IMPORTANCE = True

# ==================== 评估配置 ====================

# 分类评估指标
CLASSIFICATION_METRICS = [
    'accuracy',
    'precision',
    'recall',
    'f1',
    'roc_auc',
]

# 回归评估指标
REGRESSION_METRICS = [
    'mae',       # Mean Absolute Error
    'rmse',      # Root Mean Squared Error
    'r2',        # R-squared
    'mape',      # Mean Absolute Percentage Error
]

# 是否生成可视化图表
GENERATE_PLOTS = False  # 暂时禁用，避免matplotlib依赖问题

# ==================== 预测配置 ====================

# 预测结果保存路径
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
PREDICTIONS_DIR.mkdir(exist_ok=True)

# 是否保存预测概率（分类任务）
SAVE_PROBABILITIES = True

# ==================== 日志配置 ====================

# 是否显示详细日志
VERBOSE = True

# 日志文件路径
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / "training.log"

# ==================== 模型文件名 ====================

def get_model_path(task_type, target_name):
    """获取模型保存路径"""
    if task_type == 'classification':
        return MODEL_DIR / f"logistic_{target_name}.pkl"
    elif task_type == 'regression':
        return MODEL_DIR / f"ridge_{target_name}.pkl"
    else:
        raise ValueError(f"task_type 必须是 'classification' 或 'regression'")

def get_feature_names_path():
    """获取特征名称保存路径"""
    return MODEL_DIR / "feature_names.json"

def get_results_path(task_type, target_name):
    """获取结果保存路径"""
    return RESULTS_DIR / f"{task_type}_{target_name}_report.txt"
