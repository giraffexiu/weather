"""
配置文件：定义特征工程的所有参数
"""
import os
from pathlib import Path

# ==================== 路径配置 ====================
# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "data"

# 输入数据路径
INPUT_DATA_PATH = DATA_ROOT / "data_clean" / "cleaned_data" / "weather_daily_cleaned.csv"

# 输出路径
OUTPUT_DIR = Path(__file__).parent / "processed_data"
OUTPUT_DIR.mkdir(exist_ok=True)

TRAIN_OUTPUT_PATH = OUTPUT_DIR / "train_features.csv"
TEST_OUTPUT_PATH = OUTPUT_DIR / "test_features.csv"

# 预处理对象保存路径
PREPROCESSOR_DIR = OUTPUT_DIR / "preprocessors"
PREPROCESSOR_DIR.mkdir(exist_ok=True)

SCALER_PATH = PREPROCESSOR_DIR / "scaler.pkl"
CITY_MAPPING_PATH = PREPROCESSOR_DIR / "city_mapping.json"
COUNTRY_MAPPING_PATH = PREPROCESSOR_DIR / "country_mapping.json"

# ==================== 时间切分配置 ====================
TRAIN_END_DATE = "2023-12-31"
TEST_START_DATE = "2024-01-01"

# ==================== 特征列定义 ====================
# 原始数值特征
NUMERICAL_FEATURES = [
    'latitude',
    'longitude',
    'temperature_2m_max',
    'temperature_2m_min',
    'temperature_2m_mean',
    'precipitation_sum',
    'rain_sum',
    'wind_speed_10m_max',
    'shortwave_radiation_sum'
]

# 类别特征
CATEGORICAL_FEATURES = [
    'city',
    'country'
]

# 时间特征
TIME_COLUMN = 'time'

# ==================== 特征工程配置 ====================
# 是否生成周期性编码
USE_CYCLICAL_ENCODING = True

# 是否生成派生特征
CREATE_DERIVED_FEATURES = True

# 标准化方法: 'standard' or 'minmax'
SCALING_METHOD = 'standard'

# ==================== 随机种子 ====================
RANDOM_SEED = 42
