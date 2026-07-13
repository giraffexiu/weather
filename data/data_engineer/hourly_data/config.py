"""
配置文件：定义小时天气数据特征工程的所有参数
"""
from pathlib import Path

# ==================== 路径配置 ====================
# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "data"

# 输入数据路径
INPUT_DATA_PATH = DATA_ROOT / "data_clean" / "cleaned_data" / "weather_hourly_cleaned.csv"

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
WEATHER_CODE_MAPPING_PATH = PREPROCESSOR_DIR / "weather_code_mapping.json"

# ==================== 时间切分配置 ====================
TRAIN_END_DATE = "2023-12-31"
TEST_START_DATE = "2024-01-01"

# ==================== 特征列定义 ====================
# 原始数值特征（小时数据包含的瞬时气象指标）
NUMERICAL_FEATURES = [
    'latitude',
    'longitude',
    'temperature_2m',
    'apparent_temperature',
    'relative_humidity_2m',
    'pressure_msl',
    'precipitation',
    'rain',
    'snowfall',
    'cloud_cover',
    'wind_speed_10m',
    'wind_direction_10m',
    'wind_gusts_10m',
    'shortwave_radiation'
]

# 类别特征（小时数据比日数据多了 weather_code）
CATEGORICAL_FEATURES = [
    'city',
    'country',
    'weather_code'
]

# 时间特征
TIME_COLUMN = 'time'

# ==================== 特征工程配置 ====================
# 是否生成周期性编码（小时数据额外需要小时周期编码）
USE_CYCLICAL_ENCODING = True

# 是否生成派生特征
CREATE_DERIVED_FEATURES = True

# 标准化方法: 'standard' or 'minmax'
SCALING_METHOD = 'standard'

# ==================== WMO 天气编码含义 ====================
WMO_WEATHER_MAP = {
    0:  "晴天 Clear",
    1:  "部分多云 Partly Cloudy",
    2:  "多云 Cloudy",
    3:  "阴天 Overcast",
    51: "小毛毛雨 Light Drizzle",
    53: "中毛毛雨 Moderate Drizzle",
    55: "大毛毛雨 Dense Drizzle",
    61: "小雨 Light Rain",
    63: "中雨 Moderate Rain",
    65: "大雨 Heavy Rain",
    71: "小雪 Light Snow",
    73: "中雪 Moderate Snow",
    75: "大雪 Heavy Snow",
}

# ==================== 随机种子 ====================
RANDOM_SEED = 42
