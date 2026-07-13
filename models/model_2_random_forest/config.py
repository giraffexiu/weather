"""
随机森林模型配置 (Model 2: Random Forest)
双粒度天气分类：日级 + 小时级，均为 6 类天气分类
"""
import os
from pathlib import Path

# ==================== 路径配置 ====================
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "data"

# 特征工程产出路径
DAILY_TRAIN_PATH = DATA_ROOT / "data_engineer" / "daily_data" / "processed_data" / "train_features.csv"
DAILY_TEST_PATH = DATA_ROOT / "data_engineer" / "daily_data" / "processed_data" / "test_features.csv"
HOURLY_TRAIN_PATH = DATA_ROOT / "data_engineer" / "hourly_data" / "processed_data" / "train_features.csv"
HOURLY_TEST_PATH = DATA_ROOT / "data_engineer" / "hourly_data" / "processed_data" / "test_features.csv"

# 日级标签来源（小时级清洗数据，含 weather_code）
HOURLY_CLEANED_PATH = DATA_ROOT / "data_clean" / "cleaned_data" / "weather_hourly_cleaned.csv"

# 模型保存路径
MODEL_DIR = Path(__file__).parent / "saved_models"
MODEL_DIR.mkdir(exist_ok=True)
DAILY_MODEL_PATH = MODEL_DIR / "rf_daily.pkl"
HOURLY_MODEL_PATH = MODEL_DIR / "rf_hourly.pkl"
FEATURE_CONFIG_PATH = MODEL_DIR / "feature_config.json"

# 输出路径
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
DAILY_OUTPUT_DIR = OUTPUT_DIR / "daily"
HOURLY_OUTPUT_DIR = OUTPUT_DIR / "hourly"
DAILY_OUTPUT_DIR.mkdir(exist_ok=True)
HOURLY_OUTPUT_DIR.mkdir(exist_ok=True)

# ==================== 6 类天气定义 ====================
WEATHER_CATEGORIES = ["Clear", "Cloudy", "Overcast", "Drizzle", "Rain", "Snow"]

# WMO weather_code -> category 映射
WMO_TO_CATEGORY = {
    0: "Clear",
    1: "Cloudy",
    2: "Cloudy",
    3: "Overcast",
    51: "Drizzle", 53: "Drizzle", 55: "Drizzle",
    61: "Rain", 63: "Rain", 65: "Rain",
    71: "Snow", 73: "Snow", 75: "Snow",
}

# 日级标签聚合策略：当天最严重天气（优先级从高到低）
DAILY_PRIORITY = ["Snow", "Rain", "Drizzle", "Overcast", "Cloudy", "Clear"]

# ==================== 随机种子 ====================
RANDOM_SEED = 42

# ==================== 排除列（不作为特征使用） ====================
# 这些列是标识/时间/标签列，不能作为特征
# season 和 day_period 是字符串列，由 data_loader._encode_categorical 编码为整数后保留为特征
EXCLUDE_COLS_BASE = [
    "city", "country", "time", "weather_code", "weather_code_id",
    "target",  # 标签列
]

# 日级数据独有的排除列
DAILY_EXCLUDE_EXTRA = []
# 小时级数据独有的排除列
HOURLY_EXCLUDE_EXTRA = []

# ==================== 时序特征增强（滞后/滑动窗口） ====================
# 滞后阶数：小时级覆盖日周期(24h)，日级覆盖月周期(30d)
LAG_PERIODS_HOURLY = [1, 2, 3, 6, 12, 24]
LAG_PERIODS_DAILY = [1, 2, 3, 7, 14, 30]
# 需要构造滞后特征的列
LAG_COLS_HOURLY = ["temperature_2m", "pressure_msl", "relative_humidity_2m",
                   "precipitation", "cloud_cover", "wind_speed_10m"]
LAG_COLS_DAILY = ["temperature_2m_mean", "precipitation_sum", "wind_speed_10m_max",
                  "pressure_mean", "humidity_mean", "cloud_cover_mean",
                  "today_weather_cat"]

# 多尺度滑动窗口
ROLLING_WINDOWS_HOURLY = [3, 7, 12, 24]   # 短期变化 + 半日 + 全天
ROLLING_WINDOWS_DAILY = [3, 7, 14, 30]     # 短期 + 周 + 半月 + 月
# 滚动统计类型
ROLLING_STATS = ["mean", "std"]

# 气压变化率周期
HOURLY_PRESSURE_CHANGE_PERIODS = [1, 3, 6]
DAILY_PRESSURE_CHANGE_PERIOD = 1  # 前一天的气压 - 今天的气压

# Cloudy 专项区分特征
CLOUD_FEATURE_COLS_HOURLY = ["cloud_cover"]  # 用于构造云量变化率/趋势
CLOUD_FEATURE_COLS_DAILY = ["cloud_cover_mean"]

# 特征交互列
INTERACTION_FEATURES_HOURLY = [
    "temp_humidity",        # 温度×湿度
    "pressure_wind",        # 气压×风向
    "cloud_rad_ratio",      # 云量/辐射比
]
INTERACTION_FEATURES_DAILY = [
    "temp_humidity_daily",
    "pressure_wind_daily",
    "cloud_rad_ratio_daily",
]

# ==================== RandomForest 超参数网格 ====================
# 扩展网格：更多树 + 更多正则化参数
PARAM_GRID = {
    "n_estimators": [200, 300, 500],
    "max_depth": [20, 30, None],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["log2", 0.3, 0.5],
}

# 固定参数
RF_FIXED_PARAMS = {
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "class_weight": "balanced",
    "oob_score": True,
    "bootstrap": True,
}

# 小时级少数类手动加权（Rain/Snow 仅 1.7%）
HOURLY_CLASS_WEIGHT = {
    "Clear": 1, "Cloudy": 1, "Overcast": 1,
    "Drizzle": 1, "Rain": 5, "Snow": 5,
}

# 日级使用 balanced 即可（少数类占比尚可 7%~15%）
DAILY_CLASS_WEIGHT = "balanced"

# ==================== 交叉验证 ====================
CV_SPLITS = 5
CV_SCORING = "f1_macro"

# ==================== 小时级子采样比例（386万行太慢） ====================
HOURLY_SUBSAMPLE_FRAC = 0.1  # 先用 10% 子样本粗搜参数
HOURLY_FULL_N_ESTIMATORS = 500  # 全量训练树数（不再限制）

# ==================== OOB 曲线配置 ====================
OOB_N_ESTIMATORS_RANGE = list(range(50, 801, 50))
