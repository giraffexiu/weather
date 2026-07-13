"""
随机森林模型配置 (Model 2: Random Forest)
小时级多目标回归：5 变量连续值预测
预测目标与 model_3 (Deep Learning) 统一：
  - temperature_2m (温度)
  - precipitation (降水量)
  - wind_speed_10m (风速)
  - apparent_temperature (体感温度)
  - relative_humidity_2m (相对湿度)
"""
import os
from pathlib import Path

# ==================== 路径配置 ====================
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "data"

# 特征工程产出路径（上游 pipeline 按时间切分：train=2015~2023, test=2024）
HOURLY_TRAIN_PATH = DATA_ROOT / "data_engineer" / "hourly_data" / "processed_data" / "train_features.csv"
HOURLY_TEST_PATH = DATA_ROOT / "data_engineer" / "hourly_data" / "processed_data" / "test_features.csv"

# 小时级清洗数据（含 weather_code）
HOURLY_CLEANED_PATH = DATA_ROOT / "data_clean" / "cleaned_data" / "weather_hourly_cleaned.csv"

# 模型保存路径
MODEL_DIR = Path(__file__).parent / "saved_models"
MODEL_DIR.mkdir(exist_ok=True)
HOURLY_MODEL_PATH = MODEL_DIR / "rf_hourly.pkl"
FEATURE_CONFIG_PATH = MODEL_DIR / "feature_config.json"

# 输出路径
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
HOURLY_OUTPUT_DIR = OUTPUT_DIR / "hourly"
HOURLY_OUTPUT_DIR.mkdir(exist_ok=True)

# ==================== 数据重切分 ====================
# 上游 CSV: train=2015~2023, test=2024
# 本层将 2023 年数据从训练集移入测试集，扩大测试集覆盖年份
TRAIN_CUTOFF = "2023-01-01"      # 训练集: time < 此日期 (2015~2022)
TEST_START = "2023-01-01"        # 测试集: time >= 此日期 (2023~2024)

# ==================== 多目标回归目标列 ====================
# 小时级：直接从特征数据中取这 5 列作为回归目标
HOURLY_TARGET_COLUMNS = [
    "temperature_2m",
    "precipitation",
    "wind_speed_10m",
    "apparent_temperature",
    "relative_humidity_2m",
]

# 降水 log1p 变换索引（precipitation 是第 1 个目标，0-indexed）
# 降水是长尾分布（大量0值+极端值），log1p 后更接近正态，RF 回归更稳定
PRECIPITATION_TARGET_IDX = 1
USE_LOG_TRANSFORM_PRECIP = True

# ==================== 随机种子 ====================
RANDOM_SEED = 42

# ==================== 排除列（不作为特征使用） ====================
# 这些列是标识/时间/目标列，不能作为特征
# season 和 day_period 是字符串列，由 data_loader._encode_categorical 编码为整数后保留为特征
EXCLUDE_COLS_BASE = [
    "city", "country", "time", "weather_code", "weather_code_id",
    "target_temperature_2m", "target_precipitation", "target_wind_speed_10m",
    "target_apparent_temperature", "target_relative_humidity_2m",
]

# 小时级数据独有的排除列
HOURLY_EXCLUDE_EXTRA = []

# ==================== 时序特征增强（滞后/滑动窗口） ====================
# 滞后阶数：覆盖日周期(24h) + 48h（两天周期）
LAG_PERIODS_HOURLY = [1, 2, 3, 6, 12, 24, 48]
# 需要构造滞后特征的列
LAG_COLS_HOURLY = ["temperature_2m", "pressure_msl", "relative_humidity_2m",
                   "precipitation", "cloud_cover", "wind_speed_10m"]

# 多尺度滑动窗口
ROLLING_WINDOWS_HOURLY = [3, 7, 12, 24, 48]   # 短期 + 半日 + 全天 + 两天
# 滚动统计类型
ROLLING_STATS = ["mean", "std"]

# 气压变化率周期
HOURLY_PRESSURE_CHANGE_PERIODS = [1, 3, 6]

# Cloudy 专项区分特征
CLOUD_FEATURE_COLS_HOURLY = ["cloud_cover"]  # 用于构造云量变化率/趋势

# 特征交互列
INTERACTION_FEATURES_HOURLY = [
    "temp_humidity",        # 温度×湿度
    "pressure_wind",        # 气压×风向
    "cloud_rad_ratio",      # 云量/辐射比
]

# ==================== RandomForest 超参数网格（精简版，加速训练） ====================
PARAM_GRID = {
    "n_estimators": [200, 300],
    "max_depth": [20, 30, None],
    "min_samples_leaf": [2, 4],
    "max_features": ["log2", 0.3],
}

# 固定参数（回归专用：无 class_weight）
RF_FIXED_PARAMS = {
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "oob_score": True,
    "bootstrap": True,
}

# ==================== 交叉验证 ====================
CV_SPLITS = 5
CV_SCORING = "neg_mean_squared_error"

# ==================== 小时级子采样比例（386万行太慢） ====================
HOURLY_SUBSAMPLE_FRAC = 0.1  # 先用 10% 子样本粗搜参数

# ==================== OOB 曲线配置 ====================
OOB_N_ESTIMATORS_RANGE = list(range(50, 801, 50))
