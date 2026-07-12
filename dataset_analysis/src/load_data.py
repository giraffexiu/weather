"""
数据加载与合并模块
===================
功能：
  1. 加载 dataset 目录下所有城市的 CSV 文件
  2. 分别合并 daily 和 hourly 数据为统一 DataFrame
  3. 输出数据集概览信息
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List
import warnings

warnings.filterwarnings("ignore")

# 项目路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

# 字段元数据定义
DAILY_COLUMNS = [
    "city", "country", "latitude", "longitude", "data_type", "time",
    "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
    "precipitation_sum", "rain_sum", "wind_speed_10m_max",
    "shortwave_radiation_sum"
]

HOURLY_COLUMNS = [
    "city", "country", "latitude", "longitude", "data_type", "time",
    "temperature_2m", "apparent_temperature", "relative_humidity_2m",
    "pressure_msl", "precipitation", "rain", "snowfall", "cloud_cover",
    "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
    "shortwave_radiation", "weather_code"
]

# 字段含义映射
FIELD_MEANING = {
    "city": "城市名",
    "country": "国家",
    "latitude": "纬度",
    "longitude": "经度",
    "data_type": "数据类型（daily/hourly）",
    "time": "时间戳",
    # Daily 特有
    "temperature_2m_max": "2米最高温度（°C）",
    "temperature_2m_min": "2米最低温度（°C）",
    "temperature_2m_mean": "2米平均温度（°C）",
    "precipitation_sum": "日总降水量（mm）",
    "rain_sum": "日总降雨量（mm）",
    "wind_speed_10m_max": "10米最大风速（km/h）",
    "shortwave_radiation_sum": "日总短波辐射（MJ/m²）",
    # Hourly 特有
    "temperature_2m": "2米温度（°C）",
    "apparent_temperature": "体感温度（°C）",
    "relative_humidity_2m": "2米相对湿度（%）",
    "pressure_msl": "海平面气压（hPa）",
    "precipitation": "降水量（mm）",
    "rain": "降雨量（mm）",
    "snowfall": "降雪量（cm）",
    "cloud_cover": "云量（%）",
    "wind_speed_10m": "10米风速（km/h）",
    "wind_direction_10m": "10米风向（°）",
    "wind_gusts_10m": "10米阵风（km/h）",
    "shortwave_radiation": "短波辐射（W/m²）",
    "weather_code": "天气代码（WMO标准）",
}

# 天气代码含义
WEATHER_CODE_MAP = {
    0: "晴天", 1: "大部晴朗", 2: "多云", 3: "阴天",
    45: "雾", 48: "雾凇",
    51: "小毛毛雨", 53: "中毛毛雨", 55: "大毛毛雨",
    61: "小雨", 63: "中雨", 65: "大雨",
    71: "小雪", 73: "中雪", 75: "大雪",
    80: "小阵雨", 81: "中阵雨", 82: "大阵雨",
    85: "小阵雪", 86: "大阵雪",
    95: "雷暴", 96: "小冰雹雷暴", 99: "大冰雹雷暴",
}

# 数值特征列表（用于训练）
NUMERIC_FEATURES_DAILY = [
    "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
    "precipitation_sum", "rain_sum", "wind_speed_10m_max",
    "shortwave_radiation_sum"
]

NUMERIC_FEATURES_HOURLY = [
    "temperature_2m", "apparent_temperature", "relative_humidity_2m",
    "pressure_msl", "precipitation", "rain", "snowfall", "cloud_cover",
    "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
    "shortwave_radiation"
]

# 分类特征
CATEGORICAL_FEATURES_DAILY = ["city", "country"]
CATEGORICAL_FEATURES_HOURLY = ["city", "country", "weather_code"]

# ID/元数据列（不参与训练）
META_COLUMNS = ["latitude", "longitude", "data_type"]


def load_single_file(filepath: Path) -> pd.DataFrame:
    """加载单个CSV文件"""
    df = pd.read_csv(filepath)
    return df


def load_all_files(data_type: str) -> pd.DataFrame:
    """
    加载并合并所有城市的 daily 或 hourly 数据

    参数:
        data_type: "daily" 或 "hourly"

    返回:
        合并后的 DataFrame，按 city 和 time 排序
    """
    pattern = f"weather_{data_type}_*.csv"
    files = sorted(DATASET_DIR.glob(pattern))

    if not files:
        raise FileNotFoundError(f"未找到匹配模式 {pattern} 的文件，请检查 {DATASET_DIR}")

    print(f"\n{'='*60}")
    print(f"加载 {data_type.upper()} 数据")
    print(f"{'='*60}")
    print(f"找到 {len(files)} 个文件")

    dfs = []
    for f in files:
        city_name = f.stem.replace(f"weather_{data_type}_", "").rsplit("_", 2)[0]
        df = load_single_file(f)
        dfs.append(df)
        print(f"  ✓ {city_name}: {len(df):,} 条记录")

    merged_df = pd.concat(dfs, ignore_index=True)

    # 转换时间列
    merged_df["time"] = pd.to_datetime(merged_df["time"])

    # 按城市和时间排序
    merged_df = merged_df.sort_values(["city", "time"]).reset_index(drop=True)

    print(f"\n合并后总计: {len(merged_df):,} 条记录, {merged_df['city'].nunique()} 个城市")
    print(f"时间范围: {merged_df['time'].min()} ~ {merged_df['time'].max()}")

    return merged_df


def get_dataset_summary(df: pd.DataFrame, name: str) -> Dict:
    """
    生成数据集摘要信息

    返回包含数据维度、字段信息、数据类型分布的字典
    """
    summary = {
        "name": name,
        "shape": df.shape,
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
        "time_range": (df["time"].min(), df["time"].max()),
        "cities": df["city"].unique().tolist(),
        "n_cities": df["city"].nunique(),
        "n_countries": df["country"].nunique(),
    }

    # 分类字段
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    time_cols = ["time"]
    meta_cols = [c for c in META_COLUMNS if c in df.columns]

    summary["numerical_columns"] = num_cols
    summary["categorical_columns"] = cat_cols
    summary["time_columns"] = time_cols
    summary["meta_columns"] = meta_cols

    return summary


def load_and_merge_all() -> Tuple[pd.DataFrame, pd.DataFrame, Dict, Dict]:
    """
    加载所有数据的主入口函数

    返回:
        df_daily: 合并后的日数据
        df_hourly: 合并后的小时数据
        summary_daily: daily 数据摘要
        summary_hourly: hourly 数据摘要
    """
    df_hourly = load_all_files("hourly")
    df_daily = load_all_files("daily")

    summary_hourly = get_dataset_summary(df_hourly, "hourly")
    summary_daily = get_dataset_summary(df_daily, "daily")

    return df_daily, df_hourly, summary_daily, summary_hourly


def build_data_dictionary(df: pd.DataFrame, data_type: str) -> pd.DataFrame:
    """
    构建数据字典

    返回 DataFrame 格式的数据字典：
    字段名 | 数据类型 | 含义 | 是否用于训练 | 处理方式
    """
    rows = []
    target_col = "temperature_2m" if data_type == "hourly" else "temperature_2m_mean"
    features = NUMERIC_FEATURES_HOURLY if data_type == "hourly" else NUMERIC_FEATURES_DAILY
    cat_features = CATEGORICAL_FEATURES_HOURLY if data_type == "hourly" else CATEGORICAL_FEATURES_DAILY

    for col in df.columns:
        meaning = FIELD_MEANING.get(col, "未知")

        if col == target_col:
            usage = "标签（预测目标）"
            process = "标准化（与特征使用相同scaler）"
        elif col in META_COLUMNS:
            usage = "否（元数据）"
            process = "删除"
        elif col == "time":
            usage = "是（时间索引）"
            process = "提取年/月/日/星期/季节等时间特征；用于序列排序"
        elif col in features:
            usage = "是（数值特征）"
            process = "缺失值填充 → 异常值处理 → 标准化（StandardScaler）"
        elif col in cat_features:
            if col == "weather_code":
                usage = "是（类别特征）"
                process = "Label Encoding 或保留原始编码（WMO标准）"
            else:
                usage = "是（类别特征）"
                process = "Label Encoding → 作为分组键（不直接输入模型）"
        elif col == "data_type":
            usage = "否（元数据）"
            process = "删除"
        else:
            usage = "待定"
            process = "根据分析决定"

        dtype = str(df[col].dtype)
        rows.append({
            "字段名": col,
            "数据类型": dtype,
            "含义": meaning,
            "是否用于训练": usage,
            "处理方式": process,
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    # 自测：加载数据并打印摘要
    df_daily, df_hourly, summary_daily, summary_hourly = load_and_merge_all()

    print("\n" + "=" * 60)
    print("DAILY 数据摘要")
    print("=" * 60)
    for k, v in summary_daily.items():
        if k not in ["dtypes"]:
            print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("HOURLY 数据摘要")
    print("=" * 60)
    for k, v in summary_hourly.items():
        if k not in ["dtypes"]:
            print(f"  {k}: {v}")

    print("\n数据字典 - Daily:")
    print(build_data_dictionary(df_daily, "daily").to_string(index=False))

    print("\n数据字典 - Hourly:")
    print(build_data_dictionary(df_hourly, "hourly").to_string(index=False))
