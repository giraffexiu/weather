"""
特征工程模块
============
为 LSTM 天气预测模型构建特征

功能：
  1. 时间特征提取（年、月、日、星期、季节、周期编码）
  2. 数值特征标准化
  3. 类别特征编码
  4. 构建滑动窗口序列（LSTM输入格式）
  5. 特征选择（相关性过滤）
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, List, Dict, Optional
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# 默认时间序列参数
DEFAULT_DAILY_CONFIG = {
    "lookback": 30,  # 过去30天
    "forecast": 7,   # 预测未来7天
    "target": "temperature_2m_mean",
}

DEFAULT_HOURLY_CONFIG = {
    "lookback": 168,  # 过去7天（168小时）
    "forecast": 24,   # 预测未来24小时
    "target": "temperature_2m",
}


def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    从 time 列提取时间特征

    提取：
    - year, month, day, hour
    - dayofweek（星期几，0=周一）
    - is_weekend（是否周末）
    - quarter（季度）
    - season（季节：春=0, 夏=1, 秋=2, 冬=3）
    - day_of_year（一年中的第几天）
    - month_sin, month_cos（月份周期编码）
    - day_sin, day_cos（日期周期编码）
    - hour_sin, hour_cos（小时周期编码，仅hourly数据）

    周期性编码原理：
    将循环特征（如月份1-12）映射到单位圆上，
    使12月和1月在特征空间中相邻
    """
    df = df.copy()

    # 确保 time 是 datetime 类型
    if df["time"].dtype != "datetime64[ns]":
        df["time"] = pd.to_datetime(df["time"])

    # 基础时间特征
    df["year"] = df["time"].dt.year.astype(np.int32)
    df["month"] = df["time"].dt.month.astype(np.int32)
    df["day"] = df["time"].dt.day.astype(np.int32)
    df["dayofweek"] = df["time"].dt.dayofweek.astype(np.int32)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(np.int32)
    df["quarter"] = df["time"].dt.quarter.astype(np.int32)
    df["day_of_year"] = df["time"].dt.dayofyear.astype(np.int32)

    # 季节编码
    df["season"] = df["month"].apply(lambda m: (m % 12 + 3) // 3 - 1).clip(0, 3).astype(np.int32)

    # 小时特征（仅hourly数据）
    if "hour" not in df.columns:
        try:
            df["hour"] = df["time"].dt.hour.astype(np.int32)
        except Exception:
            df["hour"] = 0

    # 周期性编码
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["day_sin"] = np.sin(2 * np.pi * df["day"] / 31)
    df["day_cos"] = np.cos(2 * np.pi * df["day"] / 31)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dayofweek_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dayofweek_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

    # 年份归一化（相对2015年）
    df["year_normalized"] = (df["year"] - 2015).astype(np.float32)

    return df


def encode_categorical(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    类别特征编码

    - city: Label Encoding（0-8）
    - country: Label Encoding
    - weather_code: 保留原始WMO编码（已有数值含义）

    返回:
        (encoded_df, encoders_dict)
    """
    df = df.copy()
    encoders = {}

    # city 编码
    if "city" in df.columns:
        le_city = LabelEncoder()
        df["city_encoded"] = le_city.fit_transform(df["city"]).astype(np.int32)
        encoders["city"] = le_city

    # country 编码
    if "country" in df.columns:
        le_country = LabelEncoder()
        df["country_encoded"] = le_country.fit_transform(df["country"]).astype(np.int32)
        encoders["country"] = le_country

    # weather_code 保持原值（已是数值），填-1表示缺失
    if "weather_code" in df.columns:
        df["weather_code"] = df["weather_code"].fillna(-1).astype(np.int32)

    return df, encoders


def build_feature_list(data_type: str, include_time_features: bool = True) -> List[str]:
    """
    构建最终特征列表

    参数:
        data_type: "daily" 或 "hourly"
        include_time_features: 是否包含时间特征

    返回:
        特征列名列表
    """
    from load_data import NUMERIC_FEATURES_DAILY, NUMERIC_FEATURES_HOURLY

    if data_type == "daily":
        features = list(NUMERIC_FEATURES_DAILY)
    else:
        features = list(NUMERIC_FEATURES_HOURLY)

    # 添加编码后的类别特征
    features.append("city_encoded")
    if data_type == "hourly":
        features.append("weather_code")

    if include_time_features:
        time_feats = [
            "month_sin", "month_cos", "day_sin", "day_cos",
            "hour_sin", "hour_cos", "dayofweek_sin", "dayofweek_cos",
            "year_normalized", "is_weekend",
        ]
        features.extend(time_feats)

    return features


def scale_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    scaler: Optional[StandardScaler] = None,
    fit: bool = True,
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    数值特征标准化（Z-score）

    参数:
        fit: True 则拟合并转换，False 则仅转换（用于验证/测试集）

    返回:
        (scaled_df, scaler)
    """
    df = df.copy()

    if scaler is None:
        scaler = StandardScaler()

    available_cols = [c for c in feature_cols if c in df.columns]

    if fit:
        df[available_cols] = scaler.fit_transform(df[available_cols])
    else:
        df[available_cols] = scaler.transform(df[available_cols])

    return df, scaler


def build_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    lookback: int,
    forecast: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    构建 LSTM 滑动窗口序列

    对于每个时间步，用过去 lookback 步的特征预测未来 forecast 步的目标值。

    注意：此函数假设数据已按城市分组并按时间排序。
    跨城市边界处不会混合数据（每个城市的序列独立）。

    参数:
        df: 已排序的DataFrame（按 city, time）
        feature_cols: 特征列名列表
        target_col: 目标列名
        lookback: 回溯窗口大小（输入序列长度）
        forecast: 预测窗口大小（输出序列长度）

    返回:
        X: (n_samples, lookback, n_features)
        y: (n_samples, forecast)
    """
    X_list, y_list = [], []

    # 确保列存在
    available_features = [c for c in feature_cols if c in df.columns]
    if target_col not in df.columns:
        raise ValueError(f"目标列 {target_col} 不在DataFrame中")

    # 按城市分组构建序列，避免跨城市数据混合
    if "city" in df.columns:
        groups = df.groupby("city")
    else:
        groups = [("all", df)]

    for city_name, group in groups:
        values = group[available_features].values.astype(np.float32)
        targets = group[target_col].values.astype(np.float32)

        n = len(group)
        for i in range(n - lookback - forecast + 1):
            X_list.append(values[i : i + lookback])
            y_list.append(targets[i + lookback : i + lookback + forecast])

    if not X_list:
        raise ValueError(
            f"无法构建任何序列。请检查 lookback={lookback}, forecast={forecast} "
            f"是否小于数据长度。"
        )

    X = np.stack(X_list, axis=0)
    y = np.stack(y_list, axis=0)

    return X, y


def compute_correlation_matrix(
    df: pd.DataFrame,
    columns: List[str],
) -> pd.DataFrame:
    """
    计算特征相关性矩阵（Pearson）

    返回:
        相关性矩阵 DataFrame
    """
    available = [c for c in columns if c in df.columns]
    return df[available].corr()


def filter_high_correlation(
    corr_matrix: pd.DataFrame,
    threshold: float = 0.95,
) -> List[str]:
    """
    过滤高相关特征（>threshold 则保留一个）

    返回:
        需要保留的特征列表
    """
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    to_drop = [
        column
        for column in upper.columns
        if any(upper[column].abs() > threshold)
    ]

    keep = [c for c in corr_matrix.columns if c not in to_drop]
    return keep


def feature_engineering_pipeline(
    df: pd.DataFrame,
    data_type: str,
    target_col: str,
    lookback: int,
    forecast: int,
    scaler: Optional[StandardScaler] = None,
    fit: bool = True,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, StandardScaler, List[str]]:
    """
    特征工程完整流水线

    1. 提取时间特征
    2. 编码类别特征
    3. 标准化数值特征
    4. 构建 LSTM 序列

    返回:
        X, y, processed_df, scaler, feature_list
    """
    print(f"\n{'='*60}")
    print(f"特征工程流水线 - {data_type.upper()}")
    print(f"{'='*60}")

    # 1. 时间特征
    print("1. 提取时间特征...")
    df = extract_time_features(df)

    # 2. 类别编码
    print("2. 类别特征编码...")
    df, encoders = encode_categorical(df)

    # 3. 构建特征列表
    feature_cols = build_feature_list(data_type, include_time_features=True)
    feature_cols = [c for c in feature_cols if c in df.columns]
    print(f"   特征数量: {len(feature_cols)}")

    # 4. 标准化
    print("3. 数值特征标准化...")
    num_features = [c for c in feature_cols if c in df.columns]
    df, scaler = scale_features(df, num_features, scaler, fit=fit)

    # 5. 构建序列
    print(f"4. 构建LSTM序列 (lookback={lookback}, forecast={forecast})...")
    X, y = build_sequences(df, feature_cols, target_col, lookback, forecast)
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")

    return X, y, df, scaler, feature_cols


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    from load_data import load_and_merge_all

    _, df_hourly, _, _ = load_and_merge_all()

    # 测试特征工程
    X, y, df_proc, scaler, feats = feature_engineering_pipeline(
        df_hourly, "hourly",
        target_col="temperature_2m",
        lookback=168, forecast=24,
    )
    print(f"\n序列构建完成: X={X.shape}, y={y.shape}")
