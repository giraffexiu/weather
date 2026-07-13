"""
数据加载与目标构造模块 (Data Loader) — 仅小时级
多目标回归：对 5 个目标列各自 shift(-1)，目标=下一小时的气象变量值

数据重切分：上游 CSV 按年切分(train=2015~2023, test=2024)，
本层将 2023 年数据从训练集移入测试集（train=2015~2022, test=2023~2024）
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Tuple, List

import config


def _add_lag_features(df: pd.DataFrame, cols: List[str], periods: List[int],
                      group_col: str = "city_id") -> pd.DataFrame:
    """添加滞后特征（按城市分组）"""
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        for lag in periods:
            df[f"{col}_lag_{lag}"] = df.groupby(group_col)[col].shift(lag)
    return df


def _add_pressure_change(df: pd.DataFrame, col: str, shift_period: int = 3,
                         group_col: str = "city_id") -> pd.DataFrame:
    """气压变化率（前 N 小时 - 当前值）"""
    df = df.copy()
    if col in df.columns:
        df[f"{col}_change_{shift_period}"] = (
            df.groupby(group_col)[col].shift(shift_period) - df[col])
    return df


def _encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """将残留的字符串类别列编码为整数（season, day_period）"""
    df = df.copy()
    mappings = {
        "season": {"winter": 0, "spring": 1, "summer": 2, "autumn": 3},
        "day_period": {"night": 0, "morning": 1, "forenoon": 2,
                        "afternoon": 3, "evening": 4},
    }
    for col, mp in mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mp).astype(int)
    return df


def _add_multi_scale_rolling(df: pd.DataFrame, cols: List[str], windows: List[int],
                             group_col: str = "city_id") -> pd.DataFrame:
    """多尺度滑动窗口统计（mean + std，多个窗口大小）"""
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        grp = df.groupby(group_col)[col]
        for w in windows:
            df[f"{col}_roll{w}_mean"] = grp.transform(
                lambda x: x.rolling(w, min_periods=1).mean())
            df[f"{col}_roll{w}_std"] = grp.transform(
                lambda x: x.rolling(w, min_periods=1).std())
    return df


def _add_cloud_features(df: pd.DataFrame, cloud_col: str,
                        group_col: str = "city_id") -> pd.DataFrame:
    """云量变化率 + 云量趋势"""
    df = df.copy()
    if cloud_col not in df.columns:
        return df
    grp = df.groupby(group_col)[cloud_col]
    df[f"{cloud_col}_variability"] = grp.transform(
        lambda x: x.rolling(24, min_periods=1).std())
    df[f"{cloud_col}_trend"] = grp.transform(lambda x: x.diff(3))
    df[f"{cloud_col}_direction"] = grp.transform(
        lambda x: np.sign(x.diff(3))).fillna(0).astype(int)
    return df


def _add_interaction_features_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """小时级特征交互"""
    df = df.copy()
    if "temperature_2m" in df.columns and "relative_humidity_2m" in df.columns:
        df["temp_humidity"] = df["temperature_2m"] * df["relative_humidity_2m"]
    if "pressure_msl" in df.columns and "wind_direction_10m" in df.columns:
        df["pressure_wind"] = df["pressure_msl"] * df["wind_direction_10m"]
    if "cloud_cover" in df.columns and "shortwave_radiation" in df.columns:
        df["cloud_rad_ratio"] = df["cloud_cover"] / (df["shortwave_radiation"].abs() + 0.1)
    return df


def _get_feature_cols(df: pd.DataFrame, target_names: List[str]) -> List[str]:
    """获取特征列（排除标识/时间/目标列）"""
    exclude = set(config.EXCLUDE_COLS_BASE)
    exclude.update(target_names)
    feature_cols = [c for c in df.columns if c not in exclude]
    return feature_cols


def _resplit_by_date(train_df: pd.DataFrame, test_df: pd.DataFrame,
                     cutoff: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    按日期重切分：将 train_df 中 time >= cutoff 的行移入 test_df

    Args:
        train_df: 上游训练集 CSV（2015~2023）
        test_df: 上游测试集 CSV（2024）
        cutoff: 切分日期，train 保留 time < cutoff，test 收集 time >= cutoff

    Returns:
        (new_train, new_test)
    """
    cutoff_ts = pd.Timestamp(cutoff)

    train_df = train_df.copy()
    train_df["time"] = pd.to_datetime(train_df["time"])

    mask_move = train_df["time"] >= cutoff_ts
    moved = train_df[mask_move].copy()
    new_train = train_df[~mask_move].copy()

    test_df = test_df.copy()
    test_df["time"] = pd.to_datetime(test_df["time"])
    new_test = pd.concat([moved, test_df], ignore_index=True)

    return new_train, new_test


def load_hourly(subset_frac: float = 1.0, verbose: bool = True) -> Tuple:
    """
    加载小时级数据并构造多目标回归标签

    数据流：
      1. 读取上游 train_features.csv(2015~2023) + test_features.csv(2024)
      2. 按 TRAIN_CUTOFF 重切分：train=2015~2022, test=2023~2024
      3. 构造标签：对 5 个目标列各自 shift(-1) = 下一小时值
      4. 对 precipitation 目标做 log1p 变换（长尾→近似正态）
      5. 时序特征增强：lag(1~48h) + rolling(3~48h) + 气压变化 + 云量趋势 + 交互

    Args:
        subset_frac: 子采样比例（仅对训练集，加速 GridSearch）
        verbose: 是否打印详细信息

    Returns:
        (X_train, y_train, X_test, y_test, feature_names)
    """
    if verbose:
        print(f"\n{'='*60}")
        print("加载小时级数据（多目标回归）")
        print(f"{'='*60}")

    raw_train = pd.read_csv(config.HOURLY_TRAIN_PATH)
    raw_test = pd.read_csv(config.HOURLY_TEST_PATH)

    if verbose:
        print(f"上游训练集: {len(raw_train):,} 行")
        print(f"上游测试集: {len(raw_test):,} 行")

    # 重切分：2023 年数据从训练集移入测试集
    train_df, test_df = _resplit_by_date(raw_train, raw_test, config.TRAIN_CUTOFF)

    if verbose:
        print(f"\n重切分后:")
        print(f"  训练集: {len(train_df):,} 行 ({train_df['time'].min()} ~ {train_df['time'].max()})")
        print(f"  测试集: {len(test_df):,} 行 ({test_df['time'].min()} ~ {test_df['time'].max()})")

    target_cols = config.HOURLY_TARGET_COLUMNS
    missing = [c for c in target_cols if c not in train_df.columns]
    if missing:
        raise ValueError(f"训练集缺少目标列: {missing}")

    # 排序 + 编码
    for df in (train_df, test_df):
        df.sort_values(["city_id", "time"], inplace=True)
        df.reset_index(drop=True, inplace=True)
    train_df = _encode_categorical(train_df)
    test_df = _encode_categorical(test_df)

    if subset_frac < 1.0:
        train_df = _subsample_by_city(train_df, subset_frac)
        if verbose:
            print(f"子采样 {subset_frac:.0%} 后训练集: {len(train_df):,} 行")

    # 构造标签：下一小时的 5 个气象变量值
    for df in (train_df, test_df):
        for col in target_cols:
            df[f"target_{col}"] = df.groupby("city_id")[col].shift(-1)
        df.dropna(subset=[f"target_{col}" for col in target_cols], inplace=True)

    # precipitation 目标 log1p 变换
    precip_target = "target_precipitation"
    if config.USE_LOG_TRANSFORM_PRECIP and precip_target in train_df.columns:
        for df in (train_df, test_df):
            df[precip_target] = np.log1p(df[precip_target] - df[precip_target].min())
        if verbose:
            print(f"\nprecipitation 目标已 log1p 变换")

    # 时序特征增强
    for i, df in enumerate([train_df, test_df]):
        df = _add_lag_features(df, config.LAG_COLS_HOURLY, config.LAG_PERIODS_HOURLY)
        df = _add_multi_scale_rolling(df, ["temperature_2m", "pressure_msl", "cloud_cover"],
                                      config.ROLLING_WINDOWS_HOURLY)
        for period in config.HOURLY_PRESSURE_CHANGE_PERIODS:
            df = _add_pressure_change(df, "pressure_msl", shift_period=period)
        df = _add_cloud_features(df, "cloud_cover")
        df = _add_interaction_features_hourly(df)
        if i == 0:
            train_df = df
        else:
            test_df = df

    # 删除因最大滞后产生的 NaN 行
    max_lag = max(config.LAG_PERIODS_HOURLY)
    max_lag_col = f"{config.LAG_COLS_HOURLY[0]}_lag_{max_lag}"
    if max_lag_col in train_df.columns:
        train_df.dropna(subset=[max_lag_col], inplace=True)
        test_df.dropna(subset=[max_lag_col], inplace=True)
    for df in (train_df, test_df):
        roll_nan_cols = [c for c in df.columns
                         if ("_roll" in c and c.endswith("_std"))
                         or c.endswith("_variability") or c.endswith("_trend")]
        df[roll_nan_cols] = df[roll_nan_cols].fillna(0)

    target_names = [f"target_{col}" for col in target_cols]
    feature_cols = _get_feature_cols(train_df, target_names)
    if verbose:
        print(f"\n特征数: {len(feature_cols)}")
        print(f"目标变量: {target_cols}")
        print(f"目标统计(train):\n{train_df[target_names].describe().round(2).to_string()}")

    return (train_df[feature_cols].values, train_df[target_names].values,
            test_df[feature_cols].values, test_df[target_names].values,
            feature_cols)


def _subsample_by_city(df: pd.DataFrame, frac: float) -> pd.DataFrame:
    """按城市分组子采样（系统抽样：每隔 k 行取一行，保持时序全覆盖）"""
    parts = []
    for city_id, group in df.groupby("city_id"):
        step = max(int(1 / frac), 1)
        sampled = group.iloc[::step]
        if len(sampled) < 1000:
            sampled = group.iloc[:1000]
        parts.append(sampled)
    return pd.concat(parts, ignore_index=True)


def save_feature_config(feature_names: List[str], granularity: str):
    """保存特征列配置（供 predict.py 使用）"""
    config_data = {}
    if config.FEATURE_CONFIG_PATH.exists():
        with open(config.FEATURE_CONFIG_PATH, "r", encoding="utf-8") as f:
            config_data = json.load(f)
    config_data[f"{granularity}_features"] = feature_names
    with open(config.FEATURE_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    print("=== 测试小时级数据加载（子采样 5%） ===")
    X_train, y_train, X_test, y_test, feat = load_hourly(subset_frac=0.05)
    print(f"\nX_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
    print(f"特征列示例: {feat[:5]}...")
