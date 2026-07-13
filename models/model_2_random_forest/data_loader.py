"""
数据加载与标签构造模块 (Data Loader)
双粒度天气分类：日级 + 小时级

小时级标签：直接从 weather_code 合并 6 类，目标=下一小时天气
日级标签：从小时级清洗数据按"当天最严重天气"聚合，目标=明天天气
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Tuple, List

import config


def weather_code_to_category(code) -> str:
    """WMO weather_code -> 6 类天气"""
    if pd.isna(code):
        return None
    code = int(code)
    return config.WMO_TO_CATEGORY.get(code, "Clear")


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


def _add_rolling_features(df: pd.DataFrame, cols: List[str], window: int,
                          group_col: str = "city_id") -> pd.DataFrame:
    """添加滑动窗口统计特征（按城市分组）"""
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        grp = df.groupby(group_col)[col]
        df[f"{col}_roll{window}_mean"] = grp.transform(
            lambda x: x.rolling(window, min_periods=1).mean())
        df[f"{col}_roll{window}_std"] = grp.transform(
            lambda x: x.rolling(window, min_periods=1).std())
    return df


def _add_pressure_change(df: pd.DataFrame, col: str, shift_period: int = 3,
                         group_col: str = "city_id") -> pd.DataFrame:
    """气压变化率（前 N 小时/天 - 当前值）"""
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
    """Cloudy 专项区分特征：云量变化率 + 云量趋势"""
    df = df.copy()
    if cloud_col not in df.columns:
        return df
    grp = df.groupby(group_col)[cloud_col]
    # 云量变化率（24h 标准差，高=Cloudy 波动，低=Clear/Overcast 稳定）
    df[f"{cloud_col}_variability"] = grp.transform(
        lambda x: x.rolling(24, min_periods=1).std())
    # 云量趋势（3 步差分，正=变阴，负=变晴）
    df[f"{cloud_col}_trend"] = grp.transform(lambda x: x.diff(3))
    # 云量变化方向（1=增加，-1=减少，0=稳定）
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


def _add_interaction_features_daily(df: pd.DataFrame) -> pd.DataFrame:
    """日级特征交互"""
    df = df.copy()
    if "temperature_2m_mean" in df.columns and "humidity_mean" in df.columns:
        df["temp_humidity_daily"] = df["temperature_2m_mean"] * df["humidity_mean"]
    if "pressure_mean" in df.columns and "wind_direction_mean" in df.columns:
        df["pressure_wind_daily"] = df["pressure_mean"] * df["wind_direction_mean"]
    if "cloud_cover_mean" in df.columns and "shortwave_radiation_sum" in df.columns:
        df["cloud_rad_ratio_daily"] = df["cloud_cover_mean"] / (
            df["shortwave_radiation_sum"].abs() + 0.1)
    return df


def _get_feature_cols(df: pd.DataFrame, granularity: str) -> List[str]:
    """获取特征列（排除标识/时间/标签列）"""
    exclude = set(config.EXCLUDE_COLS_BASE)
    if granularity == "daily":
        exclude.update(config.DAILY_EXCLUDE_EXTRA)
    else:
        exclude.update(config.HOURLY_EXCLUDE_EXTRA)
    feature_cols = [c for c in df.columns if c not in exclude]
    return feature_cols


def load_hourly(subset_frac: float = 1.0, verbose: bool = True) -> Tuple:
    """
    加载小时级数据并构造 6 分类标签

    目标：用当前小时特征预测下一小时的天气类别
    标签来源：weather_code -> WMO_TO_CATEGORY -> shift(-1)

    Args:
        subset_frac: 子采样比例（<1 时随机抽取城市内的连续块）
        verbose: 是否打印详细信息

    Returns:
        (X_train, y_train, X_test, y_test, feature_names)
    """
    if verbose:
        print(f"\n{'='*60}")
        print("加载小时级数据")
        print(f"{'='*60}")

    train_df = pd.read_csv(config.HOURLY_TRAIN_PATH)
    test_df = pd.read_csv(config.HOURLY_TEST_PATH)

    if verbose:
        print(f"训练集: {len(train_df):,} 行 × {len(train_df.columns)} 列")
        print(f"测试集: {len(test_df):,} 行 × {len(test_df.columns)} 列")

    # 按城市+时间排序 + 字符串列编码
    for df in (train_df, test_df):
        df.sort_values(["city_id", "time"], inplace=True)
        df.reset_index(drop=True, inplace=True)
    train_df = _encode_categorical(train_df)
    test_df = _encode_categorical(test_df)

    # 子采样（仅训练集，加速 GridSearch）
    if subset_frac < 1.0:
        train_df = _subsample_by_city(train_df, subset_frac)
        if verbose:
            print(f"子采样 {subset_frac:.0%} 后训练集: {len(train_df):,} 行")

    # 构造标签：下一小时的天气类别
    for df in (train_df, test_df):
        df["target"] = df.groupby("city_id")["weather_code"].shift(-1)
        df["target"] = df["target"].map(config.WMO_TO_CATEGORY)
        # 删除每个城市最后一行（无下一小时数据）
        df.dropna(subset=["target"], inplace=True)

    # 时序特征增强：多步滞后 + 多尺度滑动窗口 + 气压变化率 + Cloudy 专项 + 交互特征
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

    # 删除因最大滞后(24)产生的 NaN 行
    max_lag_col = f"{config.LAG_COLS_HOURLY[0]}_lag_{config.LAG_PERIODS_HOURLY[-1]}"
    if max_lag_col in train_df.columns:
        train_df.dropna(subset=[max_lag_col], inplace=True)
        test_df.dropna(subset=[max_lag_col], inplace=True)
    # 滚动 std / cloud variability 的 NaN 填 0
    for df in (train_df, test_df):
        roll_nan_cols = [c for c in df.columns
                         if ("_roll" in c and c.endswith("_std"))
                         or c.endswith("_variability") or c.endswith("_trend")]
        df[roll_nan_cols] = df[roll_nan_cols].fillna(0)

    feature_cols = _get_feature_cols(train_df, "hourly")
    if verbose:
        print(f"特征数: {len(feature_cols)}")
        print(f"标签分布(train):\n{train_df['target'].value_counts(normalize=True).round(4).to_string()}")

    return (train_df[feature_cols].values, train_df["target"].values,
            test_df[feature_cols].values, test_df["target"].values,
            feature_cols)


def load_daily(verbose: bool = True) -> Tuple:
    """
    加载日级数据并构造聚合标签

    目标：用今天特征预测明天的天气类别
    标签来源：小时级清洗数据按 city+date 聚合"当天最严重天气"，
              shift(-1) 得到明天的天气

    Returns:
        (X_train, y_train, X_test, y_test, feature_names)
    """
    if verbose:
        print(f"\n{'='*60}")
        print("加载日级数据")
        print(f"{'='*60}")

    train_df = pd.read_csv(config.DAILY_TRAIN_PATH)
    test_df = pd.read_csv(config.DAILY_TEST_PATH)

    if verbose:
        print(f"训练集: {len(train_df):,} 行 × {len(train_df.columns)} 列")
        print(f"测试集: {len(test_df):,} 行 × {len(test_df.columns)} 列")

    # 从小时级清洗数据聚合日级标签
    daily_label = _aggregate_daily_labels(verbose=verbose)

    # 从小时级清洗数据聚合日级补充特征（气压/湿度/云量/风向/当天天气）
    daily_extra = _aggregate_hourly_to_daily_features(verbose=verbose)

    # 合并标签和补充特征到日级特征
    train_df["date"] = pd.to_datetime(train_df["time"]).dt.date
    test_df["date"] = pd.to_datetime(test_df["time"]).dt.date
    for i, df in enumerate([train_df, test_df]):
        df["target"] = df.apply(
            lambda row: daily_label.get((row["city"], row["date"]), None),
            axis=1)
        merged = df.merge(daily_extra, on=["city", "date"], how="left")
        if i == 0:
            train_df = merged
        else:
            test_df = merged

    # 字符串列编码（season 等）
    train_df = _encode_categorical(train_df)
    test_df = _encode_categorical(test_df)

    # 按城市+时间排序后 shift(-1) 得到明天的天气
    for df in (train_df, test_df):
        df.sort_values(["city_id", "time"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        df["target"] = df.groupby("city_id")["target"].shift(-1)
        df.dropna(subset=["target"], inplace=True)

    # 时序特征增强：多步滞后 + 多尺度滑动窗口 + 气压变化率 + Cloudy 专项 + 交互特征
    for i, df in enumerate([train_df, test_df]):
        df = _add_lag_features(df, config.LAG_COLS_DAILY, config.LAG_PERIODS_DAILY)
        df = _add_multi_scale_rolling(df, ["temperature_2m_mean", "precipitation_sum",
                                            "pressure_mean", "cloud_cover_mean"],
                                      config.ROLLING_WINDOWS_DAILY)
        df = _add_pressure_change(df, "pressure_mean",
                                 shift_period=config.DAILY_PRESSURE_CHANGE_PERIOD)
        df = _add_cloud_features(df, "cloud_cover_mean")
        df = _add_interaction_features_daily(df)
        if i == 0:
            train_df = df
        else:
            test_df = df

    # 删除因最大滞后(30)产生的 NaN 行
    max_lag_col = f"{config.LAG_COLS_DAILY[0]}_lag_{config.LAG_PERIODS_DAILY[-1]}"
    if max_lag_col in train_df.columns:
        train_df.dropna(subset=[max_lag_col], inplace=True)
        test_df.dropna(subset=[max_lag_col], inplace=True)
    for df in (train_df, test_df):
        roll_nan_cols = [c for c in df.columns
                         if ("_roll" in c and c.endswith("_std"))
                         or c.endswith("_variability") or c.endswith("_trend")]
        df[roll_nan_cols] = df[roll_nan_cols].fillna(0)

    # 删除 date 辅助列
    for df in (train_df, test_df):
        df.drop(columns=["date"], inplace=True, errors="ignore")

    feature_cols = _get_feature_cols(train_df, "daily")
    if verbose:
        print(f"特征数: {len(feature_cols)}")
        print(f"标签分布(train):\n{train_df['target'].value_counts(normalize=True).round(4).to_string()}")

    return (train_df[feature_cols].values, train_df["target"].values,
            test_df[feature_cols].values, test_df["target"].values,
            feature_cols)


def _aggregate_daily_labels(verbose: bool = True) -> dict:
    """
    从小时级清洗数据按城市+日期聚合"当天最严重天气"

    聚合策略：当天出现过的所有天气类别中，取优先级最高的
    优先级：Snow > Rain > Drizzle > Overcast > Cloudy > Clear

    Returns:
        {(city, date): category} 字典
    """
    if verbose:
        print("  从小时级清洗数据聚合日级标签...")

    h = pd.read_csv(config.HOURLY_CLEANED_PATH,
                    usecols=["city", "time", "weather_code"])
    h["date"] = pd.to_datetime(h["time"]).dt.date
    h["category"] = h["weather_code"].map(config.WMO_TO_CATEGORY)
    h.dropna(subset=["category"], inplace=True)

    # 当天最严重天气
    priority_index = {cat: i for i, cat in enumerate(config.DAILY_PRIORITY)}

    def most_severe(categories):
        return min(set(categories), key=lambda c: priority_index.get(c, len(priority_index)))

    daily_label = h.groupby(["city", "date"])["category"].agg(most_severe)
    daily_label = daily_label.to_dict()

    if verbose:
        print(f"  聚合完成: {len(daily_label):,} 个 (city, date) 标签")

    return daily_label


def _aggregate_hourly_to_daily_features(verbose: bool = True) -> pd.DataFrame:
    """
    从小时级清洗数据按城市+日期聚合出日级补充特征

    日级清洗数据天生缺失气压/湿度/云量/风向，这些是最强天气预测信号。
    从小时级数据按 city+date 聚合出以下特征：

    - pressure_mean / pressure_min: 气压均值/最低（低气压系统=恶劣天气）
    - humidity_mean / humidity_max: 湿度均值/峰值（高湿=降水/阴天）
    - cloud_cover_mean / cloud_cover_max: 云量均值/峰值（直接决定阴/晴分类）
    - wind_speed_mean: 平均风速
    - wind_direction_mean: 主导风向（circular mean via u/v components）
    - today_weather_cat: 当天最严重天气类别（编码为整数，当前天气状态是最强预测特征）
    - gust_max: 最大阵风

    Returns:
        DataFrame with columns [city, date, *aggregated features]
    """
    if verbose:
        print("  从小时级清洗数据聚合日级补充特征（气压/湿度/云量/风向）...")

    h = pd.read_csv(config.HOURLY_CLEANED_PATH)
    h["date"] = pd.to_datetime(h["time"]).dt.date

    # 风向 circular mean: convert to u/v, average, convert back
    wind_rad = np.deg2rad(h["wind_direction_10m"])
    h["wind_u"] = np.sin(wind_rad)
    h["wind_v"] = np.cos(wind_rad)

    # 当天最严重天气
    h["category"] = h["weather_code"].map(config.WMO_TO_CATEGORY)
    priority_index = {cat: i for i, cat in enumerate(config.DAILY_PRIORITY)}
    h["cat_priority"] = h["category"].map(
        lambda c: priority_index.get(c, len(priority_index)))

    agg = h.groupby(["city", "date"]).agg(
        pressure_mean=("pressure_msl", "mean"),
        pressure_min=("pressure_msl", "min"),
        humidity_mean=("relative_humidity_2m", "mean"),
        humidity_max=("relative_humidity_2m", "max"),
        cloud_cover_mean=("cloud_cover", "mean"),
        cloud_cover_max=("cloud_cover", "max"),
        wind_speed_mean=("wind_speed_10m", "mean"),
        gust_max=("wind_gusts_10m", "max"),
        wind_u_mean=("wind_u", "mean"),
        wind_v_mean=("wind_v", "mean"),
        today_cat_priority=("cat_priority", "min"),  # min priority = most severe
    ).reset_index()

    # Convert wind u/v back to direction
    agg["wind_direction_mean"] = (
        np.rad2deg(np.arctan2(agg["wind_u_mean"], agg["wind_v_mean"])) + 360) % 360
    agg.drop(columns=["wind_u_mean", "wind_v_mean"], inplace=True)

    # Convert today_cat_priority back to category string then to int code
    inv_priority = {v: k for k, v in priority_index.items()}
    agg["today_weather_cat"] = agg["today_cat_priority"].map(inv_priority)
    cat_to_int = {c: i for i, c in enumerate(config.WEATHER_CATEGORIES)}
    agg["today_weather_cat"] = agg["today_weather_cat"].map(cat_to_int)
    agg.drop(columns=["today_cat_priority"], inplace=True)

    if verbose:
        print(f"  聚合完成: {len(agg):,} 行, {len(agg.columns)} 列")
        print(f"  新增列: {[c for c in agg.columns if c not in ('city','date')]}")

    return agg


def _subsample_by_city(df: pd.DataFrame, frac: float) -> pd.DataFrame:
    """
    按城市分组子采样（系统抽样：每隔 k 行取一行，保持时序全覆盖）
    避免只取冬季数据导致的类别分布偏移
    """
    parts = []
    for city_id, group in df.groupby("city_id"):
        step = max(int(1 / frac), 1)
        # 系统抽样：每隔 step 行取一行，覆盖全年各季节
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
    print("=== 测试日级数据加载 ===")
    X_train, y_train, X_test, y_test, feat = load_daily()
    print(f"\nX_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
    print(f"特征列示例: {feat[:5]}...")
    print(f"\n标签类别: {np.unique(y_train)}")

    print("\n=== 测试小时级数据加载（子采样 5%） ===")
    X_train_h, y_train_h, X_test_h, y_test_h, feat_h = load_hourly(subset_frac=0.05)
    print(f"\nX_train: {X_train_h.shape}, y_train: {y_train_h.shape}")
    print(f"X_test: {X_test_h.shape}, y_test: {y_test_h.shape}")
    print(f"标签类别: {np.unique(y_train_h)}")
