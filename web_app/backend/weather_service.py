"""
天气预测服务
- 从样本集中加载城市经纬度
- 调用 Open-Meteo API 获取未来预报
- 运行 Hourly + Daily 模型
- 天气状况解读
"""
import math
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import httpx

_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parent.parent

# Open-Meteo API
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

HOURLY_PARAMS = [
    "temperature_2m", "apparent_temperature", "relative_humidity_2m",
    "precipitation", "rain", "snowfall", "pressure_msl",
    "cloud_cover", "shortwave_radiation",
    "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
    "weather_code",
]

DAILY_PARAMS = [
    "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
    "apparent_temperature_max", "apparent_temperature_min", "apparent_temperature_mean",
    "precipitation_sum", "rain_sum", "snowfall_sum",
    "wind_speed_10m_max", "wind_gusts_10m_max",
    "shortwave_radiation_sum", "weather_code",
]

# ==================== 49个样本集城市经纬度 ====================
CITIES_CONFIG: Dict[str, Dict] = {
    "London":       {"latitude": 51.5072, "longitude": -0.1276, "country": "UK"},
    "Paris":        {"latitude": 48.8566, "longitude": 2.3522, "country": "France"},
    "Berlin":       {"latitude": 52.5200, "longitude": 13.4050, "country": "Germany"},
    "Madrid":       {"latitude": 40.4168, "longitude": -3.7038, "country": "Spain"},
    "Rome":         {"latitude": 41.9028, "longitude": 12.4964, "country": "Italy"},
    "Amsterdam":    {"latitude": 52.3676, "longitude": 4.9041, "country": "Netherlands"},
    "Brussels":     {"latitude": 50.8503, "longitude": 4.3517, "country": "Belgium"},
    "Vienna":       {"latitude": 48.2082, "longitude": 16.3738, "country": "Austria"},
    "Zurich":       {"latitude": 47.3769, "longitude": 8.5417, "country": "Switzerland"},
    "Lisbon":       {"latitude": 38.7223, "longitude": -9.1393, "country": "Portugal"},
    "Milan":        {"latitude": 45.4642, "longitude": 9.1900, "country": "Italy"},
    "Stockholm":    {"latitude": 59.3293, "longitude": 18.0686, "country": "Sweden"},
    "Oslo":         {"latitude": 59.9139, "longitude": 10.7522, "country": "Norway"},
    "Copenhagen":   {"latitude": 55.6761, "longitude": 12.5683, "country": "Denmark"},
    "Helsinki":     {"latitude": 60.1699, "longitude": 24.9384, "country": "Finland"},
    "Reykjavik":    {"latitude": 64.1466, "longitude": -21.9426, "country": "Iceland"},
    "Warsaw":       {"latitude": 52.2297, "longitude": 21.0122, "country": "Poland"},
    "Prague":       {"latitude": 50.0755, "longitude": 14.4378, "country": "Czech"},
    "Budapest":     {"latitude": 47.4979, "longitude": 19.0402, "country": "Hungary"},
    "Bucharest":    {"latitude": 44.4268, "longitude": 26.1025, "country": "Romania"},
    "Sofia":        {"latitude": 42.6977, "longitude": 23.3219, "country": "Bulgaria"},
    "Athens":       {"latitude": 37.9838, "longitude": 23.7275, "country": "Greece"},
    "Barcelona":    {"latitude": 41.3851, "longitude": 2.1734, "country": "Spain"},
    "Valencia":     {"latitude": 39.4699, "longitude": -0.3763, "country": "Spain"},
    "Seville":      {"latitude": 37.3891, "longitude": -5.9845, "country": "Spain"},
    "Naples":       {"latitude": 40.8518, "longitude": 14.2681, "country": "Italy"},
    "Turin":        {"latitude": 45.0703, "longitude": 7.6869, "country": "Italy"},
    "Florence":     {"latitude": 43.7696, "longitude": 11.2558, "country": "Italy"},
    "Munich":       {"latitude": 48.1351, "longitude": 11.5820, "country": "Germany"},
    "Hamburg":      {"latitude": 53.5511, "longitude": 9.9937, "country": "Germany"},
    "Frankfurt":    {"latitude": 50.1109, "longitude": 8.6821, "country": "Germany"},
    "Cologne":      {"latitude": 50.9375, "longitude": 6.9603, "country": "Germany"},
    "Dublin":       {"latitude": 53.3498, "longitude": -6.2603, "country": "Ireland"},
    "Edinburgh":    {"latitude": 55.9533, "longitude": -3.1883, "country": "UK"},
    "Manchester":   {"latitude": 53.4808, "longitude": -2.2426, "country": "UK"},
    "Birmingham":   {"latitude": 52.4862, "longitude": -1.8904, "country": "UK"},
    "Lyon":         {"latitude": 45.7640, "longitude": 4.8357, "country": "France"},
    "Marseille":    {"latitude": 43.2965, "longitude": 5.3698, "country": "France"},
    "Toulouse":     {"latitude": 43.6047, "longitude": 1.4442, "country": "France"},
    "Nice":         {"latitude": 43.7102, "longitude": 7.2620, "country": "France"},
    "Tallinn":      {"latitude": 59.4370, "longitude": 24.7536, "country": "Estonia"},
    "Riga":         {"latitude": 56.9496, "longitude": 24.1052, "country": "Latvia"},
    "Vilnius":      {"latitude": 54.6872, "longitude": 25.2797, "country": "Lithuania"},
    "Belgrade":     {"latitude": 44.7866, "longitude": 20.4489, "country": "Serbia"},
    "Zagreb":       {"latitude": 45.8150, "longitude": 15.9819, "country": "Croatia"},
    "Ljubljana":    {"latitude": 46.0569, "longitude": 14.5058, "country": "Slovenia"},
    "Porto":        {"latitude": 41.1579, "longitude": -8.6291, "country": "Portugal"},
    "Geneva":       {"latitude": 46.2044, "longitude": 6.1432, "country": "Switzerland"},
    "Basel":        {"latitude": 47.5596, "longitude": 7.5886, "country": "Switzerland"},
}
CITY_NAMES = sorted(CITIES_CONFIG.keys())

# ==================== 天气代码 ====================
WEATHER_CODES: Dict[int, str] = {
    0: "晴天", 1: "大部晴朗", 2: "多云", 3: "阴天",
    45: "有雾", 48: "雾凇",
    51: "小毛毛雨", 53: "毛毛雨", 55: "大毛毛雨",
    56: "小冻毛毛雨", 57: "冻毛毛雨",
    61: "小雨", 63: "中雨", 65: "大雨",
    66: "小冻雨", 67: "冻雨",
    71: "小雪", 73: "中雪", 75: "大雪", 77: "雪粒",
    80: "小阵雨", 81: "阵雨", 82: "大阵雨",
    85: "小阵雪", 86: "阵雪",
    95: "雷暴", 96: "雷暴+小冰雹", 99: "雷暴+大冰雹",
}

WEATHER_CATEGORY: Dict[int, str] = {
    0: "sunny", 1: "sunny", 2: "cloudy", 3: "cloudy",
    45: "fog", 48: "fog",
    51: "rain", 53: "rain", 55: "rain", 56: "rain", 57: "rain",
    61: "rain", 63: "rain", 65: "rain", 66: "rain", 67: "rain",
    71: "snow", 73: "snow", 75: "snow", 77: "snow",
    80: "rain", 81: "rain", 82: "rain",
    85: "snow", 86: "snow",
    95: "storm", 96: "storm", 99: "storm",
}


def _wc_desc(code: int) -> str:
    return WEATHER_CODES.get(code, f"未知({code})")


def _wc_cat(code: int) -> str:
    return WEATHER_CATEGORY.get(code, "cloudy")


# ==================== API ====================

async def fetch_forecast(lat: float, lon: float) -> Dict:
    """从 Open-Meteo 获取未来16天预报数据"""
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": ",".join(HOURLY_PARAMS),
        "daily": ",".join(DAILY_PARAMS),
        "timezone": "auto",
        "forecast_days": 16,
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(FORECAST_URL, params=params)
        resp.raise_for_status()
        return resp.json()


# ==================== 特征工程 ====================

def _cyclical(values: np.ndarray, period: float) -> Tuple[np.ndarray, np.ndarray]:
    rad = 2 * math.pi * values / period
    return np.sin(rad), np.cos(rad)


def engineer_hourly(df: pd.DataFrame) -> Tuple:
    """
    构建小时模型输入：numerical(35), cyclical(8), binary(15), categorical(3), season(1), day_period(1)
    返回 6 个数组，每个 shape (n_samples, n_feat)
    """
    n = len(df)
    t = pd.to_datetime(df['time'])
    # 确保是 DatetimeIndex
    if isinstance(t, pd.Series):
        t = pd.DatetimeIndex(t)
    lat = float(df['latitude'].iloc[0]) if 'latitude' in df.columns else 0
    lon = float(df['longitude'].iloc[0]) if 'longitude' in df.columns else 0

    numerical = np.zeros((n, 35), dtype=np.float32)
    numerical[:, 0] = lat
    numerical[:, 1] = lon

    col_map = {
        'temperature_2m': 2, 'apparent_temperature': 3,
        'relative_humidity_2m': 4, 'pressure_msl': 5,
        'precipitation': 6, 'rain': 7, 'snowfall': 8,
        'cloud_cover': 9, 'shortwave_radiation': 10,
        'wind_speed_10m': 11, 'wind_direction_10m': 12, 'wind_gusts_10m': 13,
    }
    for col, idx in col_map.items():
        if col in df.columns:
            numerical[:, idx] = df[col].fillna(0).values

    T = numerical[:, 2]
    V = np.clip(numerical[:, 11], 0.1, None)
    RH = np.clip(numerical[:, 4], 0, 100)
    V_pow = np.power(V, 0.16)

    # 派生特征 (14-21)
    numerical[:, 14] = np.where(T <= 10, 13.12 + 0.6215 * T - 11.37 * V_pow + 0.3965 * T * V_pow, T)
    numerical[:, 15] = np.where(T > 27, -8.784695 + 1.61139411 * T + 2.338549 * RH - 0.14611605 * T * RH - 0.01230809 * T**2 - 0.01642483 * RH**2 + 0.00221173 * T**2 * RH + 0.00072546 * T * RH**2 - 0.00000358 * T**2 * RH**2, T)
    numerical[:, 16] = np.clip(numerical[:, 6] - numerical[:, 7], 0, None)
    numerical[:, 17] = np.where(V > 0.1, numerical[:, 13] / V, 0)
    wd_rad = np.deg2rad(numerical[:, 12])
    numerical[:, 18] = V * np.cos(wd_rad)
    numerical[:, 19] = V * np.sin(wd_rad)
    prec = numerical[:, 6]
    numerical[:, 20] = np.select([prec <= 0.01, prec <= 2.5, prec <= 7.6, prec <= 25, prec > 25], [0, 1, 2, 3, 4])
    numerical[:, 21] = (numerical[:, 20] >= 3).astype(float) + (numerical[:, 13] > 20).astype(float) + (numerical[:, 11] > 15).astype(float)

    # 等级特征 (22-26)
    numerical[:, 22] = pd.cut(T, bins=[-np.inf, -10, 0, 10, 20, 30, 35, np.inf], labels=[0,1,2,3,4,5,6]).codes.astype(float)
    numerical[:, 23] = pd.cut(V, bins=[-np.inf, 1, 5, 11, 19, 28, np.inf], labels=[0,1,2,3,4,5]).codes.astype(float)
    numerical[:, 24] = pd.cut(RH, bins=[-np.inf, 30, 50, 70, 90, np.inf], labels=[0,1,2,3,4]).codes.astype(float)
    numerical[:, 25] = pd.cut(numerical[:, 5], bins=[-np.inf, 980, 1000, 1020, 1040, np.inf], labels=[0,1,2,3,4]).codes.astype(float)
    numerical[:, 26] = pd.cut(numerical[:, 9], bins=[-np.inf, 20, 50, 80, 100, np.inf], labels=[0,1,2,3,4]).codes.astype(float)

    # 时间特征 (27-34)
    numerical[:, 27] = t.year.to_numpy().astype(np.float32)
    numerical[:, 28] = t.month.to_numpy().astype(np.float32)
    numerical[:, 29] = t.day.to_numpy().astype(np.float32)
    numerical[:, 30] = t.dayofyear.to_numpy().astype(np.float32)
    numerical[:, 31] = t.dayofweek.to_numpy().astype(np.float32)
    numerical[:, 32] = t.quarter.to_numpy().astype(np.float32)
    numerical[:, 33] = np.array([d.isocalendar()[1] for d in t], dtype=np.float32)
    numerical[:, 34] = t.hour.to_numpy().astype(np.float32)

    # 周期特征 (8)
    cyclical = np.zeros((n, 8), dtype=np.float32)
    cyclical[:, 0], cyclical[:, 1] = _cyclical(t.month.to_numpy(), 12)
    cyclical[:, 2], cyclical[:, 3] = _cyclical(t.dayofyear.to_numpy(), 365)
    cyclical[:, 4], cyclical[:, 5] = _cyclical(t.dayofweek.to_numpy(), 7)
    cyclical[:, 6], cyclical[:, 7] = _cyclical(t.hour.to_numpy(), 24)

    # 二值特征 (15)
    binary = np.zeros((n, 15), dtype=np.float32)
    binary[:, 0] = (t.dayofweek.to_numpy() >= 5).astype(float)
    binary[:, 1] = (T < 0).astype(float)
    binary[:, 2] = (T > 35).astype(float)
    binary[:, 3] = (numerical[:, 6] > 0).astype(float)
    binary[:, 4] = (numerical[:, 8] > 0).astype(float)
    binary[:, 5] = (numerical[:, 16] > 0).astype(float)
    binary[:, 6] = (V > 10).astype(float)
    binary[:, 7] = (V > 20).astype(float)
    binary[:, 8] = (RH < 30).astype(float)
    binary[:, 9] = (RH > 80).astype(float)
    binary[:, 10] = (numerical[:, 9] < 20).astype(float)
    binary[:, 11] = (numerical[:, 9] > 80).astype(float)
    binary[:, 12] = (numerical[:, 21] >= 2).astype(float)
    binary[:, 13] = float(abs(lat) > 55)
    binary[:, 14] = float((abs(lat) < 45) & (abs(lon) < 40))

    # 类别特征 (3): city_id=25(默认), country_id=14(默认), weather_code_id
    categorical = np.zeros((n, 3), dtype=np.float32)
    categorical[:, 0] = 25
    categorical[:, 1] = 14
    if 'weather_code' in df.columns:
        wc_map = {0:0,1:1,2:2,3:3,45:4,48:4,51:5,53:5,55:5,56:5,57:5,61:6,63:6,65:6,66:6,67:6,71:7,73:7,75:7,77:7,80:8,81:8,82:8,85:9,86:9,95:10,96:11,99:12}
        wc = df['weather_code'].fillna(0).values
        categorical[:, 2] = np.array([wc_map.get(int(c), 0) for c in wc])

    # season (1)
    season = np.zeros((n, 1), dtype=np.float32)
    m = t.month.to_numpy()
    season[np.isin(m, [12,1,2]), 0] = 0
    season[np.isin(m, [3,4,5]), 0] = 1
    season[np.isin(m, [6,7,8]), 0] = 2
    season[np.isin(m, [9,10,11]), 0] = 3

    # day_period (1)
    day_period = np.zeros((n, 1), dtype=np.float32)
    h = t.hour.to_numpy()
    day_period[(h >= 6) & (h < 12), 0] = 1
    day_period[(h >= 12) & (h < 18), 0] = 2
    day_period[(h >= 18) & (h < 22), 0] = 3
    day_period[(h < 6), 0] = 4
    day_period[(h >= 22), 0] = 4

    return numerical, cyclical, binary, categorical, season, day_period


def engineer_daily(df: pd.DataFrame) -> Tuple:
    """
    构建日模型输入：numerical(22), cyclical(6), binary(9), categorical(2), season(1)
    注意：训练时的 numerical 包含 ORDINAL + TIME = 12 + 3 + 7 = 22
    """
    n = len(df)
    t = pd.to_datetime(df['time'])
    if isinstance(t, pd.Series):
        t = pd.DatetimeIndex(t)
    lat = float(df['latitude'].iloc[0]) if 'latitude' in df.columns else 0
    lon = float(df['longitude'].iloc[0]) if 'longitude' in df.columns else 0

    numerical = np.zeros((n, 22), dtype=np.float32)
    numerical[:, 0] = lat
    numerical[:, 1] = lon

    # 温度 (2-6)
    for col, idx in [('temperature_2m_max', 2), ('temperature_2m_min', 3), ('temperature_2m_mean', 4)]:
        if col in df.columns:
            numerical[:, idx] = df[col].fillna(0).values
    numerical[:, 5] = numerical[:, 2] - numerical[:, 3]  # temperature_range

    # feels_like
    if 'apparent_temperature_mean' in df.columns:
        numerical[:, 6] = df['apparent_temperature_mean'].fillna(0).values

    # 降水 (7-9)
    for col, idx in [('precipitation_sum', 7), ('rain_sum', 8)]:
        if col in df.columns:
            numerical[:, idx] = df[col].fillna(0).values
    numerical[:, 9] = np.clip(numerical[:, 7] - numerical[:, 8], 0, None)

    # 风 & 辐射 (10-11)
    if 'wind_speed_10m_max' in df.columns:
        numerical[:, 10] = df['wind_speed_10m_max'].fillna(0).values
    if 'shortwave_radiation_sum' in df.columns:
        numerical[:, 11] = df['shortwave_radiation_sum'].fillna(0).values

    # 时间特征 (12-18)
    numerical[:, 12] = t.month.to_numpy().astype(np.float32)
    numerical[:, 13] = t.year.to_numpy().astype(np.float32)
    numerical[:, 14] = t.day.to_numpy().astype(np.float32)
    numerical[:, 15] = t.dayofyear.to_numpy().astype(np.float32)
    numerical[:, 16] = t.dayofweek.to_numpy().astype(np.float32)
    numerical[:, 17] = t.quarter.to_numpy().astype(np.float32)
    numerical[:, 18] = np.array([d.isocalendar()[1] for d in t], dtype=np.float32)

    # 等级特征 (19-21)
    numerical[:, 19] = pd.cut(numerical[:, 7], bins=[-np.inf, 0.1, 2.5, 10, 25, np.inf], labels=[0,1,2,3,4]).codes.astype(float)
    numerical[:, 20] = pd.cut(numerical[:, 10], bins=[-np.inf, 6, 12, 20, 30, 40, np.inf], labels=[0,1,2,3,4,5]).codes.astype(float)
    numerical[:, 21] = pd.cut(numerical[:, 11], bins=[-np.inf, 5000, 15000, np.inf], labels=[0,1,2]).codes.astype(float)

    # 周期特征 (6)
    cyclical = np.zeros((n, 6), dtype=np.float32)
    cyclical[:, 0], cyclical[:, 1] = _cyclical(t.month.to_numpy(), 12)
    cyclical[:, 2], cyclical[:, 3] = _cyclical(t.dayofyear.to_numpy(), 365)
    cyclical[:, 4], cyclical[:, 5] = _cyclical(t.dayofweek.to_numpy(), 7)

    # 二值特征 (9)
    binary = np.zeros((n, 9), dtype=np.float32)
    binary[:, 0] = (numerical[:, 3] < 0).astype(float)
    binary[:, 1] = (numerical[:, 2] > 30).astype(float)
    binary[:, 2] = (numerical[:, 7] > 0).astype(float)
    binary[:, 3] = (numerical[:, 7] > 10).astype(float)
    binary[:, 4] = (numerical[:, 9] > 0).astype(float)
    binary[:, 5] = (numerical[:, 10] > 10).astype(float)
    binary[:, 6] = (numerical[:, 11] > 10000).astype(float)
    binary[:, 7] = (numerical[:, 21] >= 2).astype(float)
    binary[:, 8] = float(abs(lat) > 55)

    # 类别 (2)
    categorical = np.zeros((n, 2), dtype=np.float32)
    categorical[:, 0] = 25
    categorical[:, 1] = 14

    # season (1)
    season = np.zeros((n, 1), dtype=np.float32)
    m = t.month.to_numpy()
    season[np.isin(m, [12,1,2]), 0] = 0
    season[np.isin(m, [3,4,5]), 0] = 1
    season[np.isin(m, [6,7,8]), 0] = 2
    season[np.isin(m, [9,10,11]), 0] = 3

    return numerical, cyclical, binary, categorical, season


# ==================== 预测 ====================

def _build_hourly_batch(arrays):
    """构建小时模型 batch 格式 (1, seq_len, n_feat)"""
    return {
        'numerical': np.expand_dims(arrays[0], 0),
        'cyclical': np.expand_dims(arrays[1], 0),
        'binary': np.expand_dims(arrays[2], 0),
        'categorical': np.expand_dims(arrays[3], 0),
        'season': np.expand_dims(arrays[4], 0),
        'day_period': np.expand_dims(arrays[5], 0),
    }


def _build_daily_batch(arrays):
    """构建日模型 batch 格式 (1, seq_len, n_feat)"""
    return {
        'numerical': np.expand_dims(arrays[0], 0),
        'cyclical': np.expand_dims(arrays[1], 0),
        'binary': np.expand_dims(arrays[2], 0),
        'categorical': np.expand_dims(arrays[3], 0),
        'season': np.expand_dims(arrays[4], 0),
    }


async def predict(city: str, target_time: str) -> Dict:
    """
    核心预测函数：
    1. 从 CITIES_CONFIG 获取经纬度
    2. 调用 Open-Meteo API 获取预报
    3. 运行 Hourly + Daily 模型
    """
    if city not in CITIES_CONFIG:
        raise ValueError(f"城市 '{city}' 不在样本集中。可用: {', '.join(CITY_NAMES[:10])}...共{len(CITY_NAMES)}个")

    cfg = CITIES_CONFIG[city]
    lat, lon = cfg["latitude"], cfg["longitude"]
    target_dt = datetime.fromisoformat(target_time)

    # 请求未来16天预报
    raw = await fetch_forecast(lat, lon)

    hourly = _predict_hourly(raw, lat, lon, target_dt)
    daily = _predict_daily(raw, lat, lon, target_dt)

    return {
        "city": city,
        "country": cfg["country"],
        "latitude": lat,
        "longitude": lon,
        "target_time": target_time,
        "hourly": hourly,
        "daily": daily,
    }


def _predict_hourly(raw: Dict, lat: float, lon: float, target_dt: datetime) -> Dict:
    hourly = raw.get("hourly", {})
    times = pd.to_datetime(hourly.get("time", []))
    if len(times) == 0:
        return {"error": "无小时预报数据"}

    df = pd.DataFrame(hourly)
    df['time'] = times
    df['latitude'] = lat
    df['longitude'] = lon

    num, cyc, bin_f, cat, season, dp = engineer_hourly(df)

    # 找目标时间索引
    target_hour = target_dt.replace(minute=0, second=0, microsecond=0)
    target_idx = None
    for i, ts in enumerate(times):
        dt = ts.to_pydatetime()
        if abs((dt - target_hour).total_seconds()) <= 3600:
            target_idx = i
            break
    if target_idx is None:
        diffs = [abs((ts.to_pydatetime() - target_dt).total_seconds()) for ts in times]
        target_idx = int(np.argmin(diffs))
    if target_idx < 24:
        target_idx = 24 if len(times) > 24 else max(0, len(times) - 1)

    # 构建序列（24小时窗口）
    seq_len = 24
    start = target_idx - seq_len + 1
    arrays = [num, cyc, bin_f, cat, season, dp]
    seq = [a[start:start + seq_len] for a in arrays]

    # API 数据
    if target_idx < len(times):
        target_time_str = str(times[target_idx])
    else:
        target_time_str = target_dt.isoformat()

    api_data = {}
    for col in ["temperature_2m", "apparent_temperature", "relative_humidity_2m",
                "precipitation", "rain", "snowfall", "pressure_msl",
                "cloud_cover", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m", "weather_code"]:
        if col in hourly and target_idx < len(hourly[col]):
            api_data[col] = hourly[col][target_idx]

    wc = int(api_data.get("weather_code", 0))

    # 模型预测
    model_pred = None
    try:
        import torch
        from hourly_ensemble.model_wrapper import HourModelWrapper
        from hourly_ensemble.post_process import post_process

        model = HourModelWrapper()
        batch = _build_hourly_batch(seq)
        t_batch = {k: torch.tensor(v, dtype=torch.float32) for k, v in batch.items()}
        with torch.no_grad():
            pred = model.model(t_batch).cpu().numpy()[0]
        pred_real = post_process(pred.reshape(1, -1))[0]
        model_pred = {
            "temperature_2m": round(float(pred_real[0]), 1),
            "precipitation": round(float(pred_real[1]), 2),
            "wind_speed_10m": round(float(pred_real[2]), 1),
            "apparent_temperature": round(float(pred_real[3]), 1),
            "relative_humidity_2m": round(float(pred_real[4]), 1),
        }
    except Exception as e:
        model_pred = None

    return {
        "time": target_time_str,
        "api_forecast": api_data,
        "model_prediction": model_pred,
        "weather_code": wc,
        "weather_desc": _wc_desc(wc),
        "weather_category": _wc_cat(wc),
    }


def _predict_daily(raw: Dict, lat: float, lon: float, target_dt: datetime) -> Dict:
    daily = raw.get("daily", {})
    times = pd.to_datetime(daily.get("time", []))
    if len(times) == 0:
        return {"error": "无日预报数据"}

    df = pd.DataFrame(daily)
    df['time'] = times
    df['latitude'] = lat
    df['longitude'] = lon

    num, cyc, bin_f, cat, season = engineer_daily(df)

    # 找目标日期
    target_date = target_dt.date()
    target_idx = None
    for i, t in enumerate(times):
        if t.date() == target_date:
            target_idx = i
            break
    if target_idx is None:
        diffs = [abs((t.date() - target_date).days) for t in times]
        target_idx = int(np.argmin(diffs))
    if target_idx < 7:
        target_idx = 7 if len(times) > 7 else max(0, len(times) - 1)

    # 构建序列（7天窗口）
    seq_len = 7
    start = target_idx - seq_len + 1
    arrays = [num, cyc, bin_f, cat, season]
    seq = [a[start:start + seq_len] for a in arrays]

    # API 数据
    if target_idx < len(times):
        target_date_str = str(times[target_idx].date())
    else:
        target_date_str = target_date.isoformat()

    api_data = {}
    for col in ["temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
                "apparent_temperature_max", "apparent_temperature_min", "apparent_temperature_mean",
                "precipitation_sum", "rain_sum", "snowfall_sum",
                "wind_speed_10m_max", "wind_gusts_10m_max",
                "shortwave_radiation_sum", "weather_code"]:
        if col in daily and target_idx < len(daily[col]):
            api_data[col] = daily[col][target_idx]

    wc = int(api_data.get("weather_code", 0))

    # 模型预测
    model_pred = None
    try:
        import torch
        from daily_ensemble.model_wrapper import Model3Wrapper

        model = Model3Wrapper()
        batch = _build_daily_batch(seq)
        t_batch = {k: torch.tensor(v, dtype=torch.float32) for k, v in batch.items()}
        with torch.no_grad():
            pred = model.model(t_batch).cpu().numpy()[0]
        model_pred = {
            "temperature_2m_mean": round(float(pred[0]), 1),
            "temperature_2m_max": round(float(pred[1]), 1),
            "temperature_2m_min": round(float(pred[2]), 1),
            "temperature_range": round(float(pred[3]), 1),
            "feels_like_temperature": round(float(pred[4]), 1),
            "precipitation_sum": round(float(pred[5]), 2),
            "rain_sum": round(float(pred[6]), 2),
            "snow_sum": round(float(pred[7]), 2),
            "wind_speed_10m_max": round(float(pred[8]), 1),
        }
    except Exception as e:
        model_pred = None

    return {
        "date": target_date_str,
        "api_forecast": api_data,
        "model_prediction": model_pred,
        "weather_code": wc,
        "weather_desc": _wc_desc(wc),
        "weather_category": _wc_cat(wc),
    }
