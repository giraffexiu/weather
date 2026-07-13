"""
特征分组配置（小时数据）
定义不同类型的特征列名
"""

# ==================== 类别特征（用于 Embedding） ====================
CATEGORICAL_FEATURES = [
    'city_id',           # 城市ID (0-48)
    'country_id',        # 国家ID (0-28)
    'weather_code_id',   # 天气编码ID
]

# ==================== 连续数值特征（已标准化） ====================
NUMERICAL_FEATURES = [
    # 地理位置
    'latitude',
    'longitude',

    # 温度特征
    'temperature_2m',
    'apparent_temperature',
    'wind_chill',
    'heat_index',

    # 降水特征
    'precipitation',
    'rain',
    'snowfall',
    'solid_precip',

    # 风速风向特征
    'wind_speed_10m',
    'wind_direction_10m',
    'wind_gusts_10m',
    'gust_factor',
    'wind_u',
    'wind_v',

    # 湿度/气压特征
    'relative_humidity_2m',
    'pressure_msl',

    # 云量/辐射特征
    'cloud_cover',
    'shortwave_radiation',

    # 恶劣天气指数
    'severe_weather_index',
]

# ==================== 周期性编码特征（sin/cos） ====================
CYCLICAL_FEATURES = [
    'month_sin',
    'month_cos',
    'day_of_year_sin',
    'day_of_year_cos',
    'day_of_week_sin',
    'day_of_week_cos',
    'hour_sin',
    'hour_cos',
]

# ==================== 二值特征（0/1标志位） ====================
BINARY_FEATURES = [
    'is_weekend',
    'is_freezing',
    'is_hot',
    'is_rainy',
    'is_snowy',
    'is_snow',
    'is_windy',
    'is_strong_wind',
    'is_dry',
    'is_humid',
    'is_clear',
    'is_overcast',
    'is_severe_weather',
    'is_high_latitude',
    'is_mediterranean',
]

# ==================== 等级特征（有序分类） ====================
ORDINAL_FEATURES = [
    'temperature_level',       # 0-6
    'precipitation_intensity',  # 0-4
    'wind_level',              # 0-5
    'humidity_level',          # 0-4
    'pressure_level',          # 0-4
    'cloud_level',             # 0-4
]

# ==================== 时间特征（数值类型） ====================
TIME_FEATURES = [
    'year',
    'month',
    'day',
    'hour',
    'day_of_year',
    'day_of_week',
    'quarter',
    'week_of_year',
    'day_period',     # 时段编码 (0-4)
]

# ==================== 季节特征（需要编码） ====================
SEASON_FEATURE = 'season'
SEASON_MAPPING = {
    'winter': 0,
    'spring': 1,
    'summer': 2,
    'autumn': 3
}

# ==================== 忽略的特征 ====================
IGNORED_FEATURES = [
    'city',            # 原始城市名（已编码为 city_id）
    'country',         # 原始国家名（已编码为 country_id）
    'time',            # 时间字符串（已提取为 year, month, day, hour 等）
    'weather_code',    # 原始天气编码（已编码为 weather_code_id）
]

# ==================== 特征组汇总 ====================
def get_feature_groups():
    """
    返回所有特征分组的字典

    Returns:
        dict: 特征分组字典
    """
    return {
        'categorical': CATEGORICAL_FEATURES,
        'numerical': NUMERICAL_FEATURES + ORDINAL_FEATURES + TIME_FEATURES,
        'cyclical': CYCLICAL_FEATURES,
        'binary': BINARY_FEATURES,
        'season': [SEASON_FEATURE],
        'ignored': IGNORED_FEATURES
    }


def get_all_feature_columns():
    """
    获取所有需要使用的特征列名（不包括忽略的）

    Returns:
        list: 特征列名列表
    """
    groups = get_feature_groups()
    all_features = (
        groups['categorical'] +
        groups['numerical'] +
        groups['cyclical'] +
        groups['binary'] +
        groups['season']
    )
    return all_features


def get_feature_dims():
    """
    获取各组特征的维度

    Returns:
        dict: 特征维度字典
    """
    groups = get_feature_groups()
    return {
        'categorical': len(groups['categorical']),
        'numerical': len(groups['numerical']),
        'cyclical': len(groups['cyclical']),
        'binary': len(groups['binary']),
    }


def print_feature_summary():
    """打印特征配置摘要"""
    groups = get_feature_groups()
    dims = get_feature_dims()

    print("\n" + "=" * 60)
    print("特征配置摘要（小时数据）")
    print("=" * 60)

    for group_name, features in groups.items():
        if group_name == 'ignored':
            continue
        print(f"\n{group_name.upper()} 特征 ({len(features)} 个):")
        for feat in features:
            print(f"  - {feat}")

    print(f"\n忽略特征 ({len(groups['ignored'])} 个):")
    print(f"  {', '.join(groups['ignored'])}")

    print(f"\n总计使用特征: {sum(dims.values())} 个")

    print("\n注意:")
    print("  - 'season' 是字符串，需要映射为数值: winter=0, spring=1, summer=2, autumn=3")
    print("  - 'day_period' 是字符串，需要映射为数值: night=0, morning=1, forenoon=2, afternoon=3, evening=4")
    print("  - 'weather_code_id' 可用于 Embedding")

    print("=" * 60 + "\n")
