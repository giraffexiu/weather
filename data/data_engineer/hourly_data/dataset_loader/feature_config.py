"""
特征分组配置（小时数据）
定义不同类型的特征列名
"""

# ==================== 类别特征（用于 Embedding） ====================
CATEGORICAL_FEATURES = [
    'city_id',          # 城市ID
    'country_id',       # 国家ID
    'weather_code_id',  # 天气编码ID
]

# ==================== 连续数值特征（已标准化） ====================
NUMERICAL_FEATURES = [
    # 地理位置
    'latitude',
    'longitude',
    
    # 温度特征
    'temperature_2m',
    'apparent_temperature',
    'relative_humidity_2m',
    
    # 气压
    'pressure_msl',
    
    # 降水特征
    'precipitation',
    'rain',
    'snowfall',
    
    # 云量与辐射
    'cloud_cover',
    'shortwave_radiation',
    
    # 风速特征
    'wind_speed_10m',
    'wind_direction_10m',
    'wind_gusts_10m',
    
    # 派生特征
    'wind_chill',
    'heat_index',
    'solid_precip',
    'gust_factor',
    'wind_u',
    'wind_v',
    
    # 强度指标
    'precipitation_intensity',
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
    'temperature_level',    # 温度等级
    'wind_level',           # 风力等级
    'humidity_level',       # 湿度等级
    'pressure_level',       # 气压等级
    'cloud_level',          # 云量等级
]

# 注意：等级特征可以选择：
# 1. 当作数值特征直接使用
# 2. 使用 Embedding（类似类别特征）
# 这里默认将其归入数值特征，如需 Embedding 请移到 CATEGORICAL_FEATURES

# ==================== 时间特征（数值类型） ====================
TIME_FEATURES = [
    'year',         # 年份（可捕捉长期趋势/气候变化）
    'month',        # 月份（1-12，离散时间）
    'day',          # 日期（1-31）
    'hour',         # 小时（0-23）
    'day_of_year',  # 一年中的第几天（1-366）
    'day_of_week',  # 星期几（0-6）
    'quarter',      # 季度（1-4）
    'week_of_year', # 一年中的第几周（1-53）
]

# 注意：这些数值时间特征与 sin/cos 周期编码互补：
# - 数值特征：捕捉离散的时间点信息
# - sin/cos 编码：捕捉周期性和连续性
# 两者可以同时使用，让模型学习更丰富的时间模式

# ==================== 季节特征（需要编码） ====================
SEASON_FEATURE = 'season'  # 值: 'winter', 'spring', 'summer', 'autumn'
SEASON_MAPPING = {
    'winter': 0,
    'spring': 1,
    'summer': 2,
    'autumn': 3
}

# ==================== 时段特征（需要编码） ====================
DAY_PERIOD_FEATURE = 'day_period'  # 值: 'night', 'morning', 'forenoon', 'afternoon', 'evening'
DAY_PERIOD_MAPPING = {
    'night': 0,
    'morning': 1,
    'forenoon': 2,
    'afternoon': 3,
    'evening': 4
}

# ==================== 忽略的特征 ====================
# 这些列存在于 CSV 中，但不用于模型输入（因为已有编码版本）
IGNORED_FEATURES = [
    'city',         # 原始城市名（字符串，已编码为 city_id）
    'country',      # 原始国家名（字符串，已编码为 country_id）
    'time',         # 日期时间字符串（已提取为 year, month, day, hour 等数值特征）
    'weather_code', # 原始天气编码（已编码为 weather_code_id）
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
        'numerical': NUMERICAL_FEATURES + ORDINAL_FEATURES + TIME_FEATURES,  # 包含等级特征和时间特征
        'cyclical': CYCLICAL_FEATURES,
        'binary': BINARY_FEATURES,
        'season': [SEASON_FEATURE],
        'day_period': [DAY_PERIOD_FEATURE],
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
        groups['season'] +
        groups['day_period']
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
    
    print("\n" + "="*60)
    print("特征配置摘要（小时数据）")
    print("="*60)
    
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
    print("  - 时间特征(year, month等)与sin/cos编码互补，同时使用")
    
    print("="*60 + "\n")
