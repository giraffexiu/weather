# 数据清洗配置文件

# 数据路径配置
DAILY_DATA_PATH = "../dataset/daily_data/"
HOURLY_DATA_PATH = "../dataset/hour_data/"
OUTPUT_PATH = "./cleaned_data/"

# 日数据字段配置
DAILY_COLUMNS = {
    'required': ['city', 'time', 'temperature_2m_max', 'temperature_2m_min', 
                 'temperature_2m_mean', 'precipitation_sum'],
    'optional': ['rain_sum', 'wind_speed_10m_max', 'shortwave_radiation_sum']
}

# 小时数据字段配置
HOURLY_COLUMNS = {
    'required': ['city', 'time', 'temperature_2m', 'relative_humidity_2m', 
                 'pressure_msl', 'precipitation'],
    'optional': ['apparent_temperature', 'rain', 'snowfall', 'cloud_cover', 
                 'wind_speed_10m', 'wind_direction_10m', 'shortwave_radiation']
}

# 数值字段范围限制
VALUE_RANGES = {
    'temperature_2m': (-50, 50),
    'temperature_2m_max': (-50, 50),
    'temperature_2m_min': (-50, 50),
    'temperature_2m_mean': (-50, 50),
    'apparent_temperature': (-60, 60),
    'precipitation': (0, 500),
    'precipitation_sum': (0, 500),
    'rain': (0, 500),
    'rain_sum': (0, 500),
    'wind_speed_10m': (0, 200),
    'wind_speed_10m_max': (0, 200),
    'relative_humidity_2m': (0, 100),
    'pressure_msl': (900, 1100),
    'snowfall': (0, 200),
    'cloud_cover': (0, 100),
    'shortwave_radiation': (0, None),
    'shortwave_radiation_sum': (0, None)
}

# 缺失值处理策略
FILL_STRATEGIES = {
    'temperature': 'interpolate',  # 温度使用插值
    'precipitation': 0,  # 降水缺失视为0
    'wind': 'forward',  # 风速使用前向填充
    'other': 'interpolate'  # 其他使用插值
}
