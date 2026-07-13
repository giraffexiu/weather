#!/usr/bin/env python3
"""
从Open-Meteo API获取欧洲主要城市的历史天气数据
数据范围: 2020-01-01 到 2024-12-31
输出格式: CSV文件
"""

import requests
import pandas as pd
from datetime import datetime
import time
import os

# 城市配置
CITIES = {
    "London": {"latitude": 51.5072, "longitude": -0.1276},
    "Paris": {"latitude": 48.8566, "longitude": 2.3522},
    "Berlin": {"latitude": 52.5200, "longitude": 13.4050},
    "Madrid": {"latitude": 40.4168, "longitude": -3.7038},
    "Rome": {"latitude": 41.9028, "longitude": 12.4964},
    "Amsterdam": {"latitude": 52.3676, "longitude": 4.9041},
    "Brussels": {"latitude": 50.8503, "longitude": 4.3517},
    "Vienna": {"latitude": 48.2082, "longitude": 16.3738},
    "Zurich": {"latitude": 47.3769, "longitude": 8.5417},
    "Stockholm": {"latitude": 59.3293, "longitude": 18.0686},
    "Oslo": {"latitude": 59.9139, "longitude": 10.7522},
    "Copenhagen": {"latitude": 55.6761, "longitude": 12.5683},
    "Helsinki": {"latitude": 60.1699, "longitude": 24.9384},
    "Lisbon": {"latitude": 38.7223, "longitude": -9.1393},
    "Milan": {"latitude": 45.4642, "longitude": 9.1900}
}

# 时间范围
START_DATE = "2020-01-01"
END_DATE = "2024-12-31"

# 需要获取的变量
HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "pressure_msl",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
    "cloud_cover",
    "shortwave_radiation",
    "uv_index",
    "visibility"
]

# API配置
BASE_URL = "https://archive-api.open-meteo.com/v1/era5"

# 输出目录
OUTPUT_DIR = "../"  # 输出到data目录


def fetch_weather_data(city_name, latitude, longitude):
    """
    获取单个城市的天气数据
    """
    print(f"正在获取 {city_name} 的数据...")
    
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": ",".join(HOURLY_VARIABLES)
    }
    
    try:
        response = requests.get(BASE_URL, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()
        
        if "hourly" in data:
            # 创建DataFrame
            df = pd.DataFrame(data["hourly"])
            
            # 添加城市信息
            df.insert(0, "city", city_name)
            df.insert(1, "latitude", latitude)
            df.insert(2, "longitude", longitude)
            
            print(f"✓ {city_name} 数据获取成功 ({len(df)} 条记录)")
            return df
        else:
            print(f"✗ {city_name} 数据获取失败: 响应中没有hourly数据")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"✗ {city_name} 数据获取失败: {e}")
        return None


def main():
    """
    主函数: 获取所有城市数据并保存为CSV
    """
    print("=" * 60)
    print("开始获取欧洲城市天气数据")
    print(f"时间范围: {START_DATE} 到 {END_DATE}")
    print(f"城市数量: {len(CITIES)}")
    print(f"变量数量: {len(HOURLY_VARIABLES)}")
    print("=" * 60)
    print()
    
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    all_data = []
    
    # 遍历所有城市
    for i, (city_name, coords) in enumerate(CITIES.items(), 1):
        print(f"[{i}/{len(CITIES)}] ", end="")
        
        df = fetch_weather_data(
            city_name,
            coords["latitude"],
            coords["longitude"]
        )
        
        if df is not None:
            all_data.append(df)
        
        # 添加延迟以避免API限流
        if i < len(CITIES):
            time.sleep(1)
        
        print()
    
    # 合并所有数据
    if all_data:
        print("=" * 60)
        print("正在合并数据...")
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # 保存为CSV
        output_file = os.path.join(OUTPUT_DIR, f"weather_data_{START_DATE}_to_{END_DATE}.csv")
        combined_df.to_csv(output_file, index=False, encoding="utf-8")
        
        print(f"✓ 数据已保存到: {output_file}")
        print(f"总记录数: {len(combined_df):,}")
        print(f"文件大小: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
        print()
        
        # 显示数据预览
        print("数据预览 (前5行):")
        print(combined_df.head())
        print()
        
        # 显示数据统计
        print("数据统计:")
        print(f"- 城市数量: {combined_df['city'].nunique()}")
        print(f"- 时间范围: {combined_df['time'].min()} 到 {combined_df['time'].max()}")
        print(f"- 总记录数: {len(combined_df):,}")
        print()
        
        # 每个城市的数据量
        print("各城市数据量:")
        city_counts = combined_df['city'].value_counts().sort_index()
        for city, count in city_counts.items():
            print(f"  {city:12} : {count:,} 条记录")
        
        print()
        print("=" * 60)
        print("数据获取完成!")
        print("=" * 60)
        
    else:
        print("✗ 没有成功获取任何数据")


if __name__ == "__main__":
    main()
