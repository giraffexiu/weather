#!/usr/bin/env python3
"""
从Open-Meteo API获取欧洲主要城市的历史天气数据（增强版）
数据范围: 2000-01-01 到 2024-12-31 (25年数据)
输出格式: CSV文件
目标: 生成约1GB的数据集
特性: 
- 50个欧洲城市
- 38个小时级变量
- 19个每日聚合变量
- 智能重试机制（避免429错误）
- 断点续传功能
"""

import requests
import pandas as pd
from datetime import datetime
import time
import os
import json
from pathlib import Path

# 扩展城市配置 - 50个欧洲主要城市
CITIES = {
    # 西欧
    "London": {"latitude": 51.5072, "longitude": -0.1276, "country": "UK"},
    "Paris": {"latitude": 48.8566, "longitude": 2.3522, "country": "France"},
    "Berlin": {"latitude": 52.5200, "longitude": 13.4050, "country": "Germany"},
    "Madrid": {"latitude": 40.4168, "longitude": -3.7038, "country": "Spain"},
    "Rome": {"latitude": 41.9028, "longitude": 12.4964, "country": "Italy"},
    "Amsterdam": {"latitude": 52.3676, "longitude": 4.9041, "country": "Netherlands"},
    "Brussels": {"latitude": 50.8503, "longitude": 4.3517, "country": "Belgium"},
    "Vienna": {"latitude": 48.2082, "longitude": 16.3738, "country": "Austria"},
    "Zurich": {"latitude": 47.3769, "longitude": 8.5417, "country": "Switzerland"},
    "Lisbon": {"latitude": 38.7223, "longitude": -9.1393, "country": "Portugal"},
    "Milan": {"latitude": 45.4642, "longitude": 9.1900, "country": "Italy"},
    
    # 北欧
    "Stockholm": {"latitude": 59.3293, "longitude": 18.0686, "country": "Sweden"},
    "Oslo": {"latitude": 59.9139, "longitude": 10.7522, "country": "Norway"},
    "Copenhagen": {"latitude": 55.6761, "longitude": 12.5683, "country": "Denmark"},
    "Helsinki": {"latitude": 60.1699, "longitude": 24.9384, "country": "Finland"},
    "Reykjavik": {"latitude": 64.1466, "longitude": -21.9426, "country": "Iceland"},
    
    # 东欧
    "Warsaw": {"latitude": 52.2297, "longitude": 21.0122, "country": "Poland"},
    "Prague": {"latitude": 50.0755, "longitude": 14.4378, "country": "Czech"},
    "Budapest": {"latitude": 47.4979, "longitude": 19.0402, "country": "Hungary"},
    "Bucharest": {"latitude": 44.4268, "longitude": 26.1025, "country": "Romania"},
    "Sofia": {"latitude": 42.6977, "longitude": 23.3219, "country": "Bulgaria"},
    "Athens": {"latitude": 37.9838, "longitude": 23.7275, "country": "Greece"},
    
    # 南欧
    "Barcelona": {"latitude": 41.3851, "longitude": 2.1734, "country": "Spain"},
    "Valencia": {"latitude": 39.4699, "longitude": -0.3763, "country": "Spain"},
    "Seville": {"latitude": 37.3891, "longitude": -5.9845, "country": "Spain"},
    "Naples": {"latitude": 40.8518, "longitude": 14.2681, "country": "Italy"},
    "Turin": {"latitude": 45.0703, "longitude": 7.6869, "country": "Italy"},
    "Florence": {"latitude": 43.7696, "longitude": 11.2558, "country": "Italy"},
    
    # 更多西欧城市
    "Munich": {"latitude": 48.1351, "longitude": 11.5820, "country": "Germany"},
    "Hamburg": {"latitude": 53.5511, "longitude": 9.9937, "country": "Germany"},
    "Frankfurt": {"latitude": 50.1109, "longitude": 8.6821, "country": "Germany"},
    "Cologne": {"latitude": 50.9375, "longitude": 6.9603, "country": "Germany"},
    "Dublin": {"latitude": 53.3498, "longitude": -6.2603, "country": "Ireland"},
    "Edinburgh": {"latitude": 55.9533, "longitude": -3.1883, "country": "UK"},
    "Manchester": {"latitude": 53.4808, "longitude": -2.2426, "country": "UK"},
    "Birmingham": {"latitude": 52.4862, "longitude": -1.8904, "country": "UK"},
    
    # 法国城市
    "Lyon": {"latitude": 45.7640, "longitude": 4.8357, "country": "France"},
    "Marseille": {"latitude": 43.2965, "longitude": 5.3698, "country": "France"},
    "Toulouse": {"latitude": 43.6047, "longitude": 1.4442, "country": "France"},
    "Nice": {"latitude": 43.7102, "longitude": 7.2620, "country": "France"},
    
    # 波罗的海国家
    "Tallinn": {"latitude": 59.4370, "longitude": 24.7536, "country": "Estonia"},
    "Riga": {"latitude": 56.9496, "longitude": 24.1052, "country": "Latvia"},
    "Vilnius": {"latitude": 54.6872, "longitude": 25.2797, "country": "Lithuania"},
    
    # 巴尔干半岛
    "Belgrade": {"latitude": 44.7866, "longitude": 20.4489, "country": "Serbia"},
    "Zagreb": {"latitude": 45.8150, "longitude": 15.9819, "country": "Croatia"},
    "Ljubljana": {"latitude": 46.0569, "longitude": 14.5058, "country": "Slovenia"},
    
    # 其他重要城市
    "Porto": {"latitude": 41.1579, "longitude": -8.6291, "country": "Portugal"},
    "Geneva": {"latitude": 46.2044, "longitude": 6.1432, "country": "Switzerland"},
    "Basel": {"latitude": 47.5596, "longitude": 7.5886, "country": "Switzerland"},
    "Luxembourg": {"latitude": 49.6116, "longitude": 6.1319, "country": "Luxembourg"}
}

# 时间范围 - 25年（2000-2024）
START_DATE = "2000-01-01"
END_DATE = "2024-12-31"

# 大幅扩展天气变量 - 38个小时级变量
HOURLY_VARIABLES = [
    # 温度相关
    "temperature_2m",
    "apparent_temperature",
    "dew_point_2m",
    
    # 湿度和气压
    "relative_humidity_2m",
    "pressure_msl",
    "surface_pressure",
    "vapour_pressure_deficit",
    
    # 降水
    "precipitation",
    "rain",
    "snowfall",
    "snow_depth",
    
    # 云量
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    
    # 风
    "wind_speed_10m",
    "wind_speed_100m",
    "wind_direction_10m",
    "wind_direction_100m",
    "wind_gusts_10m",
    
    # 太阳辐射
    "shortwave_radiation",
    "direct_radiation",
    "direct_normal_irradiance",
    "diffuse_radiation",
    "sunshine_duration",
    
    # 土壤温度
    "soil_temperature_0_to_7cm",
    "soil_temperature_7_to_28cm",
    "soil_temperature_28_to_100cm",
    "soil_temperature_100_to_255cm",
    
    # 土壤湿度
    "soil_moisture_0_to_7cm",
    "soil_moisture_7_to_28cm",
    "soil_moisture_28_to_100cm",
    "soil_moisture_100_to_255cm",
    
    # 其他
    "et0_fao_evapotranspiration",
    "weather_code"
]

# 每日聚合数据 - 19个变量
DAILY_VARIABLES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "apparent_temperature_max",
    "apparent_temperature_min",
    "apparent_temperature_mean",
    "precipitation_sum",
    "rain_sum",
    "snowfall_sum",
    "precipitation_hours",
    "sunrise",
    "sunset",
    "sunshine_duration",
    "daylight_duration",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "wind_direction_10m_dominant",
    "shortwave_radiation_sum",
    "et0_fao_evapotranspiration"
]

# API配置
BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

# 输出目录
OUTPUT_DIR = "../"  # 输出到data目录

# 请求配置 - 避免429错误
REQUEST_DELAY = 3  # 每次请求间隔3秒（重要！）
MAX_RETRIES = 5  # 最大重试次数
RETRY_DELAY = 15  # 重试间隔15秒
TIMEOUT = 120  # 请求超时时间


def fetch_weather_data(city_name, latitude, longitude, country):
    """
    获取单个城市的天气数据，包含重试机制
    """
    print(f"正在获取 {city_name} ({country}) 的数据...")
    
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": ",".join(HOURLY_VARIABLES),
        "daily": ",".join(DAILY_VARIABLES),
        "timezone": "GMT"
    }
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(BASE_URL, params=params, timeout=TIMEOUT)
            
            # 处理429错误（Too Many Requests）
            if response.status_code == 429:
                wait_time = RETRY_DELAY * (attempt + 1)
                print(f"  ⚠ 遇到限流(429)，等待 {wait_time} 秒后重试... (尝试 {attempt + 1}/{MAX_RETRIES})")
                time.sleep(wait_time)
                continue
            
            response.raise_for_status()
            data = response.json()
            
            # 处理hourly数据
            hourly_df = None
            if "hourly" in data and data["hourly"]:
                hourly_df = pd.DataFrame(data["hourly"])
                hourly_df.insert(0, "city", city_name)
                hourly_df.insert(1, "country", country)
                hourly_df.insert(2, "latitude", latitude)
                hourly_df.insert(3, "longitude", longitude)
                hourly_df.insert(4, "data_type", "hourly")
            
            # 处理daily数据
            daily_df = None
            if "daily" in data and data["daily"]:
                daily_df = pd.DataFrame(data["daily"])
                daily_df.insert(0, "city", city_name)
                daily_df.insert(1, "country", country)
                daily_df.insert(2, "latitude", latitude)
                daily_df.insert(3, "longitude", longitude)
                daily_df.insert(4, "data_type", "daily")
            
            hourly_count = len(hourly_df) if hourly_df is not None else 0
            daily_count = len(daily_df) if daily_df is not None else 0
            
            print(f"  ✓ {city_name} 数据获取成功 (hourly: {hourly_count:,}, daily: {daily_count:,})")
            return hourly_df, daily_df
            
        except requests.exceptions.Timeout:
            print(f"  ✗ 请求超时 (尝试 {attempt + 1}/{MAX_RETRIES})")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
        except requests.exceptions.RequestException as e:
            print(f"  ✗ 请求失败: {e} (尝试 {attempt + 1}/{MAX_RETRIES})")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
        except Exception as e:
            print(f"  ✗ 数据处理失败: {e}")
            break
    
    print(f"  ✗ {city_name} 数据获取失败（已重试{MAX_RETRIES}次）")
    return None, None


def save_checkpoint(completed_cities, checkpoint_file):
    """保存进度检查点"""
    with open(checkpoint_file, 'w') as f:
        json.dump(list(completed_cities), f)  # 转换为list


def load_checkpoint(checkpoint_file):
    """加载进度检查点"""
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                content = f.read().strip()
                if content:  # 检查文件是否为空
                    return set(json.loads(content))
        except (json.JSONDecodeError, ValueError) as e:
            print(f"⚠ 检查点文件损坏，将重新开始: {e}")
    return set()


def main():
    """
    主函数: 获取所有城市数据并保存为CSV
    """
    print("=" * 80)
    print("欧洲城市天气数据获取工具 - 增强版")
    print("=" * 80)
    print(f"时间范围: {START_DATE} 到 {END_DATE} (25年)")
    print(f"城市数量: {len(CITIES)} 个欧洲主要城市")
    print(f"小时级变量: {len(HOURLY_VARIABLES)} 个")
    print(f"每日级变量: {len(DAILY_VARIABLES)} 个")
    print(f"预计数据量: ~1GB")
    print(f"请求间隔: {REQUEST_DELAY} 秒（避免429错误）")
    print("=" * 80)
    print()
    
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 检查点文件
    checkpoint_file = os.path.join(OUTPUT_DIR, "progress_checkpoint.json")
    completed_cities = load_checkpoint(checkpoint_file)
    
    if completed_cities:
        print(f"📌 检测到断点续传：已完成 {len(completed_cities)} 个城市")
        print()
    
    hourly_data_list = []
    daily_data_list = []
    start_time = datetime.now()
    
    # 遍历所有城市
    for i, (city_name, city_info) in enumerate(CITIES.items(), 1):
        # 跳过已完成的城市
        if city_name in completed_cities:
            print(f"[{i}/{len(CITIES)}] ⏭ {city_name} 已完成，跳过")
            continue
        
        print(f"[{i}/{len(CITIES)}] ", end="")
        
        hourly_df, daily_df = fetch_weather_data(
            city_name,
            city_info["latitude"],
            city_info["longitude"],
            city_info["country"]
        )
        
        if hourly_df is not None:
            hourly_data_list.append(hourly_df)
        if daily_df is not None:
            daily_data_list.append(daily_df)
        
        # 标记为已完成
        completed_cities.add(city_name)
        save_checkpoint(completed_cities, checkpoint_file)
        
        # 添加延迟以避免API限流（重要！）
        if i < len(CITIES):
            print(f"  💤 等待 {REQUEST_DELAY} 秒...")
            time.sleep(REQUEST_DELAY)
        
        print()
    
    # 合并并保存数据
    if hourly_data_list or daily_data_list:
        print("=" * 80)
        print("正在合并和保存数据...")
        print()
        
        # 保存hourly数据
        if hourly_data_list:
            print("📊 合并小时级数据...")
            hourly_combined = pd.concat(hourly_data_list, ignore_index=True)
            hourly_file = os.path.join(OUTPUT_DIR, f"weather_hourly_{START_DATE}_to_{END_DATE}.csv")
            hourly_combined.to_csv(hourly_file, index=False, encoding="utf-8")
            hourly_size = os.path.getsize(hourly_file) / (1024*1024)
            print(f"  ✓ 小时级数据已保存: {hourly_file}")
            print(f"  ✓ 记录数: {len(hourly_combined):,}")
            print(f"  ✓ 文件大小: {hourly_size:.2f} MB")
            print()
        
        # 保存daily数据
        if daily_data_list:
            print("📊 合并每日数据...")
            daily_combined = pd.concat(daily_data_list, ignore_index=True)
            daily_file = os.path.join(OUTPUT_DIR, f"weather_daily_{START_DATE}_to_{END_DATE}.csv")
            daily_combined.to_csv(daily_file, index=False, encoding="utf-8")
            daily_size = os.path.getsize(daily_file) / (1024*1024)
            print(f"  ✓ 每日数据已保存: {daily_file}")
            print(f"  ✓ 记录数: {len(daily_combined):,}")
            print(f"  ✓ 文件大小: {daily_size:.2f} MB")
            print()
        
        # 统计信息
        if hourly_data_list:
            print("📈 数据统计:")
            print(f"  - 成功获取城市: {len(completed_cities)}/{len(CITIES)}")
            print(f"  - 时间跨度: 25年 (2000-2024)")
            print(f"  - 小时级变量: {len(HOURLY_VARIABLES)} 个")
            print(f"  - 每日级变量: {len(DAILY_VARIABLES)} 个")
            if hourly_data_list and daily_data_list:
                total_size = hourly_size + daily_size
                print(f"  - 总文件大小: {total_size:.2f} MB")
            print()
        
        # 删除检查点文件
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
        
        elapsed = datetime.now() - start_time
        print("=" * 80)
        print(f"✅ 数据获取完成！总耗时: {elapsed}")
        print("=" * 80)
        
    else:
        print("✗ 没有成功获取任何数据")


if __name__ == "__main__":
    main()
