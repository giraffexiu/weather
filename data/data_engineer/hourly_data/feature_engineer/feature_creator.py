"""
派生特征创建模块（小时数据版本）
功能：
1. 基于小时气象数据创建派生特征
2. 温度相关特征（风寒指数、酷暑标志等）
3. 降水/降雪相关特征
4. 风速风向相关特征
5. 综合天气状态特征
"""
import pandas as pd
import numpy as np
from typing import List


class FeatureCreator:
    """派生特征创建器（小时数据适配版）"""

    def __init__(self, create_all: bool = True):
        self.create_all = create_all
        self.created_features: List[str] = []

    def fit(self, df: pd.DataFrame) -> 'FeatureCreator':
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """转换：创建小时数据的派生特征"""
        df = df.copy()
        self.created_features = []

        # ===== 1. 温度相关特征 =====
        if 'temperature_2m' in df.columns:
            # 风寒指数（Wind Chill）- 简化的体感温度修正
            # Wind Chill = 13.12 + 0.6215*T - 11.37*V^0.16 + 0.3965*T*V^0.16
            if 'wind_speed_10m' in df.columns:
                T = df['temperature_2m']
                V = df['wind_speed_10m'].clip(lower=0.1)  # 避免除零
                V_pow = np.power(V, 0.16)
                df['wind_chill'] = np.where(
                    T <= 10,
                    13.12 + 0.6215 * T - 11.37 * V_pow + 0.3965 * T * V_pow,
                    T  # 温度高于10°C时风寒效应不明显
                )
                self.created_features.append('wind_chill')

            # 酷暑指数（Heat Index）- 简化的体感温度
            if 'relative_humidity_2m' in df.columns:
                T_c = df['temperature_2m']
                RH = df['relative_humidity_2m'].clip(0, 100)
                # 简化 Heat Index（仅当温度>27°C时有意义）
                df['heat_index'] = np.where(
                    T_c > 27,
                    -8.784695 + 1.61139411 * T_c + 2.338549 * RH
                    - 0.14611605 * T_c * RH - 0.01230809 * T_c**2
                    - 0.01642483 * RH**2 + 0.00221173 * T_c**2 * RH
                    + 0.00072546 * T_c * RH**2 - 0.00000358 * T_c**2 * RH**2,
                    T_c
                )
                self.created_features.append('heat_index')

            # 是否结冰
            df['is_freezing'] = (df['temperature_2m'] < 0).astype(int)
            self.created_features.append('is_freezing')

            # 是否高温
            df['is_hot'] = (df['temperature_2m'] > 35).astype(int)
            self.created_features.append('is_hot')

            # 温度等级（0:极寒, 1:冷, 2:凉爽, 3:舒适, 4:暖, 5:热, 6:酷暑）
            df['temperature_level'] = pd.cut(
                df['temperature_2m'],
                bins=[-np.inf, -10, 0, 10, 20, 30, 35, np.inf],
                labels=[0, 1, 2, 3, 4, 5, 6]
            ).astype(int)
            self.created_features.append('temperature_level')

        # ===== 2. 降水/降雪相关特征 =====
        if all(col in df.columns for col in ['precipitation', 'rain']):
            # 是否有降水
            df['is_rainy'] = (df['precipitation'] > 0).astype(int)
            self.created_features.append('is_rainy')

            # 降水强度等级
            df['precipitation_intensity'] = pd.cut(
                df['precipitation'],
                bins=[-np.inf, 0.01, 2.5, 7.6, 25, np.inf],
                labels=[0, 1, 2, 3, 4]
            ).astype(int)
            self.created_features.append('precipitation_intensity')

            # 是否为固态降水（降雪 = 总降水 - 降雨）
            df['solid_precip'] = (df['precipitation'] - df['rain']).clip(lower=0)
            self.created_features.append('solid_precip')

            # 是否下雪
            df['is_snowy'] = (df['solid_precip'] > 0).astype(int)
            self.created_features.append('is_snowy')

        if 'snowfall' in df.columns:
            df['is_snow'] = (df['snowfall'] > 0).astype(int)
            self.created_features.append('is_snow')

        # ===== 3. 风速风向相关特征 =====
        if 'wind_speed_10m' in df.columns:
            # 风力等级
            df['wind_level'] = pd.cut(
                df['wind_speed_10m'],
                bins=[-np.inf, 6, 12, 20, 30, 40, np.inf],
                labels=[0, 1, 2, 3, 4, 5]
            ).astype(int)
            self.created_features.append('wind_level')

            # 是否大风 / 强风
            df['is_windy'] = (df['wind_speed_10m'] > 20).astype(int)
            self.created_features.append('is_windy')

            df['is_strong_wind'] = (df['wind_speed_10m'] > 30).astype(int)
            self.created_features.append('is_strong_wind')

        # 阵风相关
        if 'wind_gusts_10m' in df.columns and 'wind_speed_10m' in df.columns:
            # 阵风与持续风速的差值（阵风因子）
            df['gust_factor'] = (df['wind_gusts_10m'] / df['wind_speed_10m'].clip(lower=0.1)).clip(upper=10)
            self.created_features.append('gust_factor')

        # 风向分量（北向和东向分量）
        if 'wind_direction_10m' in df.columns:
            wind_rad = np.deg2rad(df['wind_direction_10m'])
            df['wind_u'] = -df['wind_speed_10m'] * np.sin(wind_rad)  # u分量（东向正）
            df['wind_v'] = -df['wind_speed_10m'] * np.cos(wind_rad)  # v分量（北向正）
            self.created_features.extend(['wind_u', 'wind_v'])

        # ===== 4. 湿度/气压相关特征 =====
        if 'relative_humidity_2m' in df.columns:
            # 湿度等级
            df['humidity_level'] = pd.cut(
                df['relative_humidity_2m'],
                bins=[-np.inf, 30, 50, 70, 90, np.inf],
                labels=[0, 1, 2, 3, 4]
            ).astype(int)
            self.created_features.append('humidity_level')

            # 是否干燥/潮湿
            df['is_dry'] = (df['relative_humidity_2m'] < 30).astype(int)
            self.created_features.append('is_dry')
            df['is_humid'] = (df['relative_humidity_2m'] > 80).astype(int)
            self.created_features.append('is_humid')

        if 'pressure_msl' in df.columns:
            # 气压趋势类别
            df['pressure_level'] = pd.cut(
                df['pressure_msl'],
                bins=[-np.inf, 1000, 1010, 1020, 1030, np.inf],
                labels=[0, 1, 2, 3, 4]
            ).astype(int)
            self.created_features.append('pressure_level')

        # ===== 5. 云量/辐射相关特征 =====
        if 'cloud_cover' in df.columns:
            # 云量等级
            df['cloud_level'] = pd.cut(
                df['cloud_cover'],
                bins=[-np.inf, 10, 30, 60, 90, np.inf],
                labels=[0, 1, 2, 3, 4]
            ).astype(int)
            self.created_features.append('cloud_level')

            # 是否晴天/阴天
            df['is_clear'] = (df['cloud_cover'] < 10).astype(int)
            self.created_features.append('is_clear')
            df['is_overcast'] = (df['cloud_cover'] > 90).astype(int)
            self.created_features.append('is_overcast')

        # ===== 6. 综合天气状态 =====
        needed = ['is_rainy', 'is_snowy', 'is_windy', 'is_freezing', 'is_overcast']
        if all(col in df.columns for col in needed):
            # 恶劣天气指数（各项加和）
            df['severe_weather_index'] = (
                df['is_rainy'] + df['is_snowy'] + df['is_windy']
                + df['is_freezing'] + df['is_overcast']
            )
            self.created_features.append('severe_weather_index')

            # 是否恶劣天气
            df['is_severe_weather'] = (df['severe_weather_index'] >= 3).astype(int)
            self.created_features.append('is_severe_weather')

        # ===== 7. 地理位置相关特征 =====
        if 'latitude' in df.columns:
            df['is_high_latitude'] = (df['latitude'].abs() > 55).astype(int)
            self.created_features.append('is_high_latitude')

            # 是否地中海纬度（适合做区域聚类）
            df['is_mediterranean'] = (
                (df['latitude'].between(35, 45)) &
                (df['longitude'].between(-10, 35))
            ).astype(int)
            self.created_features.append('is_mediterranean')

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def get_feature_names(self) -> List[str]:
        return self.created_features.copy()

    def get_feature_info(self) -> dict:
        """获取特征说明"""
        feature_info = {
            'wind_chill': '风寒指数（体感温度修正）',
            'heat_index': '酷暑指数（高温高湿体感温度）',
            'is_freezing': '是否结冰（温度<0°C）',
            'is_hot': '是否高温（温度>35°C）',
            'temperature_level': '温度等级（0-6: 极寒→酷暑）',
            'is_rainy': '是否有降水',
            'precipitation_intensity': '降水强度等级（0-4）',
            'solid_precip': '固态降水量（降雪估算）',
            'is_snowy': '是否固态降水',
            'is_snow': '是否有降雪记录',
            'wind_level': '风力等级（0-5: 蒲福简化）',
            'is_windy': '是否大风（>20km/h）',
            'is_strong_wind': '是否强风（>30km/h）',
            'gust_factor': '阵风因子（阵风/持续风速比）',
            'wind_u': '风向u分量（东向正）',
            'wind_v': '风向v分量（北向正）',
            'humidity_level': '湿度等级（0-4: 干燥→极度潮湿）',
            'is_dry': '是否干燥（湿度<30%）',
            'is_humid': '是否潮湿（湿度>80%）',
            'pressure_level': '气压等级（0-4）',
            'cloud_level': '云量等级（0-4: 晴→阴）',
            'is_clear': '是否晴天（云量<10%）',
            'is_overcast': '是否阴天（云量>90%）',
            'severe_weather_index': '恶劣天气指数（累加）',
            'is_severe_weather': '是否恶劣天气',
            'is_high_latitude': '是否高纬度（|纬度|>55°）',
            'is_mediterranean': '是否地中海区域',
        }
        return {f: feature_info[f] for f in self.created_features if f in feature_info}
