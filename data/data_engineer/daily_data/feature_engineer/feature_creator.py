"""
派生特征创建模块
功能：
1. 基于原始特征创建新的派生特征
2. 温度相关特征（日温差、结冰标志等）
3. 降水相关特征（降雪量、暴雨标志等）
4. 其他领域知识驱动的特征
"""
import pandas as pd
import numpy as np
from typing import List


class FeatureCreator:
    """派生特征创建器"""
    
    def __init__(self, create_all: bool = True):
        """
        初始化特征创建器
        
        Args:
            create_all: 是否创建所有派生特征
        """
        self.create_all = create_all
        self.created_features: List[str] = []
        
    def fit(self, df: pd.DataFrame) -> 'FeatureCreator':
        """
        拟合（特征创建不需要拟合，保持接口一致性）
        
        Args:
            df: 输入数据框
            
        Returns:
            self
        """
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        转换：创建派生特征
        
        Args:
            df: 输入数据框
            
        Returns:
            添加派生特征后的数据框
        """
        df = df.copy()
        self.created_features = []
        
        # 1. 温度相关特征
        if all(col in df.columns for col in ['temperature_2m_max', 'temperature_2m_min']):
            # 日温差
            df['temperature_range'] = df['temperature_2m_max'] - df['temperature_2m_min']
            self.created_features.append('temperature_range')
            
            # 是否结冰日（最低温度低于0度）
            df['is_freezing'] = (df['temperature_2m_min'] < 0).astype(int)
            self.created_features.append('is_freezing')
            
            # 是否高温日（最高温度超过30度）
            df['is_hot_day'] = (df['temperature_2m_max'] > 30).astype(int)
            self.created_features.append('is_hot_day')
        
        # 2. 降水相关特征
        if 'precipitation_sum' in df.columns:
            # 是否有降水
            df['is_rainy'] = (df['precipitation_sum'] > 0).astype(int)
            self.created_features.append('is_rainy')
            
            # 是否暴雨（降水量超过10mm）
            df['is_heavy_rain'] = (df['precipitation_sum'] > 10).astype(int)
            self.created_features.append('is_heavy_rain')
            
            # 降水等级分类（0: 无降水, 1: 小雨, 2: 中雨, 3: 大雨, 4: 暴雨）
            df['precipitation_level'] = pd.cut(
                df['precipitation_sum'],
                bins=[-np.inf, 0.1, 2.5, 10, 25, np.inf],
                labels=[0, 1, 2, 3, 4]
            ).astype(int)
            self.created_features.append('precipitation_level')
        
        # 3. 降雪量估算
        if all(col in df.columns for col in ['precipitation_sum', 'rain_sum']):
            # 降雪量 = 总降水 - 降雨（简化假设）
            df['snow_sum'] = df['precipitation_sum'] - df['rain_sum']
            df['snow_sum'] = df['snow_sum'].clip(lower=0)  # 确保非负
            self.created_features.append('snow_sum')
            
            # 是否下雪
            df['is_snowy'] = (df['snow_sum'] > 0).astype(int)
            self.created_features.append('is_snowy')
        
        # 4. 风速相关特征
        if 'wind_speed_10m_max' in df.columns:
            # 风力等级（蒲福风级简化版）
            # 0: <6, 1: 6-12, 2: 12-20, 3: 20-30, 4: 30-40, 5: >40
            df['wind_level'] = pd.cut(
                df['wind_speed_10m_max'],
                bins=[-np.inf, 6, 12, 20, 30, 40, np.inf],
                labels=[0, 1, 2, 3, 4, 5]
            ).astype(int)
            self.created_features.append('wind_level')
            
            # 是否大风（风速超过20 km/h）
            df['is_windy'] = (df['wind_speed_10m_max'] > 20).astype(int)
            self.created_features.append('is_windy')
        
        # 5. 辐射相关特征
        if 'shortwave_radiation_sum' in df.columns:
            # 辐射等级（0: 低, 1: 中, 2: 高）
            df['radiation_level'] = pd.cut(
                df['shortwave_radiation_sum'],
                bins=[-np.inf, 5, 15, np.inf],
                labels=[0, 1, 2]
            ).astype(int)
            self.created_features.append('radiation_level')
            
            # 是否晴天（高辐射通常表示晴天）
            df['is_sunny'] = (df['shortwave_radiation_sum'] > 15).astype(int)
            self.created_features.append('is_sunny')
        
        # 6. 综合天气状态
        if all(col in df.columns for col in ['is_rainy', 'is_freezing', 'is_windy']):
            # 恶劣天气标志（下雨 + 大风 或 结冰 + 降水）
            df['is_severe_weather'] = (
                ((df['is_rainy'] == 1) & (df['is_windy'] == 1)) |
                ((df['is_freezing'] == 1) & (df['is_rainy'] == 1))
            ).astype(int)
            self.created_features.append('is_severe_weather')
        
        # 7. 舒适度指数（简化版）
        if all(col in df.columns for col in ['temperature_2m_mean', 'wind_speed_10m_max']):
            # 体感温度（简化风寒指数）
            # Wind Chill = 13.12 + 0.6215*T - 11.37*V^0.16 + 0.3965*T*V^0.16
            # 这里使用简化版: 感温度 = 实际温度 - 风速影响
            df['feels_like_temperature'] = (
                df['temperature_2m_mean'] - 
                0.5 * np.sqrt(df['wind_speed_10m_max'])
            )
            self.created_features.append('feels_like_temperature')
        
        # 8. 地理位置相关特征
        if all(col in df.columns for col in ['latitude', 'longitude']):
            # 是否高纬度地区（纬度 > 55度，如北欧）
            df['is_high_latitude'] = (df['latitude'].abs() > 55).astype(int)
            self.created_features.append('is_high_latitude')
        
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        拟合并转换
        
        Args:
            df: 输入数据框
            
        Returns:
            添加派生特征后的数据框
        """
        return self.fit(df).transform(df)
    
    def get_feature_names(self) -> List[str]:
        """
        获取创建的特征名列表
        
        Returns:
            特征名列表
        """
        return self.created_features.copy()
    
    def get_feature_info(self) -> dict:
        """
        获取特征信息和说明
        
        Returns:
            特征名到说明的映射
        """
        feature_info = {
            'temperature_range': '日温差（最高温 - 最低温）',
            'is_freezing': '是否结冰日（最低温 < 0°C）',
            'is_hot_day': '是否高温日（最高温 > 30°C）',
            'is_rainy': '是否有降水',
            'is_heavy_rain': '是否暴雨（降水 > 10mm）',
            'precipitation_level': '降水等级（0-4）',
            'snow_sum': '降雪量估算（总降水 - 降雨）',
            'is_snowy': '是否下雪',
            'wind_level': '风力等级（0-5）',
            'is_windy': '是否大风（风速 > 20 km/h）',
            'radiation_level': '辐射等级（0-2）',
            'is_sunny': '是否晴天（高辐射）',
            'is_severe_weather': '是否恶劣天气',
            'feels_like_temperature': '体感温度',
            'is_high_latitude': '是否高纬度地区（|纬度| > 55°）'
        }
        
        return {feat: feature_info[feat] for feat in self.created_features if feat in feature_info}


if __name__ == "__main__":
    # 测试代码
    test_df = pd.DataFrame({
        'temperature_2m_max': [10, 25, 35, -5, 15],
        'temperature_2m_min': [-2, 18, 28, -10, 10],
        'temperature_2m_mean': [4, 21.5, 31.5, -7.5, 12.5],
        'precipitation_sum': [0, 5, 15, 2, 0],
        'rain_sum': [0, 5, 12, 0, 0],
        'wind_speed_10m_max': [10, 25, 15, 30, 8],
        'shortwave_radiation_sum': [8, 18, 20, 3, 12],
        'latitude': [48.8, 51.5, 40.4, 59.9, 45.4],
        'longitude': [2.3, -0.1, -3.7, 10.8, 4.4]
    })
    
    creator = FeatureCreator()
    result = creator.fit_transform(test_df)
    
    print("原始特征:")
    print(test_df)
    print("\n创建的派生特征:")
    print(result[creator.get_feature_names()])
    print("\n特征说明:")
    for feat, desc in creator.get_feature_info().items():
        print(f"  {feat}: {desc}")
