"""
时间特征提取模块
功能：
1. 提取基础时间特征（年、月、日、星期等）
2. 生成周期性编码（sin/cos）
3. 提取季节特征
"""
import pandas as pd
import numpy as np
from typing import Dict, List


class TimeFeatureExtractor:
    """时间特征提取器"""
    
    def __init__(self, time_column: str = 'time', use_cyclical: bool = True):
        """
        初始化时间特征提取器
        
        Args:
            time_column: 时间列名
            use_cyclical: 是否使用周期性编码
        """
        self.time_column = time_column
        self.use_cyclical = use_cyclical
        
    def fit(self, df: pd.DataFrame) -> 'TimeFeatureExtractor':
        """
        拟合（时间特征提取不需要拟合，保持接口一致性）
        
        Args:
            df: 输入数据框
            
        Returns:
            self
        """
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        转换：提取时间特征
        
        Args:
            df: 输入数据框
            
        Returns:
            添加时间特征后的数据框
        """
        df = df.copy()
        
        # 确保时间列是 datetime 类型
        if not pd.api.types.is_datetime64_any_dtype(df[self.time_column]):
            df[self.time_column] = pd.to_datetime(df[self.time_column])
        
        # 提取基础时间特征
        df['year'] = df[self.time_column].dt.year
        df['month'] = df[self.time_column].dt.month
        df['day'] = df[self.time_column].dt.day
        df['day_of_year'] = df[self.time_column].dt.dayofyear
        df['day_of_week'] = df[self.time_column].dt.dayofweek  # 0=Monday, 6=Sunday
        df['quarter'] = df[self.time_column].dt.quarter
        df['week_of_year'] = df[self.time_column].dt.isocalendar().week.astype(int)
        
        # 季节特征
        df['season'] = df['month'].apply(self._get_season)
        
        # 周期性编码
        if self.use_cyclical:
            # 月份周期编码 (12个月)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            
            # 一年中的天数周期编码 (365天)
            df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
            df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
            
            # 星期几周期编码 (7天)
            df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        拟合并转换
        
        Args:
            df: 输入数据框
            
        Returns:
            添加时间特征后的数据框
        """
        return self.fit(df).transform(df)
    
    @staticmethod
    def _get_season(month: int) -> str:
        """
        根据月份获取季节（北半球季节）
        
        Args:
            month: 月份 (1-12)
            
        Returns:
            季节名称
        """
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:  # 9, 10, 11
            return 'autumn'
    
    def get_feature_names(self) -> List[str]:
        """
        获取生成的特征名列表
        
        Returns:
            特征名列表
        """
        base_features = [
            'year', 'month', 'day', 'day_of_year', 
            'day_of_week', 'quarter', 'week_of_year', 'season'
        ]
        
        if self.use_cyclical:
            cyclical_features = [
                'month_sin', 'month_cos',
                'day_of_year_sin', 'day_of_year_cos',
                'day_of_week_sin', 'day_of_week_cos'
            ]
            return base_features + cyclical_features
        
        return base_features


if __name__ == "__main__":
    # 测试代码
    test_df = pd.DataFrame({
        'time': pd.date_range('2024-01-01', periods=5, freq='D')
    })
    
    extractor = TimeFeatureExtractor(use_cyclical=True)
    result = extractor.fit_transform(test_df)
    
    print("原始数据:")
    print(test_df)
    print("\n提取的时间特征:")
    print(result)
    print("\n特征名列表:")
    print(extractor.get_feature_names())
