"""
时间特征提取模块（小时数据版本）
功能：
1. 提取基础时间特征（年、月、日、星期、小时等）
2. 生成周期性编码（sin/cos）- 包含小时周期
3. 提取季节特征
"""
import pandas as pd
import numpy as np
from typing import List


class TimeFeatureExtractor:
    """时间特征提取器（小时数据适配版）"""

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
        """拟合（时间特征提取不需要拟合，保持接口一致性）"""
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """转换：提取时间特征（含小时周期）"""
        df = df.copy()

        # 确保时间列是 datetime 类型
        if not pd.api.types.is_datetime64_any_dtype(df[self.time_column]):
            df[self.time_column] = pd.to_datetime(df[self.time_column])

        # 提取基础时间特征
        df['year'] = df[self.time_column].dt.year
        df['month'] = df[self.time_column].dt.month
        df['day'] = df[self.time_column].dt.day
        df['hour'] = df[self.time_column].dt.hour           # ★ 小时（0-23）
        df['day_of_year'] = df[self.time_column].dt.dayofyear
        df['day_of_week'] = df[self.time_column].dt.dayofweek
        df['quarter'] = df[self.time_column].dt.quarter
        df['week_of_year'] = df[self.time_column].dt.isocalendar().week.astype(int)

        # 是否周末
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

        # 时段分类（早晨/上午/下午/傍晚/夜间）
        df['day_period'] = df['hour'].apply(self._get_day_period)

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

            # ★ 小时周期编码 (24小时) - 小时数据特有能力
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """拟合并转换"""
        return self.fit(df).transform(df)

    @staticmethod
    def _get_season(month: int) -> str:
        """根据月份获取季节（北半球）"""
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'autumn'

    @staticmethod
    def _get_day_period(hour: int) -> str:
        """根据小时获取时段"""
        if 5 <= hour < 8:
            return 'morning'      # 早晨 5-7
        elif 8 <= hour < 12:
            return 'forenoon'     # 上午 8-11
        elif 12 <= hour < 17:
            return 'afternoon'    # 下午 12-16
        elif 17 <= hour < 20:
            return 'evening'      # 傍晚 17-19
        else:
            return 'night'        # 夜间 20-4

    def get_feature_names(self) -> List[str]:
        """获取生成的特征名列表"""
        base_features = [
            'year', 'month', 'day', 'hour', 'day_of_year',
            'day_of_week', 'quarter', 'week_of_year',
            'is_weekend', 'day_period', 'season'
        ]

        if self.use_cyclical:
            cyclical_features = [
                'month_sin', 'month_cos',
                'day_of_year_sin', 'day_of_year_cos',
                'day_of_week_sin', 'day_of_week_cos',
                'hour_sin', 'hour_cos'            # ★ 新增
            ]
            return base_features + cyclical_features

        return base_features


if __name__ == "__main__":
    # 测试代码
    test_df = pd.DataFrame({
        'time': pd.date_range('2024-01-01', periods=24, freq='h')
    })

    extractor = TimeFeatureExtractor(use_cyclical=True)
    result = extractor.fit_transform(test_df)

    print("原始数据:")
    print(test_df.head())
    print("\n提取的时间特征:")
    print(result[['time', 'hour', 'hour_sin', 'hour_cos', 'day_period', 'is_weekend']].head(10))
    print("\n特征名列表:")
    print(extractor.get_feature_names())
