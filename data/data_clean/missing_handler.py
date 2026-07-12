# 缺失值处理模块
import pandas as pd
import numpy as np

def analyze_missing(df, data_type='daily'):
    """分析缺失值"""
    missing = df.isnull().sum()
    missing_pct = 100 * missing / len(df)
    result = pd.DataFrame({'缺失数': missing, '缺失率%': missing_pct})
    result = result[result['缺失数'] > 0].sort_values('缺失率%', ascending=False)
    
    if len(result) > 0:
        print(f"\n{data_type}数据缺失值分析:")
        print(result.to_string())
    return result

def fill_missing(df, config):
    """填充缺失值"""
    df = df.copy()
    
    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue
            
        # 根据字段类型选择策略
        if 'temperature' in col:
            df[col] = df[col].interpolate(method='linear', limit_direction='both')
        elif any(x in col for x in ['precipitation', 'rain', 'snowfall']):
            df[col] = df[col].fillna(0)
        elif any(x in col for x in ['wind', 'pressure', 'humidity']):
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
        else:
            df[col] = df[col].interpolate(method='linear', limit_direction='both')
    
    # 删除仍有缺失的行（极少数）
    before = len(df)
    df = df.dropna()
    if before - len(df) > 0:
        print(f"删除无法填充的行数: {before - len(df)}")
    
    return df
