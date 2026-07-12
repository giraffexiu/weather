# 异常值处理模块
import pandas as pd
import numpy as np

def detect_outliers(df, value_ranges):
    """检测异常值"""
    outliers = {}
    
    for col, (min_val, max_val) in value_ranges.items():
        if col not in df.columns:
            continue
        
        if pd.api.types.is_numeric_dtype(df[col]):
            mask = pd.Series([False] * len(df))
            
            if min_val is not None:
                mask |= df[col] < min_val
            if max_val is not None:
                mask |= df[col] > max_val
            
            count = mask.sum()
            if count > 0:
                outliers[col] = count
    
    return outliers

def clip_outliers(df, value_ranges):
    """裁剪异常值"""
    df = df.copy()
    total_clipped = 0
    
    for col, (min_val, max_val) in value_ranges.items():
        if col not in df.columns:
            continue
        
        if pd.api.types.is_numeric_dtype(df[col]):
            before = df[col].copy()
            df[col] = df[col].clip(lower=min_val, upper=max_val)
            clipped = (df[col] != before).sum()
            
            if clipped > 0:
                print(f"  {col}: 裁剪 {clipped} 个异常值")
                total_clipped += clipped
    
    if total_clipped > 0:
        print(f"总计裁剪: {total_clipped} 个异常值")
    
    return df

def fix_negative_values(df):
    """修正负值（仅限非负字段）"""
    df = df.copy()
    non_negative_fields = ['precipitation', 'precipitation_sum', 'rain', 'rain_sum', 
                           'snowfall', 'wind_speed', 'cloud_cover', 'shortwave_radiation']
    
    fixed = 0
    for col in df.columns:
        if any(field in col for field in non_negative_fields):
            if pd.api.types.is_numeric_dtype(df[col]):
                neg_count = (df[col] < 0).sum()
                if neg_count > 0:
                    df[col] = df[col].clip(lower=0)
                    fixed += neg_count
    
    if fixed > 0:
        print(f"修正负值: {fixed} 个")
    
    return df
