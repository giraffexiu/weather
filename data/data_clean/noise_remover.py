# 噪声列识别与移除模块
import pandas as pd
import re

def detect_noise_columns(df):
    """识别噪声列"""
    noise_cols = []
    
    for col in df.columns:
        # 检查乱码（非ASCII字符过多）
        if df[col].dtype == 'object':
            sample = df[col].astype(str).head(100)
            non_ascii = sum(1 for s in sample for c in s if ord(c) > 127)
            if non_ascii / max(sum(len(s) for s in sample), 1) > 0.3:
                noise_cols.append((col, '乱码'))
                continue
        
        # 检查单一值列（无信息）
        if df[col].nunique() == 1:
            noise_cols.append((col, '单一值'))
            continue
        
        # 检查空列
        if df[col].isnull().sum() == len(df):
            noise_cols.append((col, '全空'))
            continue
    
    return noise_cols

def remove_noise(df, keep_cols=None):
    """移除噪声列"""
    df = df.copy()
    noise = detect_noise_columns(df)
    
    if noise:
        print("\n检测到噪声列:")
        for col, reason in noise:
            print(f"  - {col}: {reason}")
        
        # 移除噪声列
        remove_cols = [col for col, _ in noise]
        if keep_cols:
            remove_cols = [c for c in remove_cols if c not in keep_cols]
        df = df.drop(columns=remove_cols, errors='ignore')
    
    # 移除重复列
    before = len(df.columns)
    df = df.loc[:, ~df.columns.duplicated()]
    if len(df.columns) < before:
        print(f"移除重复列: {before - len(df.columns)}个")
    
    return df
