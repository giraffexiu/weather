#!/usr/bin/env python3
# 主数据清洗脚本
import os
import pandas as pd
import glob
from config import *
from missing_handler import analyze_missing, fill_missing
from noise_remover import remove_noise
from outlier_handler import clip_outliers, fix_negative_values

def clean_data(data_path, data_type, output_file):
    """清洗数据主函数"""
    print(f"\n{'='*60}")
    print(f"开始处理{data_type}数据")
    print(f"{'='*60}")
    
    # 读取所有CSV文件
    files = glob.glob(os.path.join(data_path, "*.csv"))
    print(f"找到 {len(files)} 个数据文件")
    
    all_data = []
    for i, file in enumerate(files, 1):
        if i % 10 == 0:
            print(f"读取进度: {i}/{len(files)}")
        df = pd.read_csv(file)
        all_data.append(df)
    
    df = pd.concat(all_data, ignore_index=True)
    print(f"合并后数据: {len(df)} 行 x {len(df.columns)} 列")
    
    # 1. 噪声列移除
    print(f"\n[步骤1] 噪声列识别与移除")
    df = remove_noise(df)
    print(f"处理后: {len(df.columns)} 列")
    
    # 2. 缺失值处理
    print(f"\n[步骤2] 缺失值分析与处理")
    analyze_missing(df, data_type)
    df = fill_missing(df, None)
    print(f"处理后: {len(df)} 行")
    
    # 3. 异常值处理
    print(f"\n[步骤3] 异常值检测与处理")
    df = fix_negative_values(df)
    df = clip_outliers(df, VALUE_RANGES)
    
    # 保存清洗后数据
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    output_path = os.path.join(OUTPUT_PATH, output_file)
    df.to_csv(output_path, index=False)
    print(f"\n✓ 清洗完成，保存至: {output_path}")
    print(f"  最终数据: {len(df)} 行 x {len(df.columns)} 列")
    
    return df

def main():
    """主入口"""
    print("天气数据清洗系统")
    print("="*60)
    
    # 处理日数据
    daily_df = clean_data(
        DAILY_DATA_PATH,
        "日",
        "weather_daily_cleaned.csv"
    )
    
    # 处理小时数据
    hourly_df = clean_data(
        HOURLY_DATA_PATH,
        "小时",
        "weather_hourly_cleaned.csv"
    )
    
    print("\n" + "="*60)
    print("所有数据清洗完成！")
    print("="*60)
    print(f"日数据: {len(daily_df):,} 行")
    print(f"小时数据: {len(hourly_df):,} 行")

if __name__ == "__main__":
    main()
