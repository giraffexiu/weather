"""
从原始数据重建标准化器（使用原始 statistics）
"""
import sys
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

_HERE = Path(__file__).parent

RAW_PATH = _HERE.parent.parent / "data" / "data_clean" / "cleaned_data" / "weather_hourly_cleaned.csv"
SCALER_OUT = _HERE.parent.parent / "data" / "data_engineer" / "hourly_data" / "processed_data" / "preprocessors" / "scaler.pkl"

TARGET_COLS = ['temperature_2m', 'precipitation', 'wind_speed_10m',
               'apparent_temperature', 'relative_humidity_2m']

df = pd.read_csv(RAW_PATH)
scaler = StandardScaler()
scaler.fit(df[TARGET_COLS])

with open(SCALER_OUT, 'wb') as f:
    pickle.dump(scaler, f)

print(f"标准拟合（原始数据 = {len(df):,} 行）:")
print(f"已保存: {SCALER_OUT}")
for c, m, s in zip(TARGET_COLS, scaler.mean_, scaler.scale_):
    print(f"  {c:<26} mean={m:10.4f}  std={s:10.4f}")
