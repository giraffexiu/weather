"""
模型预测后处理模块
- 反标准化（z-score → 真实物理单位）
- 物理约束（裁剪不合理值）
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional

SCALER_PATH = (
    Path(__file__).parent.parent.parent
    / "data" / "data_engineer" / "hourly_data"
    / "processed_data" / "preprocessors" / "scaler.pkl"
)

TARGET_COLS = [
    'temperature_2m', 'precipitation', 'wind_speed_10m',
    'apparent_temperature', 'relative_humidity_2m',
]


def _load_scaler():
    with open(SCALER_PATH, 'rb') as f:
        return pickle.load(f)


def inverse_transform(predictions: np.ndarray) -> np.ndarray:
    """
    将标准化空间的预测还原为真实物理单位
    
    Args:
        predictions: (N, 5) 标准化预测值
        
    Returns:
        (N, 5) 真实单位预测值
    """
    scaler = _load_scaler()
    return scaler.inverse_transform(predictions)


def apply_constraints(predictions: np.ndarray) -> np.ndarray:
    """
    施加物理约束
    
    Args:
        predictions: (N, 5) [temperature, precipitation, wind, apparent_temp, humidity]
        
    Returns:
        (N, 5) 约束后的预测
    """
    result = predictions.copy()
    # 降水 ≥ 0
    result[:, 1] = np.maximum(result[:, 1], 0.0)
    # 风速 ≥ 0
    result[:, 2] = np.maximum(result[:, 2], 0.0)
    # 湿度 [0, 100]
    result[:, 4] = np.clip(result[:, 4], 0.0, 100.0)
    return result


def post_process(predictions: np.ndarray) -> np.ndarray:
    """
    完整后处理：逆标准化 + 物理约束
    
    Args:
        predictions: (N, 5) 标准化预测值
        
    Returns:
        (N, 5) 真实物理单位预测值
    """
    real = inverse_transform(predictions)
    real = apply_constraints(real)
    return real


def post_process_dict(results: dict) -> dict:
    """
    对 model_wrapper.predict() 返回的字典做后处理
    """
    # 将 regression 列拼成矩阵
    cols_order = [
        'temperature_2m', 'precipitation', 'wind_speed_10m',
        'apparent_temperature', 'relative_humidity_2m',
    ]
    raw = np.column_stack([results['regression'][c] for c in cols_order])
    real = post_process(raw)

    # 更新字典
    for i, c in enumerate(cols_order):
        results['regression'][c] = real[:, i]
    return results


def get_scale_info() -> Dict[str, Dict[str, float]]:
    """返回各目标的标准化参数"""
    scaler = _load_scaler()
    info = {}
    for c, m, s in zip(TARGET_COLS, scaler.mean_, scaler.scale_):
        info[c] = {'mean': float(m), 'std': float(s)}
    return info


if __name__ == '__main__':
    info = get_scale_info()
    print("标准化参数:")
    for c, v in info.items():
        print(f"  {c}:  mean={v['mean']:.4f}, std={v['std']:.4f}")
