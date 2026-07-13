"""
工具函数模块（小时数据）
"""
import numpy as np
import pandas as pd
import torch
import pickle
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def set_seed(seed: int = 42):
    """设置随机种子以保证可复现性"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_cache_path(cache_dir: Path, data_path: Path, config_dict: dict) -> Path:
    """根据数据路径和配置生成唯一的缓存文件路径"""
    config_str = str(sorted(config_dict.items()))
    hash_obj = hashlib.md5(f"{data_path.name}_{config_str}".encode())
    hash_hex = hash_obj.hexdigest()[:12]
    cache_filename = f"{data_path.stem}_{hash_hex}.pkl"
    return cache_dir / cache_filename


def save_cache(data: any, cache_path: Path):
    """保存数据到缓存文件"""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"✓ 缓存已保存: {cache_path}")


def load_cache(cache_path: Path) -> Optional[any]:
    """从缓存文件加载数据"""
    if cache_path.exists():
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            print(f"✓ 从缓存加载: {cache_path}")
            return data
        except Exception as e:
            print(f"⚠ 缓存加载失败: {e}")
            return None
    return None


# day_period 字符串映射（小时数据独有）
DAY_PERIOD_MAPPING = {
    'night': 0,
    'morning': 1,
    'forenoon': 2,
    'afternoon': 3,
    'evening': 4,
}


def extract_feature_groups(
    df: pd.DataFrame,
    feature_config: dict
) -> Dict[str, np.ndarray]:
    """
    从 DataFrame 中提取分组特征

    特殊处理 season 和 day_period 字符串列 → 数值
    """
    from . import feature_config as fc

    feature_groups = {}

    for group_name, columns in feature_config.items():
        if group_name == 'ignored':
            continue

        available_cols = [col for col in columns if col in df.columns]
        missing_cols = set(columns) - set(available_cols)

        if missing_cols:
            print(f"⚠ 警告: {group_name} 中缺失列: {missing_cols}")

        if not available_cols:
            feature_groups[group_name] = np.array([]).reshape(len(df), 0)
            continue

        if group_name == 'season' and 'season' in available_cols:
            season_values = df['season'].map(fc.SEASON_MAPPING).values
            feature_groups[group_name] = season_values.reshape(-1, 1)
        elif group_name == 'numerical' and 'day_period' in available_cols:
            # day_period 是字符串，需要映射为数值
            df = df.copy()
            df['day_period'] = df['day_period'].map(DAY_PERIOD_MAPPING).fillna(0).astype(int)
            feature_groups[group_name] = df[available_cols].values
        else:
            feature_groups[group_name] = df[available_cols].values

    return feature_groups


def create_sequences_by_city(
    df: pd.DataFrame,
    feature_groups: Dict[str, np.ndarray],
    target_values: np.ndarray,
    seq_length: int,
    pred_horizon: int,
    include_current: bool = False
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """按城市分组创建时序序列"""
    sequences = {key: [] for key in feature_groups.keys()}
    targets = []

    if 'city_id' not in df.columns:
        raise ValueError("数据框中缺少 city_id 列")

    city_ids = df['city_id'].values
    unique_cities = np.unique(city_ids)

    print(f"\n正在为 {len(unique_cities)} 个城市创建时序序列...")

    for city_id in unique_cities:
        city_mask = (city_ids == city_id)
        city_indices = np.where(city_mask)[0]
        city_data_len = len(city_indices)

        if include_current:
            max_start = city_data_len - seq_length - pred_horizon + 1
        else:
            max_start = city_data_len - seq_length - pred_horizon + 1

        if max_start <= 0:
            continue

        for start_idx in range(max_start):
            if include_current:
                seq_indices = city_indices[start_idx:start_idx + seq_length]
                target_idx = city_indices[start_idx + seq_length + pred_horizon - 1]
            else:
                seq_indices = city_indices[start_idx:start_idx + seq_length]
                target_idx = city_indices[start_idx + seq_length + pred_horizon - 1]

            for group_name, group_data in feature_groups.items():
                sequences[group_name].append(group_data[seq_indices])
            targets.append(target_values[target_idx])

    for group_name in sequences:
        if len(sequences[group_name]) > 0:
            sequences[group_name] = np.array(sequences[group_name])
        else:
            sequences[group_name] = np.array([])

    targets = np.array(targets)
    print(f"✓ 创建了 {len(targets)} 个时序样本")

    return sequences, targets


def create_simple_sequences(
    feature_groups: Dict[str, np.ndarray],
    target_values: np.ndarray,
    seq_length: int,
    pred_horizon: int,
    include_current: bool = False
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    简单滑动窗口（不考虑城市边界）
    数据已按 city_id + time 排序，城市边界处会有少量跨城市窗口
    速度远快于按城市分组，适用于大数据量
    """
    sequences = {key: [] for key in feature_groups.keys()}
    targets = []

    total_samples = len(target_values)

    if include_current:
        max_start = total_samples - seq_length - pred_horizon + 1
    else:
        max_start = total_samples - seq_length - pred_horizon + 1

    print(f"创建 {max_start} 个滑动窗口序列（简单模式）...")

    for start_idx in range(max_start):
        if include_current:
            seq_indices = slice(start_idx, start_idx + seq_length)
            target_idx = start_idx + seq_length + pred_horizon - 1
        else:
            seq_indices = slice(start_idx, start_idx + seq_length)
            target_idx = start_idx + seq_length + pred_horizon - 1

        for group_name, group_data in feature_groups.items():
            sequences[group_name].append(group_data[seq_indices])
        targets.append(target_values[target_idx])

    for group_name in sequences:
        sequences[group_name] = np.array(sequences[group_name])
    targets = np.array(targets)

    print(f"✓ 创建了 {len(targets)} 个时序样本")

    return sequences, targets


def numpy_to_torch(
    sequences: Dict[str, np.ndarray],
    targets: np.ndarray,
    dtype: torch.dtype = torch.float32
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """将 numpy 数组转换为 PyTorch 张量"""
    torch_sequences = {}

    for group_name, group_data in sequences.items():
        if group_name == 'categorical':
            torch_sequences[group_name] = torch.from_numpy(group_data).long()
        else:
            torch_sequences[group_name] = torch.from_numpy(group_data).to(dtype)

    torch_targets = torch.from_numpy(targets).to(dtype)

    return torch_sequences, torch_targets


def get_data_statistics(
    df: pd.DataFrame,
    feature_groups: Dict[str, np.ndarray],
    targets: np.ndarray
) -> dict:
    """获取数据集统计信息"""
    stats = {
        'total_samples': len(df),
        'num_cities': df['city_id'].nunique() if 'city_id' in df.columns else 0,
        'time_range': f"{df['time'].min()} to {df['time'].max()}" if 'time' in df.columns else 'N/A',
        'feature_groups': {},
        'target_shape': targets.shape,
        'target_mean': targets.mean(axis=0) if len(targets) > 0 else None,
        'target_std': targets.std(axis=0) if len(targets) > 0 else None,
    }

    for group_name, group_data in feature_groups.items():
        stats['feature_groups'][group_name] = {
            'shape': group_data.shape,
            'num_features': group_data.shape[1] if len(group_data.shape) > 1 else 0
        }

    return stats


def print_data_statistics(stats: dict):
    """打印数据集统计信息"""
    print("\n" + "=" * 60)
    print("数据集统计信息（小时数据）")
    print("=" * 60)
    print(f"总样本数: {stats['total_samples']:,}")
    print(f"城市数量: {stats['num_cities']}")
    print(f"时间范围: {stats['time_range']}")

    print(f"\n特征组:")
    for group_name, group_stats in stats['feature_groups'].items():
        print(f"  {group_name}: {group_stats['shape']} ({group_stats['num_features']} 特征)")

    print(f"\n目标变量:")
    print(f"  形状: {stats['target_shape']}")
    if stats['target_mean'] is not None:
        print(f"  均值: {stats['target_mean']}")
        print(f"  标准差: {stats['target_std']}")

    print("=" * 60 + "\n")
