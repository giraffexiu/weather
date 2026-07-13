"""
工具函数模块
"""
import numpy as np
import pandas as pd
import torch
import pickle
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def set_seed(seed: int = 42):
    """
    设置随机种子以保证可复现性
    
    Args:
        seed: 随机种子
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_cache_path(cache_dir: Path, data_path: Path, config_dict: dict) -> Path:
    """
    根据数据路径和配置生成唯一的缓存文件路径
    
    Args:
        cache_dir: 缓存目录
        data_path: 数据文件路径
        config_dict: 配置字典（影响缓存的参数）
        
    Returns:
        缓存文件路径
    """
    # 创建配置字符串
    config_str = str(sorted(config_dict.items()))
    
    # 生成哈希值
    hash_obj = hashlib.md5(f"{data_path.name}_{config_str}".encode())
    hash_hex = hash_obj.hexdigest()[:12]
    
    # 缓存文件名
    cache_filename = f"{data_path.stem}_{hash_hex}.pkl"
    
    return cache_dir / cache_filename


def save_cache(data: any, cache_path: Path):
    """
    保存数据到缓存文件
    
    Args:
        data: 要缓存的数据
        cache_path: 缓存文件路径
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"✓ 缓存已保存: {cache_path}")


def load_cache(cache_path: Path) -> Optional[any]:
    """
    从缓存文件加载数据
    
    Args:
        cache_path: 缓存文件路径
        
    Returns:
        缓存的数据，如果不存在则返回 None
    """
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


def extract_feature_groups(
    df: pd.DataFrame,
    feature_config: dict
) -> Dict[str, np.ndarray]:
    """
    从 DataFrame 中提取分组特征
    
    Args:
        df: 数据框
        feature_config: 特征配置字典
        
    Returns:
        特征组字典
    """
    from . import feature_config as fc
    
    feature_groups = {}
    
    for group_name, columns in feature_config.items():
        if group_name == 'ignored':
            continue
        
        # 检查列是否存在
        available_cols = [col for col in columns if col in df.columns]
        missing_cols = set(columns) - set(available_cols)
        
        if missing_cols:
            print(f"⚠ 警告: {group_name} 中缺失列: {missing_cols}")
        
        if available_cols:
            # 特殊处理 season 列（字符串转数值）
            if group_name == 'season' and 'season' in available_cols:
                season_values = df['season'].map(fc.SEASON_MAPPING).values
                feature_groups[group_name] = season_values.reshape(-1, 1)
            # 特殊处理 day_period 列（字符串转数值）
            elif group_name == 'day_period' and 'day_period' in available_cols:
                day_period_values = df['day_period'].map(fc.DAY_PERIOD_MAPPING).values
                feature_groups[group_name] = day_period_values.reshape(-1, 1)
            else:
                feature_groups[group_name] = df[available_cols].values
        else:
            # 如果该组没有可用列，创建空数组
            feature_groups[group_name] = np.array([]).reshape(len(df), 0)
    
    return feature_groups


def create_sequences_by_city(
    df: pd.DataFrame,
    feature_groups: Dict[str, np.ndarray],
    target_values: np.ndarray,
    seq_length: int,
    pred_horizon: int,
    include_current: bool = False
) -> Tuple[Dict[str, List], List]:
    """
    按城市分组创建时序序列
    
    Args:
        df: 原始数据框（需要有 city_id 列）
        feature_groups: 特征组字典
        target_values: 目标值数组
        seq_length: 序列长度
        pred_horizon: 预测间隔
        include_current: 是否包含当前天
        
    Returns:
        (序列特征字典, 目标值列表)
    """
    sequences = {key: [] for key in feature_groups.keys()}
    targets = []
    
    # 获取城市ID
    if 'city_id' in df.columns:
        city_ids = df['city_id'].values
    else:
        raise ValueError("数据框中缺少 city_id 列")
    
    # 按城市分组
    unique_cities = np.unique(city_ids)
    
    print(f"\n正在为 {len(unique_cities)} 个城市创建时序序列...")
    
    for city_id in unique_cities:
        # 获取该城市的所有数据索引
        city_mask = (city_ids == city_id)
        city_indices = np.where(city_mask)[0]
        
        # 确保数据按时间排序（假设 CSV 已排序）
        city_data_len = len(city_indices)
        
        # 创建滑动窗口
        if include_current:
            # 使用 [t-seq_length+1:t+1] 预测 t+pred_horizon
            max_start = city_data_len - seq_length - pred_horizon + 1
        else:
            # 使用 [t-seq_length:t] 预测 t+pred_horizon
            max_start = city_data_len - seq_length - pred_horizon + 1
        
        if max_start <= 0:
            continue  # 该城市数据不足，跳过
        
        for start_idx in range(max_start):
            if include_current:
                seq_indices = city_indices[start_idx:start_idx + seq_length]
                target_idx = city_indices[start_idx + seq_length + pred_horizon - 1]
            else:
                seq_indices = city_indices[start_idx:start_idx + seq_length]
                target_idx = city_indices[start_idx + seq_length + pred_horizon - 1]
            
            # 提取序列特征
            for group_name, group_data in feature_groups.items():
                sequences[group_name].append(group_data[seq_indices])
            
            # 提取目标值
            targets.append(target_values[target_idx])
    
    # 转换为 numpy 数组
    for group_name in sequences:
        if len(sequences[group_name]) > 0:
            sequences[group_name] = np.array(sequences[group_name])
        else:
            sequences[group_name] = np.array([])
    
    targets = np.array(targets)
    
    print(f"✓ 创建了 {len(targets)} 个时序样本")
    
    return sequences, targets


def numpy_to_torch(
    sequences: Dict[str, np.ndarray],
    targets: np.ndarray,
    dtype: torch.dtype = torch.float32
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """
    将 numpy 数组转换为 PyTorch 张量
    
    Args:
        sequences: 序列特征字典（numpy）
        targets: 目标值数组（numpy）
        dtype: 张量数据类型
        
    Returns:
        (序列特征字典（torch）, 目标值张量（torch）)
    """
    torch_sequences = {}
    
    for group_name, group_data in sequences.items():
        if group_name == 'categorical':
            # 类别特征使用 long 类型
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
    """
    获取数据集统计信息
    
    Args:
        df: 数据框
        feature_groups: 特征组字典
        targets: 目标值数组
        
    Returns:
        统计信息字典
    """
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
    """
    打印数据集统计信息
    
    Args:
        stats: 统计信息字典
    """
    print("\n" + "="*60)
    print("数据集统计信息")
    print("="*60)
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
    
    print("="*60 + "\n")



