"""
天气数据集类（小时数据）
实现滑动窗口时序预测 + 缓存机制
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, Optional, Tuple

from . import config
from . import feature_config
from . import utils


class WeatherSequenceDataset(Dataset):
    """
    天气时序预测数据集（小时数据）

    特性：
    1. 滑动窗口时序序列（默认 24 小时窗口）
    2. 按城市分组或简单窗口模式
    3. 特征分组返回（categorical, numerical, cyclical, binary, season）
    4. 内存 + 磁盘缓存机制
    """

    def __init__(
        self,
        data_path: Path,
        seq_length: int = None,
        pred_horizon: int = None,
        target_columns: list = None,
        target_type: str = None,
        use_cache: bool = None,
        cache_dir: Path = None,
        group_by_city: bool = None,
        include_current: bool = None,
        transform: Optional[callable] = None,
        debug: bool = None
    ):
        self.data_path = Path(data_path)
        self.seq_length = seq_length or config.SEQ_LENGTH
        self.pred_horizon = pred_horizon or config.PRED_HORIZON
        self.target_columns = target_columns or config.TARGET_COLUMNS
        self.target_type = target_type or config.TARGET_TYPE
        self.use_cache = use_cache if use_cache is not None else config.USE_CACHE
        self.cache_dir = Path(cache_dir) if cache_dir else config.CACHE_DIR
        self.group_by_city = group_by_city if group_by_city is not None else config.GROUP_BY_CITY
        self.include_current = include_current if include_current is not None else config.INCLUDE_CURRENT_HOUR
        self.transform = transform
        self.debug = debug if debug is not None else config.DEBUG

        self.feature_groups_config = feature_config.get_feature_groups()
        self._memory_cache = None

        self._load_and_prepare_data()

    def _load_and_prepare_data(self):
        """加载和预处理数据（支持缓存）"""
        cache_config = {
            'seq_length': self.seq_length,
            'pred_horizon': self.pred_horizon,
            'target_columns': tuple(self.target_columns),
            'group_by_city': self.group_by_city,
            'include_current': self.include_current,
        }

        # 检查磁盘缓存
        if self.use_cache and config.CACHE_LEVEL in ['disk', 'both']:
            cache_path = utils.get_cache_path(
                self.cache_dir, self.data_path, cache_config)
            cached_data = utils.load_cache(cache_path)
            if cached_data is not None:
                self._load_from_cache(cached_data)
                return

        # 从头加载
        print(f"\n正在加载数据: {self.data_path}")
        if self.debug:
            self.df = pd.read_csv(self.data_path, nrows=config.DEBUG_SAMPLES)
            print(f"⚠ 调试模式: 仅加载 {len(self.df):,} 行")
        else:
            self.df = pd.read_csv(self.data_path)
        print(f"✓ 加载了 {len(self.df):,} 行数据")

        # 提取特征组
        print("\n提取特征组...")
        feature_groups = utils.extract_feature_groups(
            self.df, self.feature_groups_config)

        # 提取目标变量
        print(f"\n提取目标变量: {self.target_columns}")
        target_values = self.df[self.target_columns].values

        # 创建时序序列
        if self.group_by_city:
            print("\n按城市分组创建时序序列...")
            sequences, targets = utils.create_sequences_by_city(
                df=self.df,
                feature_groups=feature_groups,
                target_values=target_values,
                seq_length=self.seq_length,
                pred_horizon=self.pred_horizon,
                include_current=self.include_current
            )
        else:
            print("\n创建简单滑动窗口序列...")
            sequences, targets = utils.create_simple_sequences(
                feature_groups=feature_groups,
                target_values=target_values,
                seq_length=self.seq_length,
                pred_horizon=self.pred_horizon,
                include_current=self.include_current
            )

        # 转换为 PyTorch 张量
        print("\n转换为 PyTorch 张量...")
        self.sequences, self.targets = utils.numpy_to_torch(sequences, targets)

        print(f"✓ 数据集准备完成: {len(self)} 个样本")

        if self.debug:
            stats = utils.get_data_statistics(self.df, feature_groups, targets)
            utils.print_data_statistics(stats)

        # 保存到缓存
        if self.use_cache and config.CACHE_LEVEL in ['disk', 'both']:
            cache_data = {
                'sequences': self.sequences,
                'targets': self.targets,
                'seq_length': self.seq_length,
                'pred_horizon': self.pred_horizon,
                'target_columns': self.target_columns,
            }
            utils.save_cache(cache_data, cache_path)

        if self.use_cache and config.CACHE_LEVEL in ['memory', 'both']:
            self._memory_cache = {
                'sequences': self.sequences,
                'targets': self.targets,
            }

    def _load_from_cache(self, cached_data: dict):
        """从缓存加载数据"""
        self.sequences = cached_data['sequences']
        self.targets = cached_data['targets']
        self.seq_length = cached_data['seq_length']
        self.pred_horizon = cached_data['pred_horizon']
        self.target_columns = cached_data['target_columns']
        print(f"✓ 从缓存加载完成: {len(self)} 个样本")

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = {}
        for group_name, group_tensor in self.sequences.items():
            sample[group_name] = group_tensor[idx]
        sample['target'] = self.targets[idx]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def get_feature_dims(self) -> Dict[str, int]:
        dims = {}
        for group_name, group_tensor in self.sequences.items():
            if len(group_tensor.shape) == 3:
                dims[group_name] = group_tensor.shape[2]
            elif len(group_tensor.shape) == 2:
                dims[group_name] = group_tensor.shape[1]
            else:
                dims[group_name] = 0
        return dims

    def get_num_targets(self) -> int:
        if len(self.targets.shape) == 1:
            return 1
        return self.targets.shape[1]

    def get_sample_shape(self) -> dict:
        sample = self[0]
        shapes = {}
        for key, value in sample.items():
            shapes[key] = value.shape
        return shapes

    def print_info(self):
        print("\n" + "=" * 60)
        print("数据集信息（小时数据）")
        print("=" * 60)
        print(f"数据路径: {self.data_path}")
        print(f"样本数量: {len(self):,}")
        print(f"序列长度: {self.seq_length} 小时")
        print(f"预测间隔: {self.pred_horizon} 小时")
        print(f"目标变量: {self.target_columns}")
        print(f"任务类型: {self.target_type}")

        print(f"\n特征维度:")
        dims = self.get_feature_dims()
        for group_name, dim in dims.items():
            print(f"  {group_name}: {dim}")

        print(f"\n样本形状:")
        shapes = self.get_sample_shape()
        for key, shape in shapes.items():
            print(f"  {key}: {shape}")

        print("=" * 60 + "\n")
