"""
天气数据集类
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
    天气时序预测数据集
    
    特性：
    1. 滑动窗口时序序列
    2. 按城市分组（保证时序完整性）
    3. 特征分组返回（categorical, numerical, cyclical, binary）
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
        max_samples: int = None,    # 限制最大样本数（调试/加速用）
    ):
        """
        初始化天气时序数据集
        
        Args:
            data_path: 数据文件路径
            seq_length: 序列长度（使用过去多少天）
            pred_horizon: 预测间隔（预测未来第几天）
            target_columns: 目标变量列名列表
            target_type: 任务类型 ('regression' 或 'classification')
            use_cache: 是否使用缓存
            cache_dir: 缓存目录
            group_by_city: 是否按城市分组
            include_current: 是否在序列中包含当前天
            transform: 数据增强转换函数（可选）
        """
        # 使用配置文件的默认值
        self.data_path = Path(data_path)
        self.seq_length = seq_length or config.SEQ_LENGTH
        self.pred_horizon = pred_horizon or config.PRED_HORIZON
        self.target_columns = target_columns or config.TARGET_COLUMNS
        self.target_type = target_type or config.TARGET_TYPE
        self.use_cache = use_cache if use_cache is not None else config.USE_CACHE
        self.cache_dir = Path(cache_dir) if cache_dir else config.CACHE_DIR
        self.group_by_city = group_by_city if group_by_city is not None else config.GROUP_BY_CITY
        self.include_current = include_current if include_current is not None else config.INCLUDE_CURRENT_HOUR
        self.max_samples = max_samples
        self.transform = transform
        
        # 获取特征配置
        self.feature_groups_config = feature_config.get_feature_groups()
        
        # 内存缓存
        self._memory_cache = None
        
        # 加载和预处理数据
        self._load_and_prepare_data()
    
    def _load_and_prepare_data(self):
        """懒加载模式：只保留 DataFrame，按需动态切片（避免 OOM）"""
        cache_config = {
            'seq_length': self.seq_length,
            'pred_horizon': self.pred_horizon,
            'target_columns': tuple(self.target_columns),
            'group_by_city': self.group_by_city,
        }

        # 磁盘缓存（仅在启用缓存时尝试）
        if self.use_cache and config.CACHE_LEVEL in ['disk', 'both']:
            cache_path = utils.get_cache_path(
                self.cache_dir, self.data_path, cache_config
            )
            cached = utils.load_cache(cache_path)
            if cached is not None:
                self._load_cache_lazy(cached)
                return

        # 读取 CSV
        print(f"\n正在加载数据: {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        print(f"✓ 加载了 {len(self.df):,} 行数据")

        # 构建懒加载索引：group -> [start_indices]（以城市为粒度）
        city_ids = self.df['city_id'].values if 'city_id' in self.df.columns else None

        total = len(self.df)
        max_start = total - self.seq_length - self.pred_horizon + 1
        if self.max_samples is not None and max_start > self.max_samples:
            max_start = self.max_samples
            self.df = self.df.iloc[:max_start + self.seq_length + self.pred_horizon - 1]

        self._n_samples = max_start
        print(f"✓ 懒加载模式: {self._n_samples:,} 个样本（按需切片，无需预存张量）")

        # 预提取 numpy 数组（避免 __getitem__ 时反复 iloc）
        self._prepare_arrays()

        # 保存磁盘缓存（可选）
        if self.use_cache and config.CACHE_LEVEL in ['disk', 'both']:
            cache_data = {
                'df': self.df,
                'n_samples': self._n_samples,
                'seq_length': self.seq_length,
                'pred_horizon': self.pred_horizon,
                'target_columns': self.target_columns,
            }
            utils.save_cache(cache_data, cache_path)

    def _prepare_arrays(self):
        """预提取各组 numpy 数组（按列存储，切片时无需 iloc）"""
        from . import feature_config as fc
        feature_groups_dict = self.feature_groups_config

        self._arrays = {}
        for group_name, columns in feature_groups_dict.items():
            if group_name == 'ignored':
                continue
            available = [c for c in columns if c in self.df.columns]
            if not available:
                # 空组，存 dummy
                self._arrays[group_name] = np.zeros((len(self.df), 0), dtype=np.float32)
                continue

            if group_name == 'season':
                vals = self.df['season'].map(fc.SEASON_MAPPING).values.astype(np.float32)
                self._arrays[group_name] = vals.reshape(-1, 1)
            elif group_name == 'day_period':
                vals = self.df['day_period'].map(fc.DAY_PERIOD_MAPPING).values.astype(np.float32)
                self._arrays[group_name] = vals.reshape(-1, 1)
            else:
                dtype = np.int64 if group_name == 'categorical' else np.float32
                self._arrays[group_name] = self.df[available].values.astype(dtype)

        # 目标值数组
        self._target_array = self.df[self.target_columns].values.astype(np.float32)

    def _load_cache_lazy(self, cached_data: dict):
        """从缓存恢复懒加载状态"""
        self.df = cached_data['df']
        self._n_samples = cached_data['n_samples']
        self.seq_length = cached_data['seq_length']
        self.pred_horizon = cached_data['pred_horizon']
        self.target_columns = cached_data['target_columns']
        self._prepare_arrays()
        print(f"✓ 从缓存加载（懒加载模式）: {self._n_samples:,} 样本")
    
    def _create_simple_sequences(
        self,
        feature_groups: Dict[str, np.ndarray],
        target_values: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        向量化滑动窗口（numpy stride tricks，C 级别速度）
        """
        total = len(target_values)
        max_start = total - self.seq_length - self.pred_horizon + 1

        if self.max_samples is not None and max_start > self.max_samples:
            max_start = self.max_samples
            total = max_start + self.seq_length + self.pred_horizon - 1
            # 截断到 max_samples
            for k in feature_groups:
                feature_groups[k] = feature_groups[k][:total]
            target_values = target_values[:total]

        print(f"创建 {max_start:,} 个滑动窗口序列（向量化）...")

        # 目标值：从 seq_length 位置开始取 pred_horizon 后的值
        target_idx_start = self.seq_length + self.pred_horizon - 1
        targets = target_values[target_idx_start : target_idx_start + max_start]
        if targets.ndim == 1:
            targets = targets.reshape(-1, 1)

        # 特征序列：用 sliding_window_view 一次性生成所有窗口
        from numpy.lib.stride_tricks import sliding_window_view
        sequences = {}
        for group_name, group_data in feature_groups.items():
            if group_data.shape[0] == 0:
                sequences[group_name] = np.array([])
                continue
            # sliding_window_view(axis=0) on (N, F) → (N-w+1, F, w)
            # 需要转置为 (N-w+1, w, F)
            windows = sliding_window_view(group_data, self.seq_length, axis=0)
            if windows.ndim == 3:
                windows = windows.transpose(0, 2, 1)  # (samples, seq_len, features)
            sequences[group_name] = windows[:max_start].copy()  # copy 使内存连续

        return sequences, targets
    
    def _load_from_cache(self, cached_data: dict):
        """从缓存加载数据"""
        self.sequences = cached_data['sequences']
        self.targets = cached_data['targets']
        self.seq_length = cached_data['seq_length']
        self.pred_horizon = cached_data['pred_horizon']
        self.target_columns = cached_data['target_columns']
        
        print(f"✓ 从缓存加载完成: {len(self)} 个样本")
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return self._n_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        懒加载：从预提取的 numpy 数组高速切片
        
        Args:
            idx: 样本索引
            
        Returns:
            包含特征和目标的字典
        """
        s = idx
        e = s + self.seq_length
        t = s + self.seq_length + self.pred_horizon - 1

        sample = {}
        for group_name, arr in self._arrays.items():
            if arr.shape[1] == 0:
                sample[group_name] = torch.empty((self.seq_length, 0), dtype=torch.float32)
                continue
            window = arr[s:e]
            if group_name == 'categorical':
                sample[group_name] = torch.from_numpy(window).long()
            else:
                sample[group_name] = torch.from_numpy(window).float()

        target_vals = self._target_array[t]
        sample['target'] = torch.from_numpy(target_vals).float()
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
    
    def get_feature_dims(self) -> Dict[str, int]:
        """
        获取各组特征的维度
        
        Returns:
            特征维度字典
        """
        dims = {}
        for group_name, group_tensor in self.sequences.items():
            if len(group_tensor.shape) == 3:  # (n_samples, seq_length, n_features)
                dims[group_name] = group_tensor.shape[2]
            elif len(group_tensor.shape) == 2:  # (n_samples, n_features)
                dims[group_name] = group_tensor.shape[1]
            else:
                dims[group_name] = 0
        return dims
    
    def get_num_targets(self) -> int:
        """
        获取目标变量的数量
        
        Returns:
            目标数量
        """
        if len(self.targets.shape) == 1:
            return 1
        else:
            return self.targets.shape[1]
    
    def get_sample_shape(self) -> dict:
        """
        获取样本的形状信息
        
        Returns:
            形状信息字典
        """
        sample = self[0]
        shapes = {}
        for key, value in sample.items():
            shapes[key] = value.shape
        return shapes
    
    def print_info(self):
        """打印数据集信息"""
        print("\n" + "="*60)
        print("数据集信息")
        print("="*60)
        print(f"数据路径: {self.data_path}")
        print(f"样本数量: {len(self):,}")
        print(f"序列长度: {self.seq_length}")
        print(f"预测间隔: {self.pred_horizon}")
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
        
        print("="*60 + "\n")



