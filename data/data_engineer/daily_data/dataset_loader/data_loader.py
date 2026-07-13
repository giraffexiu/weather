"""
DataLoader 工厂函数
提供统一的接口创建训练和测试数据加载器
"""
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional, Dict

from . import config
from .weather_dataset import WeatherSequenceDataset
from .utils import set_seed


def get_dataloader(
    split: str = 'train',
    data_path: Optional[Path] = None,
    batch_size: Optional[int] = None,
    shuffle: Optional[bool] = None,
    num_workers: Optional[int] = None,
    pin_memory: Optional[bool] = None,
    seq_length: Optional[int] = None,
    pred_horizon: Optional[int] = None,
    target_columns: Optional[list] = None,
    target_type: Optional[str] = None,
    use_cache: Optional[bool] = None,
    cache_dir: Optional[Path] = None,
    group_by_city: Optional[bool] = None,
    include_current: Optional[bool] = None,
    transform: Optional[callable] = None,
    drop_last: bool = False,
    **kwargs
) -> DataLoader:
    """
    创建 DataLoader 的统一接口
    
    Args:
        split: 数据集类型 ('train' 或 'test')
        data_path: 数据文件路径（如果为 None，自动根据 split 选择）
        batch_size: 批次大小
        shuffle: 是否打乱数据
        num_workers: 数据加载线程数
        pin_memory: 是否固定内存
        seq_length: 序列长度
        pred_horizon: 预测间隔
        target_columns: 目标变量列名列表
        target_type: 任务类型
        use_cache: 是否使用缓存
        cache_dir: 缓存目录
        group_by_city: 是否按城市分组
        include_current: 是否包含当前天
        transform: 数据增强函数
        drop_last: 是否丢弃最后不完整的batch
        **kwargs: 其他传递给 DataLoader 的参数
        
    Returns:
        DataLoader 实例
    """
    # 设置随机种子
    set_seed(config.RANDOM_SEED)
    
    # 根据 split 自动选择数据路径
    if data_path is None:
        if split == 'train':
            data_path = config.TRAIN_DATA_PATH
        elif split == 'test':
            data_path = config.TEST_DATA_PATH
        else:
            raise ValueError(f"split 必须是 'train' 或 'test'，得到: {split}")
    
    # 使用配置文件的默认值
    batch_size = batch_size if batch_size is not None else config.BATCH_SIZE
    num_workers = num_workers if num_workers is not None else config.NUM_WORKERS
    pin_memory = pin_memory if pin_memory is not None else config.PIN_MEMORY
    
    # shuffle 的默认值根据 split 决定
    if shuffle is None:
        shuffle = config.SHUFFLE_TRAIN if split == 'train' else config.SHUFFLE_TEST
    
    # 创建数据集
    dataset = WeatherSequenceDataset(
        data_path=data_path,
        seq_length=seq_length,
        pred_horizon=pred_horizon,
        target_columns=target_columns,
        target_type=target_type,
        use_cache=use_cache,
        cache_dir=cache_dir,
        group_by_city=group_by_city,
        include_current=include_current,
        transform=transform
    )
    
    # 创建 DataLoader
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        **kwargs
    )
    
    return dataloader


def get_train_dataloader(
    batch_size: Optional[int] = None,
    shuffle: bool = True,
    **kwargs
) -> DataLoader:
    """
    快速创建训练集 DataLoader
    
    Args:
        batch_size: 批次大小
        shuffle: 是否打乱
        **kwargs: 其他参数
        
    Returns:
        训练集 DataLoader
    """
    return get_dataloader(
        split='train',
        batch_size=batch_size,
        shuffle=shuffle,
        **kwargs
    )


def get_test_dataloader(
    batch_size: Optional[int] = None,
    shuffle: bool = False,
    **kwargs
) -> DataLoader:
    """
    快速创建测试集 DataLoader
    
    Args:
        batch_size: 批次大小
        shuffle: 是否打乱（默认不打乱）
        **kwargs: 其他参数
        
    Returns:
        测试集 DataLoader
    """
    return get_dataloader(
        split='test',
        batch_size=batch_size,
        shuffle=shuffle,
        **kwargs
    )


def get_dataloaders(
    batch_size: Optional[int] = None,
    shuffle_train: bool = True,
    shuffle_test: bool = False,
    **kwargs
) -> Dict[str, DataLoader]:
    """
    同时创建训练集和测试集 DataLoader
    
    Args:
        batch_size: 批次大小
        shuffle_train: 是否打乱训练集
        shuffle_test: 是否打乱测试集
        **kwargs: 其他参数
        
    Returns:
        包含 'train' 和 'test' 键的字典
    """
    train_loader = get_train_dataloader(
        batch_size=batch_size,
        shuffle=shuffle_train,
        **kwargs
    )
    
    test_loader = get_test_dataloader(
        batch_size=batch_size,
        shuffle=shuffle_test,
        **kwargs
    )
    
    return {
        'train': train_loader,
        'test': test_loader
    }


def print_dataloader_info(dataloader: DataLoader, name: str = "DataLoader"):
    """
    打印 DataLoader 信息
    
    Args:
        dataloader: DataLoader 实例
        name: 名称
    """
    dataset = dataloader.dataset
    
    print("\n" + "="*60)
    print(f"{name} 信息")
    print("="*60)
    print(f"数据集大小: {len(dataset):,} 个样本")
    print(f"批次大小: {dataloader.batch_size}")
    print(f"批次数量: {len(dataloader):,}")
    print(f"是否打乱: {dataloader.shuffle if hasattr(dataloader, 'shuffle') else 'N/A'}")
    print(f"工作线程: {dataloader.num_workers}")
    print(f"固定内存: {dataloader.pin_memory}")
    
    # 获取一个样本查看形状
    sample = dataset[0]
    print(f"\n样本形状:")
    for key, value in sample.items():
        print(f"  {key}: {value.shape}, dtype={value.dtype}")
    
    print("="*60 + "\n")


class DataLoaderFactory:
    """
    DataLoader 工厂类（面向对象接口）
    """
    
    def __init__(
        self,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        pin_memory: Optional[bool] = None,
        **dataset_kwargs
    ):
        """
        初始化工厂
        
        Args:
            batch_size: 批次大小
            num_workers: 工作线程数
            pin_memory: 是否固定内存
            **dataset_kwargs: 传递给 Dataset 的参数
        """
        self.batch_size = batch_size or config.BATCH_SIZE
        self.num_workers = num_workers if num_workers is not None else config.NUM_WORKERS
        self.pin_memory = pin_memory if pin_memory is not None else config.PIN_MEMORY
        self.dataset_kwargs = dataset_kwargs
        
        # 缓存数据集实例
        self._train_dataset = None
        self._test_dataset = None
    
    def create_train_loader(
        self,
        shuffle: bool = True,
        drop_last: bool = False,
        **loader_kwargs
    ) -> DataLoader:
        """创建训练集 DataLoader"""
        if self._train_dataset is None:
            self._train_dataset = WeatherSequenceDataset(
                data_path=config.TRAIN_DATA_PATH,
                **self.dataset_kwargs
            )
        
        return DataLoader(
            dataset=self._train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=drop_last,
            **loader_kwargs
        )
    
    def create_test_loader(
        self,
        shuffle: bool = False,
        drop_last: bool = False,
        **loader_kwargs
    ) -> DataLoader:
        """创建测试集 DataLoader"""
        if self._test_dataset is None:
            self._test_dataset = WeatherSequenceDataset(
                data_path=config.TEST_DATA_PATH,
                **self.dataset_kwargs
            )
        
        return DataLoader(
            dataset=self._test_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=drop_last,
            **loader_kwargs
        )
    
    def create_loaders(
        self,
        shuffle_train: bool = True,
        shuffle_test: bool = False,
        **loader_kwargs
    ) -> Dict[str, DataLoader]:
        """同时创建训练集和测试集 DataLoader"""
        return {
            'train': self.create_train_loader(shuffle=shuffle_train, **loader_kwargs),
            'test': self.create_test_loader(shuffle=shuffle_test, **loader_kwargs)
        }
    
    def get_feature_dims(self) -> Dict[str, int]:
        """获取特征维度"""
        if self._train_dataset is None:
            self._train_dataset = WeatherSequenceDataset(
                data_path=config.TRAIN_DATA_PATH,
                **self.dataset_kwargs
            )
        return self._train_dataset.get_feature_dims()
    
    def get_num_targets(self) -> int:
        """获取目标数量"""
        if self._train_dataset is None:
            self._train_dataset = WeatherSequenceDataset(
                data_path=config.TRAIN_DATA_PATH,
                **self.dataset_kwargs
            )
        return self._train_dataset.get_num_targets()



