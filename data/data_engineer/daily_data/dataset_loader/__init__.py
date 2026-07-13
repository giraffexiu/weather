"""
Dataset Loader 模块
天气数据的 PyTorch 数据加载器

主要功能：
1. 时序滑动窗口数据集
2. 特征分组（categorical, numerical, cyclical, binary, season）
3. 缓存机制（内存 + 磁盘）
4. 灵活的 DataLoader 工厂函数
"""

from .weather_dataset import WeatherSequenceDataset
from .data_loader import (
    get_dataloader,
    get_train_dataloader,
    get_test_dataloader,
    get_dataloaders,
    print_dataloader_info,
    DataLoaderFactory
)
from . import config
from . import feature_config
from . import utils

__version__ = '1.0.0'

__all__ = [
    # 数据集类
    'WeatherSequenceDataset',
    
    # DataLoader 工厂函数
    'get_dataloader',
    'get_train_dataloader',
    'get_test_dataloader',
    'get_dataloaders',
    'print_dataloader_info',
    'DataLoaderFactory',
    
    # 配置模块
    'config',
    'feature_config',
    'utils',
]


# 便捷函数：快速创建标准配置的 DataLoader
def create_standard_loaders(batch_size=64, num_workers=4):
    """
    快速创建标准配置的训练和测试 DataLoader
    
    Args:
        batch_size: 批次大小
        num_workers: 工作线程数
        
    Returns:
        {'train': train_loader, 'test': test_loader}
    
    Example:
        >>> from dataset_loader import create_standard_loaders
        >>> loaders = create_standard_loaders(batch_size=64)
        >>> for batch in loaders['train']:
        ...     # 训练代码
        ...     pass
    """
    return get_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle_train=True,
        shuffle_test=False
    )


def get_feature_info():
    """
    获取特征配置信息
    
    Returns:
        dict: 特征分组字典
    """
    return feature_config.get_feature_groups()


def print_info():
    """打印模块信息"""
    print("\n" + "="*70)
    print(" "*20 + "Dataset Loader 模块")
    print("="*70)
    print(f"版本: {__version__}")
    print(f"\n配置:")
    print(f"  序列长度: {config.SEQ_LENGTH} 天")
    print(f"  预测间隔: {config.PRED_HORIZON} 天")
    print(f"  批次大小: {config.BATCH_SIZE}")
    print(f"  目标变量: {config.TARGET_COLUMNS}")
    print(f"  任务类型: {config.TARGET_TYPE}")
    print(f"  使用缓存: {config.USE_CACHE}")
    
    print(f"\n数据路径:")
    print(f"  训练集: {config.TRAIN_DATA_PATH}")
    print(f"  测试集: {config.TEST_DATA_PATH}")
    
    print(f"\n特征统计:")
    groups = feature_config.get_feature_groups()
    for group_name, features in groups.items():
        if group_name != 'ignored':
            print(f"  {group_name}: {len(features)} 个特征")
    
    print("="*70 + "\n")



