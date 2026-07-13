"""
Dataset Loader 模块（小时数据）
天气数据的 PyTorch 数据加载器

主要功能：
1. 时序滑动窗口数据集（24小时窗口）
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
    'WeatherSequenceDataset',
    'get_dataloader',
    'get_train_dataloader',
    'get_test_dataloader',
    'get_dataloaders',
    'print_dataloader_info',
    'DataLoaderFactory',
    'config',
    'feature_config',
    'utils',
]


def create_standard_loaders(batch_size=64, num_workers=4):
    """
    快速创建标准配置的训练和测试 DataLoader

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
    """获取特征配置信息"""
    return feature_config.get_feature_groups()


def print_info():
    """打印模块信息"""
    print("\n" + "=" * 70)
    print(" " * 20 + "Dataset Loader 模块（小时数据）")
    print("=" * 70)
    print(f"版本: {__version__}")
    print(f"\n配置:")
    print(f"  序列长度: {config.SEQ_LENGTH} 小时")
    print(f"  预测间隔: {config.PRED_HORIZON} 小时")
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

    print("=" * 70 + "\n")
