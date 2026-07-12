# Dataset Loader - 天气数据加载器

PyTorch 数据加载模块，用于时序天气预测任务。

## 📁 文件结构

```
dataset_loader/
├── __init__.py              # 模块入口
├── config.py                # 配置参数
├── feature_config.py        # 特征分组定义
├── utils.py                 # 工具函数
├── weather_dataset.py       # 核心 Dataset 类
├── data_loader.py           # DataLoader 工厂
├── requirements.txt         # 依赖
└── cache/                   # 缓存目录
```

## 🚀 快速使用

```python
from dataset_loader import get_dataloaders

# 创建训练和测试 DataLoader
loaders = get_dataloaders(batch_size=64)

# 训练循环
for batch in loaders['train']:
    categorical = batch['categorical']  # (batch, seq_len, 2)
    numerical = batch['numerical']      # (batch, seq_len, 22)
    cyclical = batch['cyclical']        # (batch, seq_len, 6)
    binary = batch['binary']            # (batch, seq_len, 9)
    season = batch['season']            # (batch, seq_len, 1)
    target = batch['target']            # (batch, n_targets)
    
    # 模型训练...
```

## ⚙️ 配置

修改 `config.py` 中的参数：

```python
SEQ_LENGTH = 7              # 使用过去7天
PRED_HORIZON = 1            # 预测未来1天
BATCH_SIZE = 64
TARGET_COLUMNS = ['temperature_2m_mean']
```

## 📊 数据格式

- **训练集**: 160,720 样本 (2015-2023)
- **测试集**: 17,591 样本 (2024)
- **特征数**: 40 (categorical:2, numerical:22, cyclical:6, binary:9, season:1)

## 💡 主要特性

- ✅ 时序滑动窗口
- ✅ 城市分组时序
- ✅ 特征分组返回
- ✅ 双层缓存机制
- ✅ GPU 兼容
