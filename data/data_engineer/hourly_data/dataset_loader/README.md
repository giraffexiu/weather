# Dataset Loader（小时数据）

小时级天气数据的 PyTorch 数据加载器，供 Model 3 深度学习使用。

## 功能

1. **24小时滑动窗口时序序列**：使用过去 24 小时数据预测未来 1 小时
2. **特征分组**：categorical / numerical / cyclical / binary / season
3. **多目标回归**：同时预测温度、降水、风速、体感温度、相对湿度
4. **缓存机制**：内存 + 磁盘双级缓存
5. **简单窗口模式**：数据已按城市+时间排序，关闭按城市分组可大幅提速

## 使用方法

```python
from dataset_loader import create_standard_loaders, print_info

# 查看配置
print_info()

# 创建 DataLoader
loaders = create_standard_loaders(batch_size=64)

# 训练循环
for batch in loaders['train']:
    categorical = batch['categorical']   # (64, 24, 3)  int64
    numerical   = batch['numerical']     # (64, 24, 36) float32
    cyclical    = batch['cyclical']      # (64, 24, 8)  float32
    binary      = batch['binary']        # (64, 24, 15) float32
    season      = batch['season']        # (64, 24, 1)  float32
    target      = batch['target']        # (64, 5)      float32
```

## 配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| SEQ_LENGTH | 24 | 过去 24 小时窗口 |
| PRED_HORIZON | 1 | 预测未来 1 小时 |
| TARGET_TYPE | regression | 回归任务 |
| TARGET_COLUMNS | 5 个气象变量 | 温度/降水/风速/体感/湿度 |
| GROUP_BY_CITY | False | 简单窗口模式（加速） |
| BATCH_SIZE | 64 | 批次大小 |
| USE_CACHE | True | 启用缓存 |

## 特征分组

| 组 | 数量 | 说明 |
|----|------|------|
| categorical | 3 | city_id, country_id, weather_code_id（用于 Embedding） |
| numerical | 36 | 温度/降水/风/湿度/气压/云量等（已标准化） |
| cyclical | 8 | 月/日/星期/小时的 sin/cos 编码 |
| binary | 15 | is_rainy, is_snowy 等 0/1 标志位 |
| season | 1 | 季节（编码为 0-3） |

## 文件结构

```
dataset_loader/
├── __init__.py          # 模块入口
├── config.py            # 配置参数
├── feature_config.py    # 特征分组定义
├── utils.py             # 工具函数（缓存/序列化/统计）
├── weather_dataset.py   # WeatherSequenceDataset 类
├── data_loader.py       # DataLoader 工厂函数
├── cache/               # 磁盘缓存目录（gitignore）
└── README.md
```
