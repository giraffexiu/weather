# Daily Weather Data - 特征工程模块

## 📋 概述

本模块对清洗后的日度天气数据（`weather_daily_cleaned.csv`）进行特征工程处理，生成可直接用于机器学习模型的特征数据。

### 输入
- **数据源**: `data/data_clean/cleaned_data/weather_daily_cleaned.csv`
- **数据规模**: 178,997 条记录（49个欧洲城市，2015-2024年）
- **原始特征**: 12列（城市、国家、经纬度、时间、7个气象特征）

### 输出
- **训练集**: `processed_data/train_features.csv` (2015-2023)
- **测试集**: `processed_data/test_features.csv` (2024)
- **预处理对象**: `processed_data/preprocessors/` (scaler、映射字典)

---

## 🎯 特征工程策略

### 1. 时间特征分解

天气数据具有强季节性和周期性，从 `time` 列提取：

#### 基础时间特征
- `year`: 年份
- `month`: 月份 (1-12)
- `day`: 日期 (1-31)
- `day_of_year`: 一年中的第几天 (1-366)
- `day_of_week`: 星期几 (0-6)
- `quarter`: 季度 (1-4)
- `week_of_year`: 一年中的第几周
- `season`: 季节 (winter/spring/summer/autumn)

#### 周期性编码（三角函数）
- `month_sin`, `month_cos`: 月份周期
- `day_of_year_sin`, `day_of_year_cos`: 年周期
- `day_of_week_sin`, `day_of_week_cos`: 周周期

**为什么使用周期性编码？**
- 保持循环特性：12月和1月在数值上距离很大，但实际相邻
- sin/cos 编码让模型理解"12月末 = 1月初"

---

### 2. 类别特征编码

#### City 和 Country
- 建立索引映射（为 PyTorch `nn.Embedding` 准备）
- 输出: `city_id` (0-48), `country_id` (0-28)
- 保存映射字典供推理时使用

**为什么用 Embedding 而不是 OneHot？**
- OneHot: 49+29=78 维稀疏向量
- Embedding: 推荐 8+6=14 维稠密向量
- Embedding 可学习城市/国家间的地理相似性

---

### 3. 派生特征

基于领域知识创建的特征：

#### 温度相关
- `temperature_range`: 日温差（最高温 - 最低温）
- `is_freezing`: 是否结冰日（最低温 < 0°C）
- `is_hot_day`: 是否高温日（最高温 > 30°C）
- `feels_like_temperature`: 体感温度（考虑风速影响）

#### 降水相关
- `is_rainy`: 是否有降水
- `is_heavy_rain`: 是否暴雨（>10mm）
- `precipitation_level`: 降水等级（0-4）
- `snow_sum`: 降雪量估算（总降水 - 降雨）
- `is_snowy`: 是否下雪

#### 风速相关
- `wind_level`: 风力等级（0-5）
- `is_windy`: 是否大风（>20 km/h）

#### 辐射相关
- `radiation_level`: 辐射等级（0-2）
- `is_sunny`: 是否晴天（高辐射）

#### 综合特征
- `is_severe_weather`: 是否恶劣天气
- `is_high_latitude`: 是否高纬度地区

---

### 4. 数值特征标准化

使用 **StandardScaler** (Z-score 标准化):
- 所有数值特征转换为均值=0，标准差=1
- **关键**: 只在训练集上 `fit()`，测试集用训练集参数 `transform()`
- 标准化的特征包括：
  - 原始气象特征（温度、降水、风速、辐射）
  - 地理坐标（经纬度）
  - 派生的连续特征（温差、体感温度等）
  - 周期性编码特征（sin/cos）

---

### 5. 训练/测试集切分

**时间切分策略**（非随机切分）：
- **训练集**: 2015-01-01 至 2023-12-31 (9年)
- **测试集**: 2024-01-01 至 2024-12-31 (1年)

**为什么不能随机切分？**
- 天气数据是时序相关的
- 随机切分会导致数据泄露
- 测试集应该模拟"未来预测"场景

---

## 📁 目录结构

```
daily_data/
├── feature_engineer/          # 特征工程核心模块
│   ├── __init__.py
│   ├── time_features.py       # 时间特征提取
│   ├── categorical_encoder.py # 类别特征编码
│   ├── numerical_scaler.py    # 数值标准化
│   ├── feature_creator.py     # 派生特征创建
│   └── pipeline.py            # 特征工程流水线
│
├── processed_data/            # 输出目录（自动创建）
│   ├── train_features.csv     # 训练集特征
│   ├── test_features.csv      # 测试集特征
│   ├── feature_list.txt       # 特征列表
│   └── preprocessors/         # 预处理对象
│       ├── scaler.pkl         # 标准化器
│       ├── city_mapping.json  # 城市映射
│       └── country_mapping.json # 国家映射
│
├── config.py                  # 配置文件
├── main.py                    # 主执行脚本
└── README.md                  # 本文档
```

---

## 🚀 使用方法

### 快速开始

```bash
# 进入 daily_data 目录
cd data/data_engineer/daily_data

# 运行特征工程
python main.py
```

### 配置修改

编辑 `config.py` 修改参数：

```python
# 时间切分
TRAIN_END_DATE = "2023-12-31"
TEST_START_DATE = "2024-01-01"

# 特征工程选项
USE_CYCLICAL_ENCODING = True      # 是否使用周期性编码
CREATE_DERIVED_FEATURES = True    # 是否创建派生特征
SCALING_METHOD = 'standard'       # 标准化方法: 'standard' 或 'minmax'
```

### 单独使用各个模块

```python
from feature_engineer import TimeFeatureExtractor, FeatureCreator
import pandas as pd

# 读取数据
df = pd.read_csv("your_data.csv")

# 提取时间特征
time_extractor = TimeFeatureExtractor(time_column='time', use_cyclical=True)
df = time_extractor.fit_transform(df)

# 创建派生特征
feature_creator = FeatureCreator()
df = feature_creator.fit_transform(df)
```

---

## 📊 输出说明

### 1. train_features.csv / test_features.csv

包含所有原始特征和工程特征的 CSV 文件，列包括：

- **原始特征**: city, country, latitude, longitude, time, temperature_*, precipitation_*, rain_*, wind_*, radiation
- **类别编码**: city_id, country_id
- **时间特征**: year, month, day, quarter, season, *_sin, *_cos
- **派生特征**: temperature_range, is_freezing, is_rainy, snow_sum, wind_level, etc.

**注意**: 数值特征已标准化（均值≈0，标准差≈1）

### 2. preprocessors/

保存的预处理对象，用于：
- **推理时**: 对新数据应用相同的特征转换
- **模型部署**: 确保训练和推理使用相同的预处理参数

```python
# 加载预处理器示例
from feature_engineer import NumericalScaler
scaler = NumericalScaler.load("processed_data/preprocessors/scaler.pkl")
new_data_scaled = scaler.transform(new_data)
```

### 3. feature_list.txt

所有特征列名的清单，方便查看和选择特征

---

## 🔧 后续步骤

### 用于 PyTorch 模型

```python
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class WeatherDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        
        # 提取特征
        self.city_ids = self.df['city_id'].values
        self.country_ids = self.df['country_id'].values
        self.numerical_features = self.df[[
            'latitude', 'longitude', 
            'temperature_2m_max', 'precipitation_sum',
            # ... 其他数值特征
        ]].values
        
        # 目标变量（根据任务定义）
        self.targets = self.df['target'].values  # 需要定义
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        return {
            'city_id': torch.tensor(self.city_ids[idx], dtype=torch.long),
            'country_id': torch.tensor(self.country_ids[idx], dtype=torch.long),
            'numerical': torch.tensor(self.numerical_features[idx], dtype=torch.float32),
            'target': torch.tensor(self.targets[idx], dtype=torch.float32)
        }

# 使用
train_dataset = WeatherDataset('processed_data/train_features.csv')
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
```

### Embedding 层设置

```python
import torch.nn as nn

class WeatherModel(nn.Module):
    def __init__(self):
        super().__init__()
        # City embedding: 49 个类别 -> 8 维
        self.city_embedding = nn.Embedding(49, 8)
        # Country embedding: 29 个类别 -> 6 维
        self.country_embedding = nn.Embedding(29, 6)
        # ... 其他层
```

---

## ⚙️ 技术细节

### 依赖库

```bash
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
```

### 性能考虑

- **内存占用**: ~500MB（完整数据集）
- **处理时间**: ~30秒（在普通笔记本上）
- **输出文件大小**: 
  - train_features.csv: ~150MB
  - test_features.csv: ~20MB

### 可扩展性

模块化设计便于扩展：
- 添加新的派生特征：修改 `feature_creator.py`
- 使用不同的编码方法：修改 `categorical_encoder.py`
- 调整标准化策略：修改 `numerical_scaler.py`

---

## 📝 注意事项

1. **数据泄露**: 确保所有预处理器只在训练集上 fit
2. **缺失值**: 如果测试集出现训练集未见过的城市，会被映射为 -1
3. **特征选择**: 并非所有生成的特征都必须使用，可根据模型需要选择
4. **目标变量**: 本模块不包含目标变量定义，需根据具体任务添加

---

## 🐛 常见问题

**Q: 运行时提示找不到输入文件？**
A: 检查 `config.py` 中的路径设置，确保数据清洗步骤已完成

**Q: 生成的特征太多，如何选择？**
A: 可以通过特征重要性分析（如随机森林）或模型训练验证来筛选

**Q: 如何处理新城市？**
A: 训练集未见过的城市会被编码为 -1，建议在 Embedding 层添加 padding_idx 参数

**Q: 能否改用分层抽样切分？**
A: 对于分类任务可以，但务必保持时间顺序，不建议打乱时序

---

## 📧 联系方式

如有问题或建议，请联系项目负责人。

---

**最后更新**: 2024-07-12
