# 天气数据集分析与预处理

对欧洲 9 个主要城市 2015-2024 年天气数据进行完整的探索性分析与预处理，为 LSTM 天气预测模型构建训练数据。

## 数据集

| 指标 | Daily（日数据） | Hourly（小时数据） |
|------|:-:|:-:|
| 文件数 | 9 | 9 |
| 总记录数 | 32,877 | 789,048 |
| 字段数 | 13 | 19 |
| 城市数 | 9 | 9 |
| 时间范围 | 2015-01-01 ~ 2024-12-31 | 2015-01-01 ~ 2024-12-31 |
| 缺失值 | 0 | 0 |
| 数据来源 | Open-Meteo Historical Weather API (ERA5) | 同左 |

覆盖城市：Amsterdam, Brussels, Copenhagen, Lisbon, Milan, Oslo, Stockholm, Vienna, Zurich

## 项目结构

```
dataset_analysis/
├── README.md
├── src/
│   ├── load_data.py              # 数据加载、合并、数据字典生成
│   ├── clean_data.py             # 缺失值/异常值/重复值/一致性检测
│   ├── feature_engineering.py    # 时间特征提取、标准化、LSTM 序列构建
│   ├── split_data.py             # 时序划分（训练/验证/测试）、数据泄露检测
│   └── pipeline.py               # 完整流水线（一键执行全部步骤）
├── notebooks/
│   └── run_eda.py                # EDA 可视化脚本
├── data/
│   └── processed/                # 处理后的训练数据（.npz）
├── outputs/
│   ├── figures/                  # EDA 图表（10张PNG）
│   └── reports/                  # 数据字典、异常值报告、分析报告
└── model/                        # 模型训练代码（待添加）

注意：原始数据从 `../dataset/` 目录读取（与 dataset_analysis 同级）
```

## 快速开始

### 环境

```bash
# 从项目根目录
source .venv/bin/activate
pip install -r ../requirements.txt
```

### 运行完整流水线

```bash
cd dataset_analysis
python3 src/pipeline.py
```

输出：
- 数据质量报告（缺失值、异常值、重复值）
- 数据字典 CSV
- 相关性矩阵
- LSTM 序列训练数据（`data/processed/*.npz`）
- 数据集分析报告 Markdown

### 自定义参数

```bash
# 自定义回溯窗口和预测窗口
python3 src/pipeline.py --lookback-hourly 336 --forecast-hourly 48

# 调整数据集划分比例
python3 src/pipeline.py --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
```

### 生成 EDA 图表

```bash
python3 notebooks/run_eda.py
```

生成图表包括：特征分布直方图、相关性热力图、标签分布、季节性模式、城市温度趋势、天气代码分布等。

## 数据处理流程

1. **数据加载** — 读取 9 个城市的 CSV，合并为统一 DataFrame，生成数据字典
2. **质量分析** — Z-score + IQR 异常值检测，缺失值/重复值统计，一致性检查
3. **数据清洗** — Winsorize 截尾（1%-99%），缺失值前向填充，城市 Label Encoding
4. **特征工程** — 周期性时间编码（sin/cos）、StandardScaler 标准化、滑动窗口序列构建
5. **数据划分** — 按时间顺序 7:1.5:1.5 分为训练/验证/测试集，确保无未来信息泄露

## LSTM 序列格式

| 维度 | Daily | Hourly |
|------|:-:|:-:|
| 回溯窗口 (lookback) | 30 天 | 168 小时（7天） |
| 预测窗口 (forecast) | 7 天 | 24 小时 |
| 特征数 | 18 | 24 |
| 预测目标 | temperature_2m_mean | temperature_2m |

加载训练数据：

```python
import numpy as np

data = np.load("data/processed/hourly_train.npz")
X_train, y_train = data["X"], data["y"]
# X_train: (551130, 168, 24) — 样本数, 时间步, 特征数
# y_train: (551130, 24)   — 样本数, 预测步数
```

## 数据质量结论

- 来源为 ERA5 再分析数据，质量极高：**零缺失、零重复**
- 异常值比例均在 2.5% 以下
- weather_code 全部符合 WMO 标准
- 时间范围连续无断点

## 后续模型训练建议

1. **LSTM 架构**：2-3 层 LSTM + Dropout(0.2-0.3)
2. **损失函数**：MSE（回归）/ CrossEntropyLoss（天气代码分类）
3. **优化器**：Adam + ReduceLROnPlateau
4. **Batch Size**：64-256
5. **早停**：patience=10，监控验证集 loss
6. **多城市训练**：可加入 city_embedding 层捕获城市差异
7. **评估指标**：RMSE、MAE、R² Score
