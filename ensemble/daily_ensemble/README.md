# Daily Ensemble - 软投票集成

基于 F1-Score 和 R² 的加权平均集成方法，融合 Model 1 (Logistic/Ridge Regression) 和 Model 3 (Wide & Deep) 的预测结果。

## 📋 项目结构

```
daily_ensemble/
├── config.py                    # 配置文件（路径、权重、阈值）
├── probability_converter.py     # 概率转换器（回归值→概率）
├── model_wrapper.py             # 模型包装器（统一接口）
├── soft_voting_ensemble.py      # 软投票集成主逻辑
├── evaluate_ensemble.py         # 集成模型评估
├── predict_ensemble.py          # 集成预测脚本
├── outputs/                     # 输出目录
│   ├── predictions/            # 预测结果
│   ├── results/                # 评估结果
│   └── plots/                  # 可视化图表
└── README.md                   # 本文件
```

## 🎯 核心功能

### 1. **软投票集成**

#### 回归任务（7个）
- **temperature_mean** (平均温度)
- **temperature_max** (最高温度)
- **temperature_min** (最低温度)
- **temperature_range** (温度范围)
- **temperature_feels** (体感温度)
- **precipitation** (降水量)
- **wind** (风速)

**集成方法**: 加权平均连续值
```python
ensemble_value = w1 * model1_value + w3 * model3_value
```

#### 分类任务（3个）
- **rain** (是否下雨)
- **snow** (是否下雪)
- **severe** (是否恶劣天气)

**集成方法**: 加权平均概率
```python
ensemble_prob = w1 * model1_prob + w3 * model3_prob
ensemble_pred = (ensemble_prob >= 0.5)
```

### 2. **权重计算**

基于模型性能自动计算权重：

#### 回归任务权重（基于 R²）
```python
w1 = model1_r2 / (model1_r2 + model3_r2)
w3 = model3_r2 / (model1_r2 + model3_r2)
```

#### 分类任务权重（基于 F1-Score）
```python
w1 = model1_f1 / (model1_f1 + model3_f1)
w3 = model3_f1 / (model1_f1 + model3_f1)
```

### 3. **概率转换**

Model 3 输出回归值，需要转换为分类概率：

#### Rain（降雨）
- 输入: `rain_sum` (mm)
- 方法: 基于阈值的分段映射
  - `rain_sum <= 0`: 概率 ≈ 0
  - `0 < rain_sum < 0.1mm`: 线性插值 [0, 0.5]
  - `rain_sum >= 0.1mm`: 线性插值 [0.5, 1.0]

#### Snow（降雪）
- 输入: `snow_sum` (mm)
- 方法: 同 Rain

#### Severe（恶劣天气）
- 输入: `temperature_range`, `wind_speed`, `precipitation`
- 方法: 多指标综合评分
```python
severe_prob = 0.3 * temp_score + 0.4 * wind_score + 0.3 * precip_score
```

## 🚀 使用方法

### 环境准备

```bash
cd /Users/giraffe/Downloads/Life/Homework/weather
source venv/bin/activate
```

### 1. 验证配置

```bash
cd ensemble/daily_ensemble
python config.py
```

预期输出:
```
✅ 配置验证通过！
设备: mps
Model 1 目录: /Users/giraffe/Downloads/Life/Homework/weather/models/model_1_linear
Model 3 检查点: /Users/giraffe/Downloads/Life/Homework/weather/models/model_3_deep_learning/daily_train/outputs/checkpoints/best_model.pth
测试数据: /Users/giraffe/Downloads/Life/Homework/weather/data/data_engineer/daily_data/processed_data/test_features.csv
```

### 2. 执行预测

```bash
# 使用默认测试集
python predict_ensemble.py

# 使用自定义输入
python predict_ensemble.py --input /path/to/data.csv --output /path/to/results.csv

# 调整批次大小（针对 Model 3）
python predict_ensemble.py --batch-size 256
```

### 3. 评估性能

```bash
python evaluate_ensemble.py
```

输出包括:
- 单模型性能（Model 1, Model 3）
- 集成模型性能
- 性能改进情况
- 详细评估报告（保存到 `outputs/results/ensemble_evaluation.txt`）

## 📊 输出格式

### 预测结果 CSV

```csv
time,city,country,ensemble_temp_mean,ensemble_temp_max,...,ensemble_rain_prob,ensemble_rain_pred
2024-01-01,Beijing,China,5.2,10.3,...,0.75,1
2024-01-01,Shanghai,China,8.1,12.5,...,0.35,0
...
```

包含列:
- **基本信息**: `time`, `city`, `country`
- **回归预测**: `ensemble_temp_mean`, `ensemble_temp_max`, ...
- **分类概率**: `ensemble_rain_prob`, `ensemble_snow_prob`, `ensemble_severe_prob`
- **分类预测**: `ensemble_rain_pred`, `ensemble_snow_pred`, `ensemble_severe_pred`

## 🔧 配置说明

### 修改权重方法

编辑 `config.py`:

```python
# 等权重
WEIGHT_METHOD = 'equal'

# 基于性能的权重（推荐）
WEIGHT_METHOD = 'performance_based'
```

### 调整概率转换参数

编辑 `config.py` 中的 `PROBABILITY_CONVERSION_CONFIG`:

```python
PROBABILITY_CONVERSION_CONFIG = {
    'rain': {
        'threshold': 0.1,  # 降雨阈值 (mm)
        'scale': 10.0      # Sigmoid 缩放因子
    },
    'severe': {
        'thresholds': {
            'temp_range': 15.0,  # 温度变化阈值 (°C)
            'wind_speed': 10.0,   # 风速阈值 (m/s)
            'precipitation': 5.0  # 降水阈值 (mm)
        }
    }
}
```

## 📈 性能指标

### 回归任务
- **MAE** (Mean Absolute Error): 平均绝对误差
- **RMSE** (Root Mean Squared Error): 均方根误差
- **R²** (R-squared): 决定系数
- **MAPE** (Mean Absolute Percentage Error): 平均绝对百分比误差

### 分类任务
- **Accuracy**: 准确率
- **Precision**: 精确率
- **Recall**: 召回率
- **F1-Score**: F1 分数
- **ROC-AUC**: ROC 曲线下面积

## ⚙️ 技术细节

### Model 1 加载
- 10个独立的 sklearn 模型（3个 Logistic + 7个 Ridge）
- 输入：25 维特征向量
- 输出：概率（分类）或连续值（回归）

### Model 3 加载
- 单个 PyTorch Wide & Deep 模型
- 输入：7天时序窗口
- 输出：9个目标的连续值
- 使用 DataLoader 批量预测

### 数据对齐
- Model 1 和 Model 3 使用相同的测试集
- Model 3 的时序窗口确保对齐到同一时间点
- 预测结果按样本顺序一一对应

## 🐛 故障排查

### 问题1: 找不到模型文件

**错误**: `FileNotFoundError: Model not found`

**解决**: 
```bash
# 训练 Model 1
cd models/model_1_linear
python train.py

# 验证 Model 3 存在
ls models/model_3_deep_learning/daily_train/outputs/checkpoints/best_model.pth
```

### 问题2: 特征不匹配

**错误**: `Missing features in input data`

**解决**: 确保输入数据包含所有25个特征，使用 `processed_data/test_features.csv` 格式

### 问题3: 设备不可用

**错误**: `CUDA/MPS not available`

**解决**: 代码会自动回退到 CPU，性能可能较慢但不影响功能

## 📝 模型性能（参考）

### Model 1 性能
- **Rain**: F1=0.7737, AUC=0.8489
- **Snow**: F1=0.3490, AUC=0.9159
- **Severe**: F1=0.8449, AUC=0.9614
- **Temp_mean**: R²=0.8637, MAE=0.2863

### Model 3 性能
- 需要运行 `evaluate_ensemble.py` 获取

### 集成性能
- 预期：综合两个模型的优势，提升整体性能
- 实际：运行评估后查看具体改进

## 📚 参考资料

- [软投票集成原理](https://scikit-learn.org/stable/modules/ensemble.html#voting-classifier)
- Model 1 训练: `models/model_1_linear/train.py`
- Model 3 训练: `models/model_3_deep_learning/daily_train/train.py`

## 👥 维护

如有问题或建议，请联系项目维护者。

---

**最后更新**: 2026-07-13
**版本**: 1.0.0
