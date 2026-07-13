# 模型一：传统统计与线性基准 (Linear Baseline)

## 📋 概述

基于传统机器学习的基准模型，使用 Logistic Regression 和 Ridge Regression 进行天气预测。

### 模型定位
- **作用**: 提供基准性能（Baseline），评估后续复杂模型的提升效果
- **优势**: 可解释性强、训练快速、无需GPU
- **适用场景**: 
  - 分类任务：预测"明天是否有雨/雪"
  - 回归任务：预测"明天的温度"

---

## 🎯 任务定义

### 分类任务
- **目标**: 预测明天是否会下雨 (`is_rainy`)
- **模型**: Logistic Regression with L2 regularization
- **评估指标**: Accuracy, Precision, Recall, F1-Score, ROC-AUC

### 回归任务
- **目标**: 预测明天的平均温度 (`temperature_2m_mean`)
- **模型**: Ridge Regression with L2 regularization
- **评估指标**: MAE, RMSE, R²

---

## 📁 目录结构

```
model_1_linear/
├── README.md                    # 本文档
├── QUICK_START.md               # 快速开始指南
├── ALL_RESULTS_SUMMARY.md       # 完整训练结果总结
├── config.py                    # 配置文件
├── train.py                     # ⭐ 统一训练脚本
├── evaluate.py                  # ⭐ 模型评估脚本
├── predict.py                   # ⭐ 预测脚本
├── requirements.txt             # 依赖
├── models/                      # 保存的模型 (9个)
│   ├── logistic_*.pkl          # 分类模型
│   ├── ridge_*.pkl             # 回归模型
│   └── feature_names.json      # 特征名称
└── results/                     # 评估结果
    ├── classification_*_report.txt
    ├── regression_*_report.txt
    ├── feature_importance_*.csv
    └── predictions/            # 预测结果
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
cd /Users/zhangyuecheng/Desktop/weather/models/model_1_linear
pip install -r requirements.txt
```

### 2. 训练模型

**分类任务（预测降雨）**
```bash
python train.py --task classification --target rain
```

**回归任务（预测温度）**
```bash
python train.py --task regression --target temp_mean
```

### 3. 评估模型

```bash
python evaluate.py --task classification --target rain
python evaluate.py --task regression --target temp_mean
```

### 4. 进行预测

```bash
python predict.py --task classification --target rain --input path/to/data.csv
python predict.py --task regression --target temp_mean --input path/to/data.csv
```

---

## ⚙️ 配置说明

编辑 `config.py` 修改参数：

```python
# 数据路径（Daily 或 Hourly）
DATA_TYPE = 'daily'  # 'daily' 或 'hourly'

# 分类任务目标
CLASSIFICATION_TARGET = 'is_rainy'  # 或 'is_snowy'

# 回归任务目标
REGRESSION_TARGET = 'temperature_2m_mean'

# 正则化参数
LOGISTIC_C = 1.0      # 越小正则化越强
RIDGE_ALPHA = 1.0     # 越大正则化越强

# 其他参数
RANDOM_STATE = 42
MAX_ITER = 1000
```

---

## 📊 特征使用

### Daily Data 特征
- 原始数值特征: 温度、降水、风速、辐射
- 时间特征: 月份、季节、周期编码
- 派生特征: 温差、体感温度、降水等级等
- 地理特征: 经纬度、城市/国家编码

### Hourly Data 特征
- 原始数值特征: 温度、降水、风速、湿度、气压、云量等
- 时间特征: 小时、月份、季节、周期编码
- 派生特征: 风寒指数、热指数、降水强度等
- 类别特征: 天气代码、城市、国家

---

## 📈 预期性能

### 分类任务（降雨预测）
- **Accuracy**: ~75-80%
- **F1-Score**: ~0.70-0.75
- **ROC-AUC**: ~0.80-0.85

### 回归任务（温度预测）
- **MAE**: ~2-3°C
- **RMSE**: ~3-4°C
- **R²**: ~0.85-0.90

*注: 实际性能取决于数据质量和特征工程*

---

## 🔍 模型特点

### 优势
✅ **快速训练**: 几秒钟即可完成训练  
✅ **可解释性强**: 可查看特征权重  
✅ **无需GPU**: CPU即可运行  
✅ **稳定性好**: 不易过拟合  
✅ **基准明确**: 为复杂模型提供对比

### 局限
❌ **线性假设**: 无法捕捉非线性关系  
❌ **特征依赖**: 需要手工特征工程  
❌ **时序信息**: 无法利用序列依赖  
❌ **交互效果**: 难以学习特征交互

---

## 📝 使用示例

### Python 代码

```python
# 加载训练好的模型
import pickle
import pandas as pd

# 加载分类模型
with open('models/logistic_rain.pkl', 'rb') as f:
    classifier = pickle.load(f)

# 加载数据
test_data = pd.read_csv('path/to/test_features.csv')
X_test = test_data[feature_columns]

# 预测
predictions = classifier.predict(X_test)
probabilities = classifier.predict_proba(X_test)

print(f"预测结果: {predictions}")
print(f"降雨概率: {probabilities[:, 1]}")
```

---

## 🛠️ 故障排除

**Q: 训练时报内存错误？**  
A: 对于 hourly 数据（380万+样本），可以：
- 使用采样训练：`train_classifier.py --sample 0.1`
- 使用 SGDClassifier 替代 LogisticRegression

**Q: 模型性能不佳？**  
A: 检查：
1. 特征是否正确加载（不包含目标列）
2. 数据是否标准化
3. 类别是否平衡（分类任务）

**Q: 预测时出错？**  
A: 确保：
1. 输入数据特征顺序与训练时一致
2. 特征已经过相同的预处理
3. 类别特征已经编码

---

## 📧 后续改进方向

1. **特征选择**: 使用 L1 正则化或特征重要性筛选
2. **类别平衡**: 使用 SMOTE 或调整类别权重
3. **多任务学习**: 同时预测温度和降水
4. **集成方法**: 结合多个线性模型

---

**最后更新**: 2026-07-12
