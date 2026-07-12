# 三大模型训练体系 (Three Models Training System)

## 项目概述
本目录包含三个不同算法范式的模型，用于天气预测的综合研判系统。

## 模型架构

### 1️⃣ 模型一：传统统计与线性基准 (Linear Baseline)
**目录**: `model_1_linear/`
- **算法**: Logistic Regression (分类) / Linear Regression (回归)
- **定位**: 基准模型（Baseline），提供线性和规则层面的"记忆能力"
- **特点**: 
  - 使用 L2 正则化防止过拟合
  - 适用于"明天是否有雨/雪"的分类预测
  - 适用于"明天具体温度"的回归预测

### 2️⃣ 模型二：非线性强规则捕捉 (Random Forest)
**目录**: `model_2_random_forest/`
- **算法**: Random Forest
- **定位**: 机器学习"常青树"，擅长处理结构化数据
- **特点**:
  - 捕捉特征间的非线性交互
  - 利用 OOB（袋外误差）进行无偏泛化评估
  - 通过 feature_importances_ 提供可解释性分析
  - 识别关键指标（湿度、气压降幅等）

### 3️⃣ 模型三：高维泛化专家 (Deep Learning)
**目录**: `model_3_deep_learning/`
- **算法**: Wide & Deep Neural Network (PyTorch)
- **定位**: 挖掘海量数据中隐藏的复杂长程模式
- **架构**:
  - **Wide 侧**: 当前气压、温度、风向×月份交叉特征（强规则捕捉）
  - **Deep 侧**: 标准化数值特征 + 风向Embedding，3层全连接DNN
  - **技术**: Dropout + BatchNorm 防止过拟合

## 数据流程
```
data/data_clean/cleaned_data/
    └─> data/data_engineer/daily_data/processed_data/
        └─> models/ (本目录)
            ├─> model_1_linear/
            ├─> model_2_random_forest/
            └─> model_3_deep_learning/
```

## 使用方式
每个模型目录都包含独立的训练、评估和预测脚本：
- `train.py`: 模型训练
- `evaluate.py`: 模型评估
- `predict.py`: 预测接口
- `config.py`: 模型配置参数
- `requirements.txt`: 依赖包

## 综合研判
三个模型的预测结果将通过集成学习策略进行融合，提供最终的天气预测结果。
