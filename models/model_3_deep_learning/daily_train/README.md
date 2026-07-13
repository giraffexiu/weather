# Wide & Deep 天气预测模型

## 📋 模型说明

基于PyTorch的Wide & Deep架构，用于多目标天气预测（温度、降水、风速共9个指标）。

### 模型特点
- **Wide侧**：22维手工特征（滞后、滚动统计、变化率、交叉特征）
- **Deep侧**：[64, 32]深度网络 + Embeddings + 正则化
- **损失函数**：加权MSE（温度1x, 降水3x, 风速2x）
- **参数量**：10,872

## 🚀 快速开始

### 训练模型
```bash
python train.py
```

### 评估模型
```bash
python evaluate.py
```

## 📊 性能指标（测试集）

| 指标 | MAE | RMSE | R² |
|------|-----|------|-----|
| **温度预测** | | | |
| Temperature Max | 0.218°C | 0.281 | 0.918 |
| Temperature Min | 0.206°C | 0.270 | 0.922 |
| Temperature Mean | 0.177°C | 0.233 | **0.943** |
| Temperature Range | 0.570°C | 0.716 | 0.456 |
| Feels Like | 0.173°C | 0.228 | 0.945 |
| **降水预测** | | | |
| Precipitation | 0.601mm | 1.115 | 0.087 |
| Rain | 0.600mm | 1.125 | 0.088 |
| Snow | 0.230mm | 0.832 | 0.085 |
| **风速预测** | | | |
| Wind Speed | 0.608m/s | 0.791 | 0.399 |

### 训练效率
- **训练时间**：~2分钟（早停于epoch 5）
- **最佳epoch**：5
- **设备**：CPU

## 🔧 关键改进

### 相比初始版本
1. ✅ 简化架构：hidden层 [128,64,32] → [64,32]
2. ✅ 增强特征：14维 → 22维（添加滞后、滚动、变化率）
3. ✅ 加权损失：平衡多目标学习
4. ✅ 优化训练：降低LR、增强正则化、早停

### 最大改进
- **Temperature Range R²**: 0.005 → 0.456 (+9020%)

## 📁 文件结构

```
daily_train/
├── train.py              # 训练脚本（包含WeightedMSELoss）
├── evaluate.py           # 评估脚本
├── model.py              # Wide & Deep模型定义
├── train_config.py       # 超参数配置
├── utils.py              # 工具函数
├── requirements.txt      # 依赖包
├── README.md            # 本文件
├── FIXES_APPLIED.md     # 详细修复说明
└── outputs/
    ├── checkpoints/     # 模型检查点
    └── plots/           # 训练曲线和预测分析图
```

## 🆚 与逻辑回归对比

| 模型 | 温度R² | 降水R² | 风速R² | 训练时间 |
|------|--------|--------|--------|---------|
| 逻辑回归 | 0.994-0.999 | 0.270 | 0.898 | 24秒 |
| 深度学习 | 0.918-0.945 | 0.087 | 0.399 | 2分钟 |

**结论**：对于此任务，逻辑回归整体性能更优。深度学习模型在温度预测上可用，但降水和风速预测需进一步改进。

## 💡 使用建议

### ✅ 推荐场景
- 温度预测（R² > 0.91）
- 研究学习Wide & Deep架构
- 对比深度学习与传统方法

### ⚠️ 限制
- 降水预测性能有限（R² < 0.1）
- 风速预测中等（R² ≈ 0.4）
- 需要更多数据和特征改进

## 📚 技术细节

### 特征工程（Wide侧）
```python
# 滞后特征 (7个)
temp_lag1, temp_lag2, temp_lag7
precip_lag1, precip_lag2
wind_lag1
lat, lon

# 滚动统计 (5个)
temp_7d_mean, temp_7d_std, temp_3d_mean
precip_7d_sum, precip_3d_sum

# 变化率 (2个)
temp_diff_1day, temp_diff_7day

# 交叉特征 (8个)
month×precip, lat×temp, season×wind (4维), is_rainy×precip, month
```

### 超参数配置
```python
# 模型
hidden_dims = [64, 32]
dropout = 0.4
embeddings = [6, 6, 3]  # city, country, season

# 训练
batch_size = 128
learning_rate = 0.0005
weight_decay = 1e-3
patience = 8
```

## 📖 更多信息

详细的修复过程、对比分析和技术总结请参考：
- `FIXES_APPLIED.md` - 完整修复说明和技术细节

## 📝 引用

如需引用此模型，请参考：
```
Wide & Deep天气预测模型
特征工程：滞后+滚动+变化率+交叉
损失函数：加权MSE
性能：温度R²=0.943, 参数量10,872
```
