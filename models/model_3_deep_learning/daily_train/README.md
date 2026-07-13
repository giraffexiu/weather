# Wide & Deep 天气预测模型 - 训练模块

## 📋 项目概述

基于 Wide & Deep 架构的日度温度预测模型，利用过去7天的多维天气特征预测未来第1天的平均温度。

### 数据规模
- **训练集**: 160,720 样本 (2015-2023, 9年)
- **测试集**: 17,591 样本 (2024, 1年)
- **特征维度**: 40维 (分组: categorical, numerical, cyclical, binary, season)
- **覆盖范围**: 49个欧洲城市

### 模型架构
- **Wide侧**: 15维交叉特征 → 线性层 → 输出
- **Deep侧**: 123维特征 → 3层MLP [128→64→32] → 输出
- **融合**: Wide输出 + Deep输出
- **参数量**: ~22K

---

## 🚀 快速开始

### 1. 环境准备

```bash
cd models/model_3_deep_learning/daily_train
pip install -r requirements.txt
```

### 2. 训练模型

```bash
python train.py
```

训练过程会自动：
- 加载 dataset_loader 提供的数据
- 训练 80 个 epoch (可早停)
- 保存最佳模型到 `outputs/checkpoints/best_model.pth`
- 生成训练曲线和预测分析图

### 3. 输出文件

```
outputs/
├── checkpoints/
│   └── best_model.pth          # 最佳模型
├── plots/
│   ├── training_curves.png     # 训练曲线
│   └── prediction_analysis.png # 预测分析
└── logs/
```

---

## 📊 模型详情

### Wide侧特征 (15维)
1. **原始特征**: 当前温度、降水、风速、纬度
2. **统计特征**: 7天温度均值、标准差、降水总和
3. **交叉特征**:
   - month × precipitation (季节性降水模式)
   - latitude × temperature (地理规律)
   - season × wind_speed (季节性风力)
   - is_rainy × precipitation (降雨强度)

### Deep侧特征 (123维)
1. **类别Embedding** (20维):
   - city: 49类 → 8维
   - country: 29类 → 8维
   - season: 4类 → 4维

2. **数值聚合** (88维):
   - mean: 22维 (7天趋势)
   - std: 22维 (7天波动)
   - last: 22维 (当前状态)
   - diff: 22维 (总变化量)

3. **周期编码** (6维): sin/cos 月、年、周周期
4. **二值特征** (9维): 事件发生频率

### 网络结构

```
Deep侧:
Input(123) → Linear(128) + BN + ReLU + Dropout(0.3)
          → Linear(64)  + BN + ReLU + Dropout(0.3)
          → Linear(32)       + ReLU + Dropout(0.2)
          → Linear(1)

Wide侧:
Input(15) → Linear(1)

融合:
output = wide_out + deep_out
```

---

## ⚙️ 配置说明

编辑 `config.py` 修改超参数：

```python
TRAIN_CONFIG = {
    'batch_size': 64,
    'epochs': 80,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'dropout': 0.3,
    
    # 早停
    'early_stopping': True,
    'patience': 15,
}
```

---

## 📈 性能目标

- **MAE**: < 3.0°C
- **RMSE**: < 5.0°C
- **R²**: > 0.85

---

## 🔧 技术要点

### 1. 时序特征聚合
由于 Wide & Deep 不是时序模型，对7天序列采用统计聚合：
- `mean`: 捕捉趋势
- `std`: 捕捉波动
- `last`: 捕捉当前状态
- `diff`: 捕捉变化率

### 2. 类别特征处理
利用"按城市分组"特性：
- city_id 和 country_id 在7天内不变，取任意一天即可
- season 可能跨月，取最后一天

### 3. Wide侧设计
手工交叉特征捕捉天气领域规律：
- 季节×降水 → 夏季暴雨模式
- 纬度×温度 → 地理气候规律
- 季节×风速 → 冬季大风

### 4. 正则化策略
- Dropout: 0.3 (深层), 0.2 (浅层)
- BatchNorm: 每层隐藏层
- Weight Decay: 1e-4
- Gradient Clipping: 1.0

---

## 📝 注意事项

1. **数据路径**: 自动从 `data/data_engineer/daily_data/dataset_loader` 加载
2. **GPU支持**: 自动检测CUDA，无GPU则使用CPU
3. **内存占用**: 约 2GB (训练时)
4. **训练时间**: ~10-15分钟 (GPU), ~30-40分钟 (CPU)

---

## 🐛 常见问题

**Q: 找不到 dataset_loader 模块？**  
A: 确保在项目根目录运行，或检查 `sys.path` 设置

**Q: CUDA out of memory？**  
A: 降低 batch_size 到 32 或 16

**Q: 训练过拟合？**  
A: 增加 dropout 到 0.4，或启用更强的 weight_decay

**Q: R² 低于 0.85？**  
A: 检查数据质量，尝试调整特征工程或增加 hidden_dims

---

## 📧 技术架构

```
数据流:
CSV (161K rows) 
  → dataset_loader (滑动窗口)
  → batch {categorical, numerical, cyclical, binary, season, target}
  → FeaturePreprocessor (时序聚合 + Embedding)
  → Wide & Deep (并行处理)
  → 融合输出
```

---

**最后更新**: 2024-07-13
