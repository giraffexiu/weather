# 🚀 Quick Start Guide

## 5分钟快速上手

### 📋 前置条件

```bash
# 1. 确保虚拟环境已激活
cd /Users/giraffe/Downloads/Life/Homework/weather
source venv/bin/activate

# 2. 确保 Model 1 和 Model 3 已训练
ls models/model_1_linear/models/*.pkl  # 应该看到10个模型文件
ls models/model_3_deep_learning/daily_train/outputs/checkpoints/best_model.pth  # 应该存在
```

### ✅ Step 1: 验证安装

```bash
cd ensemble/daily_ensemble
python test_all.py
```

**预期输出**：
```
✅ 所有测试通过 (4/4)
系统已就绪，可以执行预测或评估
```

### 🔮 Step 2: 执行预测

```bash
python predict_ensemble.py
```

**输出文件**：`outputs/predictions/ensemble_predictions.csv`

**示例输出**：
```
预测完成！共 17934 个样本

【回归任务】
----------------------------------------------------------------------
  temp_mean            | 均值:   -0.008 | 标准差:    0.985 | 范围: [-4.165, 3.177]
  precipitation        | 均值:    0.003 | 标准差:    1.004 | 范围: [-0.444, 22.651]

【分类任务】
----------------------------------------------------------------------
  rain                 | 正例率:  56.2% | 平均概率: 0.562 | 预测为1: 10,088 个
  severe               | 正例率:  28.0% | 平均概率: 0.285 | 预测为1: 5,014 个

预测结果已保存到: outputs/predictions/ensemble_predictions.csv
```

### 📊 Step 3: 评估性能

```bash
python evaluate_ensemble.py
```

**输出文件**：`outputs/results/ensemble_evaluation.txt`

**示例输出**：
```
【回归任务评估】
----------------------------------------------------------------------
TEMP_MEAN
  Model 1    | MAE: 0.2863 | RMSE: 0.3642 | R²: 0.8637
  Model 3    | MAE: 0.2456 | RMSE: 0.3124 | R²: 0.8956
  Ensemble   | MAE: 0.2512 | RMSE: 0.3201 | R²: 0.8892
  ✓ 改进    | R²: +0.0255 | MAE: -0.0056

【分类任务评估】
----------------------------------------------------------------------
RAIN
  Model 1    | F1: 0.7737 | AUC: 0.8489 | Acc: 0.7595
  Model 3    | F1: 0.7856 | AUC: 0.8612 | Acc: 0.7689
  Ensemble   | F1: 0.7912 | AUC: 0.8634 | Acc: 0.7734
  ✓ 改进    | F1: +0.0056 | AUC: +0.0022
```

## 🎨 自定义使用

### 使用自己的数据

```bash
python predict_ensemble.py \
    --input /path/to/your/data.csv \
    --output /path/to/output.csv \
    --batch-size 256
```

**数据格式要求**：
- 必须包含 Model 1 需要的25个特征
- 必须包含 `time`, `city`, `country` 等基本列
- CSV格式，UTF-8编码

### 调整权重方法

编辑 `config.py`：

```python
# 使用等权重（简单平均）
WEIGHT_METHOD = 'equal'

# 使用基于性能的权重（推荐）
WEIGHT_METHOD = 'performance_based'
```

### 调整概率转换参数

编辑 `config.py`：

```python
PROBABILITY_CONVERSION_CONFIG = {
    'rain': {
        'threshold': 0.1,   # 降雨阈值 (mm)
        'scale': 10.0       # Sigmoid缩放
    },
    'severe': {
        'thresholds': {
            'temp_range': 15.0,      # 温度变化阈值
            'wind_speed': 10.0,      # 风速阈值
            'precipitation': 5.0     # 降水阈值
        },
        'weights': {
            'temp_range': 0.3,       # 温度权重
            'wind_speed': 0.4,       # 风速权重
            'precipitation': 0.3     # 降水权重
        }
    }
}
```

## 📖 查看预测结果

### 使用 Python

```python
import pandas as pd

# 读取预测结果
df = pd.read_csv('outputs/predictions/ensemble_predictions.csv')

# 查看回归预测
print(df[['time', 'city', 'ensemble_temp_mean', 'ensemble_precipitation']].head())

# 查看分类预测
print(df[['time', 'city', 'ensemble_rain_prob', 'ensemble_rain_pred']].head())

# 统计分析
print(df['ensemble_temp_mean'].describe())
print(df['ensemble_rain_pred'].value_counts())
```

### 使用命令行

```bash
# 查看前10行
head -11 outputs/predictions/ensemble_predictions.csv | column -t -s,

# 统计下雨预测
cut -d',' -f X outputs/predictions/ensemble_predictions.csv | sort | uniq -c
# (X为 ensemble_rain_pred 列的索引)
```

## 🔧 故障排查

### 问题1: "Model not found"

```bash
# 检查模型文件
ls -l models/model_1_linear/models/
ls -l models/model_3_deep_learning/daily_train/outputs/checkpoints/

# 如果缺失，重新训练
cd models/model_1_linear && python train.py
```

### 问题2: "Missing features"

确保输入数据包含所有必需特征：

```python
import json

# 查看需要的特征
with open('models/model_1_linear/models/feature_names.json') as f:
    features = json.load(f)
    print(f"需要 {len(features)} 个特征:")
    print(features)
```

### 问题3: 内存不足

```bash
# 减小批次大小
python predict_ensemble.py --batch-size 64
```

### 问题4: 预测速度慢

```bash
# 增大批次大小（如果内存充足）
python predict_ensemble.py --batch-size 512

# 或者使用GPU（如果可用）
# 代码会自动检测并使用MPS/CUDA
```

## 📚 更多资源

- **完整文档**: 查看 `README.md`
- **实现细节**: 查看 `IMPLEMENTATION_SUMMARY.md`
- **配置说明**: 查看 `config.py` 中的注释
- **API文档**: 查看各模块的 docstring

## 🎯 常见任务

### 只预测特定城市

```python
import pandas as pd
from model_wrapper import Model1Wrapper, Model3Wrapper
from probability_converter import ProbabilityConverter
from soft_voting_ensemble import SoftVotingEnsemble
from config import PROBABILITY_CONVERSION_CONFIG
from dataset_loader import get_dataloaders

# 加载数据
test_df = pd.read_csv('data/data_engineer/daily_data/processed_data/test_features.csv')

# 筛选特定城市
beijing_df = test_df[test_df['city'] == 'Beijing']

# 初始化模型和集成器
model1 = Model1Wrapper()
converter = ProbabilityConverter(PROBABILITY_CONVERSION_CONFIG)
model3 = Model3Wrapper(probability_converter=converter)
ensemble = SoftVotingEnsemble(model1, model3)

# 预测（需要创建对应的DataLoader）
# ...
```

### 批量评估多个时间段

```bash
# 将数据按月份分割，分别预测
for month in {1..12}; do
    python predict_ensemble.py \
        --input data/test_month_${month}.csv \
        --output results/predictions_month_${month}.csv
done
```

### 生成性能报告

```python
# 运行评估后读取结果
with open('outputs/results/ensemble_evaluation.txt', 'r') as f:
    report = f.read()
    print(report)
```

## 💡 最佳实践

1. **定期重新评估**
   - 每次调整配置后运行 `evaluate_ensemble.py`
   - 确保集成效果符合预期

2. **验证数据质量**
   - 检查输入数据的缺失值
   - 确保特征分布与训练集一致

3. **监控预测结果**
   - 检查预测分布是否合理
   - 关注异常值和边界情况

4. **保存配置**
   - 记录使用的配置参数
   - 便于复现结果

## 🎉 完成！

现在你已经掌握了 Daily Ensemble 的基本使用方法。

如有问题，请参考完整文档或联系项目维护者。

Happy Forecasting! 🌤️
