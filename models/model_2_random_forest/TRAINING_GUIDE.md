# 随机森林训练指南 (Random Forest Training Guide)

> 模型：Model 2 Random Forest  
> 项目：final2/weather — 欧洲城市天气预测系统  
> 粒度：小时级多目标回归（5 变量连续值预测）  
> 日期：2026-07-12（更新：2026-07-14，移除日级）

---

## 1. 环境准备

- **Python**: 3.13+
- **核心依赖**: scikit-learn 1.9, pandas 3.0, numpy 2.4, matplotlib, seaborn
- **数据依赖**:
  - `data/data_engineer/hourly_data/processed_data/` — 小时级特征（67 列基础，经 data_loader 增强至 143 列）

```bash
# 安装依赖
pip install -r requirements.txt

# 确认特征工程已完成
python data/data_engineer/hourly_data/main.py
```

---

## 2. 数据加载与标签构造

### 2.1 小时级：5 变量多目标回归

预测目标：下一小时的 5 个气象变量连续值：
- temperature_2m（温度）
- precipitation（降水量）
- wind_speed_10m（风速）
- apparent_temperature（体感温度）
- relative_humidity_2m（相对湿度）

标签构造流程：

```python
# data_loader.py -> load_hourly()
df = df.sort_values(["city_id", "time"])
# 目标：下一小时的 5 个气象变量值
for col in target_cols:
    df[f"target_{col}"] = df.groupby("city_id")[col].shift(-1)
df = df.dropna(subset=[f"target_{col}" for col in target_cols])
```

**防泄露检查**：
- 用当前小时的特征预测**下一小时**的变量值（shift(-1)）
- 严格使用已有时间切分（train: 2015~2022, test: 2023~2024）
- 删除每个城市最后一条记录（无"下一小时"数据）

### 2.2 数据重切分

上游 CSV: train=2015~2023, test=2024。本层将 2023 年数据从训练集移入测试集：
- 训练集: time < 2023-01-01 (2015~2022)
- 测试集: time >= 2023-01-01 (2023~2024)

### 2.3 precipitation log1p 变换

降水是长尾分布（大量0值+极端值），log1p 变换后更接近正态，RF 回归更稳定。

### 2.4 时序特征增强（data_loader 内构造）

| 特征类型 | 具体特征 | 作用 |
|----------|----------|------|
| 滞后特征 | `temp_lag_1/2/3/6/12/24/48` | 前 1-48 小时的值 |
| 滑动窗口 | `temp_roll3/7/12/24/48_mean/std` | 过去窗口的均值/波动率 |
| 气压变化率 | `pressure_change_1/3/6` | 气压骤降=暴雨前兆 |
| 交互特征 | `temp_humidity`, `pressure_wind`, `cloud_rad_ratio` | 捕捉特征间非线性关系 |

---

## 3. 训练流程

### 3.1 快速训练（跳过 GridSearch）

```bash
python train_hourly.py --no-grid
```

使用已知最优参数（无需搜索）：
```python
best_params = {
    "n_estimators": 300, "max_depth": 30,
    "min_samples_leaf": 2, "max_features": 0.3,
}
```

### 3.2 超参数搜索（GridSearchCV + TimeSeriesSplit）

```python
# config.py
PARAM_GRID = {
    "n_estimators": [200, 300],
    "max_depth": [20, 30, None],
    "min_samples_leaf": [2, 4],
    "max_features": ["log2", 0.3],
}

# TimeSeriesSplit 保证训练数据始终在测试数据之前（防泄露）
tscv = TimeSeriesSplit(n_splits=5)
grid = GridSearchCV(rf_base, PARAM_GRID, cv=tscv,
                     scoring="neg_mean_squared_error", n_jobs=-1, refit=True)
```

**两阶段策略**：
```python
# 阶段1：10% 子样本粗搜（系统抽样，覆盖全年各季节）
X_sub, y_sub, _, _, _ = load_hourly(subset_frac=0.1)

# 阶段2：全量训练（用阶段1 的最优参数）
X_train, y_train, X_test, y_test, _ = load_hourly(subset_frac=1.0)
rf_final.fit(X_train, y_train)
```

### 3.3 OOB 评估

```python
RF_FIXED_PARAMS = {
    "oob_score": True,    # 启用袋外评估
    "bootstrap": True,    # OOB 要求 bootstrap=True
}
rf_final.fit(X_train, y_train)
print(f"OOB Score: {rf_final.oob_score_:.4f}")
```

OOB Score 是无偏泛化估计，无需额外验证集。每个决策树训练时只用 ~63% 的数据（bootstrap 采样），剩余 ~37% 的袋外数据自动用于评估。

---

## 4. 评估

### 评估指标

| 维度 | 指标 | 说明 |
|------|------|------|
| 整体 | Overall RMSE | 5 目标平均均方根误差 |
| 整体 | Overall MAE | 5 目标平均绝对误差 |
| 整体 | Overall R² | 5 目标平均决定系数 |
| 整体 | OOB Score | 袋外无偏估计 |
| 逐目标 | RMSE / MAE / R² | 每个气象变量单独评估 |

### 当前模型表现

| 目标变量 | R² |
|---------|-----|
| temperature_2m | 0.9938 |
| apparent_temperature | 0.9937 |
| relative_humidity_2m | 0.9554 |
| wind_speed_10m | 0.9149 |
| precipitation | 0.3409 |
| **Overall** | **0.8397** |

> precipitation 的 R² 仅 0.34，长尾分布预测仍是弱项。

### 可视化产出

运行 `evaluate.py` 后在 `outputs/hourly/` 生成：
1. `feature_importance.png` — Top-20 特征水平条形图
2. `evaluation_report.md` — 完整 Markdown 评估报告
3. `evaluation_report.json` — 机器可读的指标数据

---

## 5. 预测接口

```python
from models.model_2_random_forest.predict import RFPredictor

rf = RFPredictor()                           # 加载小时级模型
result = rf.predict_hourly(X)                 # 返回预测值 + 元信息
weight = rf.get_weight()                      # OOB Score，供加权平均策略使用
importance = rf.get_feature_importance()      # 特征重要性字典
```

---

## 6. 常见问题

### Q1: 小时级训练太慢怎么办？
使用 `train_hourly.py` 的两阶段策略：先用 10% 子样本粗搜参数，再用全量训练。

### Q2: 标准化数据对随机森林有影响吗？
无实质影响。树模型基于阈值分裂，不受量纲影响。但使用标准化数据可保持与 model_1/model_3 的管道一致性。

### Q3: 如何与集成层对接？
```python
from models.model_2_random_forest.predict import RFPredictor

rf = RFPredictor()
preds = rf.predict(X)                    # → [n, 5] 预测值
weight = rf.get_weight()                # → OOB float，供加权平均
importance = rf.get_feature_importance()  # → {feature: importance}
fallback = rf.is_fallback_needed(X)     # → 布尔数组，兜底检查
```
