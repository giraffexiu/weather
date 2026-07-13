# 随机森林训练指南 (Random Forest Training Guide)

> 模型：Model 2 Random Forest  
> 项目：final2/weather — 欧洲城市天气预测系统  
> 双粒度：日级 + 小时级均做**天气分类**（6 类）  
> 日期：2026-07-12

---

## 1. 环境准备

- **Python**: 3.13+
- **核心依赖**: scikit-learn 1.9, pandas 3.0, numpy 2.4, matplotlib, seaborn
- **数据依赖**:
  - `data/data_engineer/daily_data/processed_data/` — 日级特征（43 列，无 weather_code）
  - `data/data_engineer/hourly_data/processed_data/` — 小时级特征（67 列，含 weather_code）
  - `data/data_clean/cleaned_data/weather_hourly_cleaned.csv` — 日级标签聚合来源

```bash
# 安装依赖
pip install -r requirements.txt

# 确认特征工程已完成
python data/data_engineer/daily_data/main.py
python data/data_engineer/hourly_data/main.py
```

### 6 类天气定义

| 类别 | 英文 | WMO weather_code | 含义 |
|------|------|-------------------|------|
| 晴 | Clear | 0 | 晴天 |
| 多云 | Cloudy | 1, 2 | 部分多云/多云 |
| 阴 | Overcast | 3 | 阴天 |
| 毛毛雨 | Drizzle | 51, 53, 55 | 毛毛雨 |
| 雨 | Rain | 61, 63, 65 | 雨 |
| 雪 | Snow | 71, 73, 75 | 雪 |

### 数据分布

**小时级**（直接从 weather_code 合并）：

| 类别 | 占比 | 记录数(train) |
|------|------|------|
| Overcast | 32.65% | 1,263,397 |
| Clear | 27.35% | 1,056,797 |
| Cloudy | 23.39% | 903,710 |
| Drizzle | 13.13% | 508,406 |
| Snow | 1.76% | 67,972 |
| Rain | 1.73% | 66,813 |

**日级**（最严重天气聚合）：

| 类别 | 占比 | 记录数 |
|------|------|------|
| Overcast | 33.92% | 60,712 |
| Drizzle | 32.37% | 57,938 |
| Rain | 14.52% | 25,992 |
| Cloudy | 8.00% | 14,320 |
| Snow | 6.89% | 12,341 |
| Clear | 4.29% | 7,687 |

---

## 2. 数据加载与标签构造

### 2.1 小时级：直接 6 分类

小时级特征数据已包含 `weather_code` 列。标签构造流程：

```python
# data_loader.py -> load_hourly()
df = df.sort_values(["city_id", "time"])
# 目标：下一小时的天气类别
df["target"] = df.groupby("city_id")["weather_code"].shift(-1)
df["target"] = df["target"].map(WMO_TO_CATEGORY)  # WMO code -> 6 类
df = df.dropna(subset=["target"])  # 删除每个城市最后 1 小时
```

**防泄露检查**：
- 用当前小时的特征预测**下一小时**的天气（shift(-1)）
- 严格使用已有时间切分（train: 2015-2023, test: 2024）
- 删除每个城市最后一条记录（无"下一小时"数据）

### 2.2 日级：从小时级聚合标签

日级特征数据**没有 weather_code 列**，需从小时级清洗数据聚合标签：

**"最严重天气"聚合策略**：
```python
# data_loader.py -> _aggregate_daily_labels()
# 优先级：Snow > Rain > Drizzle > Overcast > Cloudy > Clear
# 如果当天出现过雪，则当天标签=Snow（最严重）
def most_severe(categories):
    return min(set(categories), key=lambda c: priority_index[c])

daily_label = h.groupby(["city", "date"])["category"].agg(most_severe)
```

然后 join 到日级特征，再 shift(-1) 得到明天的天气：
```python
df["target"] = df.merge(daily_label, on=["city", "date"])
df["target"] = df.groupby("city_id")["target"].shift(-1)  # 明天
```

**为什么用"最严重"而非"众数"**：众数聚合会导致 Rain/Snow 几乎消失（0.48%/0.95%），而最严重聚合后 Rain 14.5%、Snow 6.9%，显著改善了少数类比例，是纯数据工程层面的不平衡缓解。

### 2.3 时序特征增强（data_loader 内构造）

| 特征类型 | 具体特征 | 作用 |
|----------|----------|------|
| 滞后特征 | `temp_lag_1/2/3`, `pressure_lag_1/2/3` | 前 1-3 小时/天的值 |
| 滑动窗口 | `temp_7d_mean`, `pressure_3h_std` | 过去窗口的均值/波动率 |
| 气压变化率 | `pressure_change_3h = pressure.shift(3) - pressure` | 气压骤降=暴雨前兆 |

---

## 3. 训练流程

### 3.1 基础训练（快速基线）

```bash
# 跳过 GridSearch，用默认参数快速跑通
python train_daily.py    # 不加参数，默认 use_grid_search=True
```

默认参数（无 GridSearch）：
```python
best_params = {
    "n_estimators": 200, "max_depth": 25,
    "min_samples_leaf": 2, "max_features": "sqrt",
}
```

### 3.2 超参数搜索（GridSearchCV + TimeSeriesSplit）

```python
# config.py
PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [15, 25, None],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", 0.3],
}

# TimeSeriesSplit 保证训练数据始终在测试数据之前（防泄露）
tscv = TimeSeriesSplit(n_splits=5)
grid = GridSearchCV(rf_base, PARAM_GRID, cv=tscv,
                     scoring="f1_macro", n_jobs=-1, refit=True)
```

**为什么用 TimeSeriesSplit 而非 KFold**：天气数据有时序依赖性，随机 KFold 会导致训练集中包含未来数据，造成数据泄露。TimeSeriesSplit 保证每折训练数据始终在测试数据之前。

**为什么用 f1_macro 而非 accuracy**：让调参过程优化少数类（Rain/Snow）而非被多数类（Overcast 33%）主导。

**小时级两阶段策略**：
```python
# 阶段1：10% 子样本粗搜（系统抽样，覆盖全年各季节）
X_sub, y_sub, _, _, _ = load_hourly(subset_frac=0.1)  # ~38 万行
grid.fit(X_sub, y_sub)

# 阶段2：全量训练（386 万行，用阶段1 的最优参数）
X_train, y_train, X_test, y_test, _ = load_hourly(subset_frac=1.0)
rf_final.fit(X_train, y_train)
```

### 3.3 OOB 评估

```python
RF_FIXED_PARAMS = {
    "oob_score": True,    # 启用袋外评估
    "bootstrap": True,    # OOB 要求 bootstrap=True
    "class_weight": "balanced",
}
rf_final.fit(X_train, y_train)
print(f"OOB Score: {rf_final.oob_score_:.4f}")
```

OOB Score 是无偏泛化估计，无需额外验证集。每个决策树训练时只用 ~63% 的数据（bootstrap 采样），剩余 ~37% 的袋外数据自动用于评估。

---

## 4. 效果优化（如何让随机森林效果更好）

### 4.1 时序特征增强：滞后特征 + 滑动窗口

在 `data_loader.py` 中构造（已在代码中实现）：

```python
# 滞后特征：前 1-3 小时/天的值
for lag in [1, 2, 3]:
    df[f"temp_lag_{lag}"] = df.groupby("city_id")["temperature_2m"].shift(lag)
    df[f"pressure_lag_{lag}"] = df.groupby("city_id")["pressure_msl"].shift(lag)

# 滑动窗口：过去 7 天/小时均温
df["temp_7d_mean"] = df.groupby("city_id")["temperature_2m"].transform(
    lambda x: x.rolling(7).mean())

# 气压变化率：气压骤降=暴雨前兆
df["pressure_change_3h"] = df.groupby("city_id")["pressure_msl"].shift(3) - df["pressure_msl"]
```

> 气象学强信号：气压 3 小时内下降 >3 hPa 通常是暴雨/恶劣天气前兆。

### 4.2 类别不平衡处理（7 种方法，按优先级递进）

> **背景**：Day2 实验报告已在医疗数据集上验证过 3 种方案（默认/balanced/SMOTE）。当前小时级 Rain/Snow 仅 1.7%，比 Day2 中已觉"极度稀疏"的 Cancer(4.8%) 更少 2.7 倍；日级 Rain/Snow 经聚合后达 7%~15%，问题较轻。

| # | 方法 | 来源 | 适用粒度 | 实现方式 | Day2 经验 |
|---|------|------|----------|----------|----------|
| 1 | **class_weight='balanced'** | Day2 已验证 | 两者 | `RandomForestClassifier(class_weight="balanced")` | ✅ 提升 Cancer 召回率，牺牲精确率 |
| 2 | **手动加大权重** | balanced 强化版 | 小时级 | `class_weight={Rain:5, Snow:5}` | balanced 不够时递进 |
| 3 | **SMOTE 过采样** | Day2 已实施 | 小时级 | `imblearn.over_sampling.SMOTE` | ⚠️ 极稀疏时效果有限，训练成本高 |
| 4 | **日级聚合放大少数类** | 项目独有 | 日级 | weather_code 按"最严重天气"聚合 | Rain 1.8%→14.5%，纯数据工程解决 |
| 5 | **Macro-F1 调参** | Day2 已理解 | 两者 | GridSearchCV `scoring="f1_macro"` | 让调参过程优化少数类而非 accuracy |
| 6 | **子采样+全量两阶段** | 工程优化 | 小时级 | 先用 10% 子样本粗搜参数，再全量训练 | 386 万行 GridSearch 太慢 |
| 7 | **集成增益** | Day7 已学 | 两者 | 3 模型不同错误模式→加权平均 | 即使单模型召回低，集成可互补 |

**推荐策略**（已在代码中实现）：
- **日级**：`class_weight="balanced"` + `scoring="f1_macro"`（少数类占比尚可，无需 SMOTE）
- **小时级**：`class_weight={Rain:5, Snow:5}` + 子样本粗搜 + 全量训练。若少数类 Recall 仍 <0.3，追加 SMOTE

### 4.3 概率校准

随机森林的概率通常不够校准，可用 Platt Scaling 校准，提升集成层软投票质量：

```python
from sklearn.calibration import CalibratedClassifierCV

# Platt Scaling
rf_calibrated = CalibratedClassifierCV(rf_final, method="sigmoid", cv=5)
rf_calibrated.fit(X_train, y_train)
```

**注意事项**：
- 用验证集/训练集 CV 校准，**不能在测试集校准**（会泄露）
- 校准后的概率更适合软投票/加权平均
- `predict.py` 的 `predict_proba` 已输出原始概率，集成层可自行决定是否校准

### 4.4 特征选择

用 `feature_importances_` 筛选 Top-30 特征，剔除噪声列：

```python
importance = dict(zip(feature_names, rf_final.feature_importances_))
top_30 = sorted(importance.items(), key=lambda x: -x[1])[:30]
# 剔除重要性极低（<0.001）的噪声列
```

- 删除常量列（如测试集中的 `year` 可能恒为 2024）
- `evaluate.py` 的 `plot_feature_importance` 已输出 Top-20 可视化

### 4.5 集成多样性（与其他模型差异化）

| 差异化策略 | 说明 |
|------------|------|
| RF 用全部特征 | DL 用精炼特征 → 不同错误模式 → 集成增益更高 |
| 使用标准化数据 | RF 对标准化不敏感，但保持管道一致性 |
| 时序特征增强 | RF 独有滞后/滑动窗口特征，DL 用 Embedding |

---

## 5. 评估与可视化

### 评估指标矩阵

| 维度 | 指标 | 说明 |
|------|------|------|
| 整体 | Accuracy | 准确率 |
| 整体 | Macro-F1 | 多类别均衡 F1 |
| 整体 | Weighted-F1 | 按类别加权的 F1 |
| 整体 | OOB Score | 袋外无偏估计 |
| 整体 | Cohen's Kappa | 一致性度量 |
| 整体 | Log Loss | 概率排序能力 |
| 分类 | Precision / Recall | 按类别输出 |
| 校准 | Brier Score | 概率校准度 |

### 可视化产出

运行 `evaluate.py` 后在 `outputs/{daily,hourly}/` 生成：

1. **confusion_matrix.png** — 混淆矩阵（绝对值 + 百分比）
2. **feature_importance.png** — Top-20 特征水平条形图
3. **probability_distribution.png** — 预测概率分布（按真实标签着色）
4. **evaluation_report.md** — 完整 Markdown 评估报告
5. **evaluation_report.json** — 机器可读的指标数据

---

## 6. 常见问题

### Q1: 小时级训练太慢怎么办？
使用 `train_hourly.py` 的两阶段策略：先用 10% 子样本粗搜参数，再用全量训练。子样本使用系统抽样（每隔 k 行取一行），覆盖全年各季节避免分布偏移。

### Q2: 少数类 Recall 很低怎么办？
1. 确认已启用 `class_weight`（小时级用手动 `{Rain:5, Snow:5}`）
2. 调大 `min_samples_leaf` 可提升少数类召回（但可能降低整体精度）
3. 若 Recall 仍 <0.3，追加 SMOTE 过采样

### Q3: 日级标签如何验证正确性？
检查聚合后的分布是否匹配预期（Overcast 34%、Drizzle 32%、Rain 14.5%）。若 Rain/Snow 接近 0%，说明用了"众数"而非"最严重"聚合策略。

### Q4: 标准化数据对随机森林有影响吗？
无实质影响。树模型基于阈值分裂，不受量纲影响。但使用标准化数据可保持与模型 1/3 的管道一致性，feature_importance 数值仍是标准化后的。如需原始量纲，可从 `preprocessors/scaler.pkl` 逆变换。

### Q5: 如何与集成层对接？
```python
from models.model_2_random_forest.predict import RFPredictor

rf = RFPredictor(granularity="daily")
proba = rf.predict_proba(X)     # → [n, 6] 概率矩阵
weight = rf.get_weight()        # → F1/OOB float，供加权平均
importance = rf.get_feature_importance()  # → {feature: importance}
max_conf = rf.get_max_confidence(X)  # → 兜底层检查阈值
```
