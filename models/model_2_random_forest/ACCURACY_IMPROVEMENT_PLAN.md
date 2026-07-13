# 随机森林正确率提升方案报告 (Accuracy Improvement Plan)

> 项目：final2/weather — 欧洲城市天气预测系统
> 模型：Model 2 Random Forest
> 前提：舍弃速度限制，追求最高正确率
> 日期：2026-07-13

---

## 一、当前瓶颈诊断

### 1.1 日级当前表现（Round 3 最优）

| 指标 | 值 |
|------|-----|
| Accuracy | 0.4484 |
| Macro-F1 | 0.4107 |
| OOB Score | 0.4629 |
| Log Loss | 1.2301 |

逐类诊断（关键短板标红）：

| 类别 | Precision | Recall | F1 | Support | 诊断 |
|------|-----------|--------|-----|---------|------|
| Clear | 0.33 | 0.33 | 0.33 | 643 | 样本少(4.3%)，P 和 R 都低 |
| **Cloudy** | **0.20** | 0.27 | **0.23** | 928 | **最差类**，精确率仅 20% |
| Overcast | 0.58 | 0.47 | 0.52 | 6224 | 最大类，R 偏低(47%) |
| Drizzle | 0.48 | 0.37 | 0.42 | 5903 | R 偏低(37%)，漏报多 |
| Rain | 0.37 | 0.52 | 0.43 | 3169 | P 低(37%)，误报多 |
| Snow | 0.42 | 0.75 | 0.54 | 1018 | R 高但 P 低，过度预测 |

**日级核心问题**：
1. Cloudy 是最差类（F1=0.23），与 Overcast 本质相似，模型分不清
2. 整体 Log Loss=1.23，概率输出质量差，模型置信度不够
3. 日级只用了 50 个特征，其中时序增强（lag/rolling）只有 3 阶，气象信号挖掘不足

### 1.2 小时级当前表现（Round 2 最优）

| 指标 | 值 |
|------|-----|
| Accuracy | 0.7093 |
| Macro-F1 | 0.6486 |
| OOB Score | 0.7315 |
| Log Loss | 0.7623 |

逐类诊断：

| 类别 | Precision | Recall | F1 | Support | 诊断 |
|------|-----------|--------|-----|---------|------|
| Clear | 0.83 | 0.81 | 0.82 | 115214 | 良好 |
| **Cloudy** | **0.49** | **0.51** | **0.50** | 77784 | **最差类**，P 和 R 都最低 |
| Overcast | 0.78 | 0.79 | 0.79 | 163886 | 良好 |
| Drizzle | 0.65 | 0.57 | 0.61 | 59027 | R 偏低，漏报 |
| Rain | 0.43 | 0.59 | 0.49 | 9436 | P 低(43%)，误报多 |
| Snow | 0.63 | 0.75 | 0.69 | 5020 | 良好 |

**小时级核心问题**：
1. Cloudy 同样是最差类（F1=0.50），与 Clear/Overcast 边界模糊
2. Rain 精确率仅 43%，模型过度预测降雨（误报）
3. 全量训练只用了 150 棵树（受速度限制），OOB 未充分收敛

### 1.3 两粒度共同瓶颈

| 共同问题 | 日级影响 | 小时级影响 |
|----------|----------|------------|
| Cloudy 类表现最差 | F1=0.23 | F1=0.50 |
| Rain 误报多（P 低） | P=0.37 | P=0.43 |
| 概率质量差（Log Loss 高） | 1.23 | 0.76 |
| 树数量受速度限制 | 400 树 | 仅 150 树 |
| 滞后阶数不足 | lag 1-3 | lag 1-3 |

---

## 二、提升方案（共 8 项，按预期收益排序）

### 方案 1：全量树数 + 全量 GridSearch（预计 +2-4% Accuracy）

**问题**：当前树数受速度限制（日级 400、小时级仅 150），OOB 曲线未收敛。

**做法**：
```python
# 日级：n_estimators 搜索到 [300, 500, 800]
# 小时级：n_estimators 提升到 [200, 300, 500]，全量训练用 500 棵
PARAM_GRID_EXPANDED = {
    "n_estimators": [300, 500, 800],
    "max_depth": [None, 30, 50],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["log2", 0.3, 0.5],
    "min_samples_split": [2, 5, 10],
}
```

**理由**：OOB 误差曲线在树数增加时趋于收敛，更多树 = 方差更低 = 更稳定。当前小时级 150 树太少，OOB 可能未收敛。

**预期**：日级 +1-2%，小时级 +2-4%（树数从 150→500 的提升最显著）

---

### 方案 2：多步滞后 + 多尺度滑动窗口（预计 +3-5% Accuracy）

**问题**：当前滞后只用了 lag_1/2/3，窗口只用了 7 步。天气有日周期（24h）和季节周期（365d），3 阶滞后无法捕捉。

**做法**：
```python
# 小时级：增加 6/12/24 步滞后（覆盖日周期）
LAG_PERIODS_HOURLY = [1, 2, 3, 6, 12, 24]
LAG_COLS_HOURLY = [
    "temperature_2m", "pressure_msl", "relative_humidity_2m",
    "precipitation", "cloud_cover", "wind_speed_10m",
]
# 多尺度窗口：3h（短期变化）+ 12h（半日）+ 24h（全天）
ROLLING_WINDOWS_HOURLY = [3, 7, 12, 24]

# 日级：增加 7/14/30 步滞后（覆盖月周期）
LAG_PERIODS_DAILY = [1, 2, 3, 7, 14, 30]
LAG_COLS_DAILY = [
    "temperature_2m_mean", "precipitation_sum", "pressure_mean",
    "humidity_mean", "cloud_cover_mean", "today_weather_cat",
]
# 多尺度窗口：3d + 7d + 14d + 30d
ROLLING_WINDOWS_DAILY = [3, 7, 14, 30]
```

**理由**：
- 24 小时滞后让模型看到"昨天同一时刻"的天气，是最强的日周期信号
- 30 天滞后让模型看到月级别的气候趋势
- 多尺度窗口（3h+24h）同时捕捉短期变化和长期趋势

**预期**：日级 +2-3%（月周期信号），小时级 +3-5%（日周期信号是最强提升点）

---

### 方案 3：概率校准（CalibratedClassifierCV）（预计 Log Loss -15-25%）

**问题**：当前 Log Loss 日级 1.23、小时级 0.76，概率输出质量差。集成层做加权平均需要准确概率。

**做法**：
```python
from sklearn.calibration import CalibratedClassifierCV

# Platt Scaling（sigmoid）
rf_calibrated = CalibratedClassifierCV(rf_final, method="sigmoid", cv=5)
rf_calibrated.fit(X_train, y_train_enc)

# 或 Isotonic Regression（非参数，数据量大时更好）
rf_calibrated = CalibratedClassifierCV(rf_final, method="isotonic", cv=5)
```

**理由**：随机森林的概率偏向极端（接近 0 或 1），校准后概率更准确，直接提升 Log Loss 和集成层软投票质量。

**注意**：校准在训练集 CV 上做，不能在测试集校准（会泄露）。

**预期**：Log Loss 降低 15-25%，间接提升集成层 Accuracy +1-2%

---

### 方案 4：Cloudy 类专项处理（预计 Cloudy F1 +10-15%）

**问题**：Cloudy 在两个粒度都是最差类。Cloudy（WMO code 1/2）与 Clear（0）和 Overcast（3）边界模糊——"部分多云"本身就是主观判断。

**做法（三选一）**：

**4a. 合并 Cloudy→Overcast**（最激进）：
```python
# 将 6 类合并为 5 类：Cloudy 并入 Overcast
WMO_TO_CATEGORY_MERGED = {
    0: "Clear",
    1: "Overcast", 2: "Overcast",  # Cloudy → Overcast
    3: "Overcast",
    51: "Drizzle", 53: "Drizzle", 55: "Drizzle",
    61: "Rain", 63: "Rain", 65: "Rain",
    71: "Snow", 73: "Snow", 75: "Snow",
}
```
效果：消除最差类，剩余 5 类的边界更清晰。但牺牲了 Cloudy 这个类别。

**4b. Cloudy 专属加权**（保守）：
```python
# 在 class_weight 中给 Cloudy 额外加权
class_weight = {
    "Clear": 1, "Cloudy": 3, "Overcast": 1,  # Cloudy 加权
    "Drizzle": 1, "Rain": 5, "Snow": 5,
}
```

**4c. 补充 Cloudy 区分特征**：
```python
# Cloudy vs Clear 的区分：云量变化率（Cloudy 云量在 10-90% 波动）
df["cloud_cover_variability"] = df.groupby("city_id")["cloud_cover"].transform(
    lambda x: x.rolling(24).std())  # 24h 云量波动率
# Cloudy vs Overcast 的区分：云量趋势
df["cloud_trend"] = df.groupby("city_id")["cloud_cover"].transform(
    lambda x: x.diff(3))  # 3h 云量变化方向
```

**推荐 4c**：不合并类别，通过补充云量变化特征让模型更好区分。

**预期**：Cloudy F1 从 0.23→0.35（日级），0.50→0.60（小时级）

---

### 方案 5：特征交互增强（预计 +1-3% Accuracy）

**问题**：当前特征都是独立的，缺少气象学已知的重要交互效应。

**做法**：
```python
# 1. 温度×湿度交互（体感温度的精确版）
df["temp_humidity_index"] = df["temperature_2m"] * df["relative_humidity_2m"]

# 2. 气压×风向交互（低压系统+特定风向=特定天气）
df["pressure_wind_interaction"] = df["pressure_msl"] * df["wind_direction_mean"]

# 3. 季节×城市交互（不同城市不同季节的天气模式）
df["season_city"] = df["season"].astype(str) + "_" + df["city_id"].astype(str)
le = LabelEncoder()
df["season_city"] = le.fit_transform(df["season_city"])

# 4. 云量×辐射交互（云量高+辐射低=阴天确认）
df["cloud_rad_ratio"] = df["cloud_cover_mean"] / (df["shortwave_radiation_sum"] + 1)

# 5. 温度×气压交互（冷锋/暖锋识别）
df["temp_pressure"] = df["temperature_2m_mean"] * df["pressure_mean"]
```

**理由**：随机森林能自动捕捉交互，但显式构造已知强交互特征可以加速学习、提升精度。

**预期**：+1-3%，主要提升 Overcast 和 Cloudy 的区分

---

### 方案 6：特征选择去噪（预计 +1-2% Accuracy）

**问题**：当前日级 50 个特征、小时级 62 个特征，其中包含噪声列。

**做法**：
```python
# 1. 从 feature_importances_ 筛选 Top-30
importances = rf_final.feature_importances_
threshold = np.sort(importances)[-30]  # 第 30 大的重要性
selected = [f for f, imp in zip(feature_names, importances) if imp >= threshold]

# 2. 删除常量列（测试集 year 恒为 2024）
constant_cols = [c for c in df.columns if df[c].nunique() == 1]

# 3. 删除重要性 < 0.001 的噪声列
noise_cols = [f for f, imp in zip(feature_names, importances) if imp < 0.001]

# 用筛选后的特征重新训练
rf_final.fit(X_train_selected, y_train_enc)
```

**理由**：噪声特征会干扰树的分裂决策，RF 虽然抗噪但不代表噪声无害。

**预期**：+1-2%，主要减少 Overcast 的误分类

---

### 方案 7：SMOTE 过采样少数类（预计 Rain/Snow Recall +5-10%）

**问题**：小时级 Rain/Snow 仅 1.7%，当前用手动加权 {Rain:5, Snow:5}，但 Recall 仍不够高。

**做法**：
```python
from imblearn.over_sampling import SMOTE

# 仅对训练集过采样，不能对测试集
smote = SMOTE(random_state=42, sampling_strategy={
    "Rain": 50000,   # 从 6.6 万提升到 5 万（小时级训练集）
    "Snow": 50000,
})
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
```

**注意**：SMOTE 在标准化数据上效果更好；需要安装 `imbalanced-learn`。

**预期**：Rain Recall +5-10%，Snow Recall +3-5%，但可能牺牲少量 Precision

---

### 方案 8：集成增益（预计 +2-4% Accuracy）

**问题**：单模型有固有上限，与其他模型集成可以互补错误。

**做法**：
```python
# 与模型 1（逻辑回归）和模型 3（Wide & Deep）加权平均
from sklearn.ensemble import VotingClassifier

# 软投票：各模型 predict_proba 加权平均
ensemble = VotingClassifier(
    estimators=[
        ("lr", lr_model),       # 线性基准
        ("rf", rf_final),        # 非线性强规则
        ("dl", dl_model),        # 高维泛化
    ],
    voting="soft",
    weights=[lr_f1, rf_f1, dl_f1],  # 按 F1 加权
)
```

**理由**：RF 在 Cloudy 上弱，DL 可能在 Cloudy 上更强；RF 在 Rain Recall 上强，LR 可能在 Precision 上更强。互补。

**注意**：这是集成层（Day7）任务，但可以提前在 RF 内部做 ExtraTrees + RandomForest 的集成。

**预期**：+2-4%，主要提升最差的 Cloudy 类

---

## 三、方案优先级与预期收益

### 日级提升路线（当前 Accuracy 0.4484）

| 优先级 | 方案 | 预期 Accuracy | 累计预期 |
|--------|------|---------------|----------|
| 1 | 方案 2（多步滞后+多窗口） | +3-5% | 0.48-0.50 |
| 2 | 方案 1（全量树数 800） | +1-2% | 0.49-0.52 |
| 3 | 方案 5（特征交互） | +1-3% | 0.50-0.55 |
| 4 | 方案 4c（Cloudy 专项特征） | +1-2% | 0.51-0.57 |
| 5 | 方案 6（特征选择） | +1-2% | 0.52-0.59 |
| 6 | 方案 3（概率校准） | Log Loss -20% | 间接提升 |

**日级预期目标**：Accuracy 0.50-0.58（当前 0.4484，提升 12-30%）

### 小时级提升路线（当前 Accuracy 0.7093）

| 优先级 | 方案 | 预期 Accuracy | 累计预期 |
|--------|------|---------------|----------|
| 1 | 方案 1（全量树数 500） | +2-4% | 0.73-0.75 |
| 2 | 方案 2（24h 滞后+多窗口） | +3-5% | 0.76-0.80 |
| 3 | 方案 4c（Cloudy 专项特征） | +1-2% | 0.77-0.82 |
| 4 | 方案 3（概率校准） | Log Loss -20% | 间接提升 |
| 5 | 方案 7（SMOTE） | Rain/Snow R +5-10% | 间接提升 |
| 6 | 方案 8（集成） | +2-4% | 集成层任务 |

**小时级预期目标**：Accuracy 0.76-0.82（当前 0.7093，提升 7-16%）

---

## 四、实施建议

### 第一批（收益最高，先做）

1. **方案 2（多步滞后+多窗口）** — 代码改动在 `config.py` + `data_loader.py`，不影响架构
2. **方案 1（全量树数）** — 只改 `PARAM_GRID` 的 n_estimators 范围，小时级全量训练用 500 树

### 第二批（针对性补强）

3. **方案 4c（Cloudy 专项特征）** — 在 `data_loader.py` 增加云量变化率特征
4. **方案 5（特征交互）** — 在 `data_loader.py` 增加交互特征
5. **方案 6（特征选择）** — 训练后用 feature_importances_ 筛选，重新训练

### 第三批（概率质量 + 集成）

6. **方案 3（概率校准）** — 加 `CalibratedClassifierCV` 包装
7. **方案 7（SMOTE）** — 安装 imbalanced-learn，仅训练集过采样
8. **方案 8（集成）** — 集成层任务，可与 RF 内部 ExtraTrees 先做小集成

---

## 五、风险与注意事项

| 风险 | 说明 | 对策 |
|------|------|------|
| 过拟合 | 全量树数 + 更多特征可能过拟合 | 监控 OOB vs 测试集差距，>5% 时增加 min_samples_leaf |
| Cloudy 合并争议 | 合并 Cloudy→Overcast 改变业务语义 | 优先用方案 4c（补特征）而非 4a（合并） |
| SMOTE 副作用 | 过采样可能引入合成噪声样本 | 仅对训练集，用 SMOTE 而非随机复制 |
| 训练时间 | 全量 GridSearch + 800 树 + SMOTE 可能需数小时 | 分批次执行，每批记录 JSON 结果 |
| 概率校准泄露 | 校准不能用测试集 | 用 `cv=5` 内部交叉校准，或从训练集切分校准集 |
