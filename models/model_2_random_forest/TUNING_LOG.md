# 随机森林调优实验日志 (Tuning Log)

> 目标：通过多轮调优，从默认参数基线逐步提升 Macro-F1 和少数类 Recall
> 每次 Round 后记录修正内容 + 最终结果，最后选出最优模型

---

## 实验概览

| Round | 粒度 | 改进内容 | Macro-F1 | OOB | Rain Recall | Snow Recall | 耗时 |
|-------|------|----------|----------|-----|-------------|-------------|------|
| 0 (基线) | 日级 | 默认参数无搜索 | 0.3669 | 0.4275 | 0.4560 | 0.7171 | - |
| 1 | 日级 | HalvingGridSearch | 0.3714 | 0.4301 | 0.4468 | 0.7279 | 3.4 min |
| **2** | **日级** | **+小时级聚合特征(气压/湿度/云量)** | **0.4117** | **0.4728** | **0.5011** | 0.7161 | 4.6 min |
| 3 | 日级 | +细调(min_samples_split, ccp_alpha) | 0.4107 | 0.4629 | 0.5216 | 0.7466 | 13.6 min |
| 0 (基线) | 小时级 | 默认参数无搜索 | 0.6457 | 0.7179 | 0.6201 | 0.7468 | - |
| 1 | 小时级 | 10% Halving搜索+100树全量 | 0.6524 | 0.7281 | 0.5484 | 0.7390 | 16.7 min |
| **2** | **小时级** | **+细调(min_samples_split, max_features扩展)** | 0.6486 | **0.7315** | **0.5914** | **0.7494** | 26.2 min |

> **最优选择标准**：Macro-F1 主指标 + 少数类(Rain+Snow) Recall 加权(0.1) 作为决胜

---

## 详细记录

### Round 0 (基线)：默认参数，无 GridSearch

**参数**：n_estimators=200, max_depth=25, min_samples_leaf=2, max_features=sqrt, class_weight=balanced

**结果**：
| 指标 | 日级 | 小时级 |
|------|------|--------|
| Accuracy | 0.4058 | 0.7063 |
| Macro-F1 | 0.3669 | 0.6457 |
| Weighted-F1 | 0.4080 | 0.7086 |
| OOB Score | 0.4275 | 0.7179 |
| Rain Recall | 0.4560 | 0.6201 |
| Snow Recall | 0.7171 | 0.7468 |

**问题诊断**：
- 日级 Macro-F1 仅 0.37，best_params=null（未搜索超参）
- 日级数据缺失关键天气预测变量（气压/湿度/云量）

---

### Round 1 (日级)：HalvingGridSearchCV 超参搜索

**修正内容**：
1. 用 HalvingGridSearchCV 替代全量 GridSearchCV（逐轮减半搜索，81 组合→5 轮淘汰→1 个最优）
2. GridSearch 基模型关闭 oob_score（CV f1_macro 才是搜索指标），仅最终模型开启 OOB

**最优参数**：`{max_depth: None, max_features: log2, min_samples_leaf: 2, n_estimators: 200}`

**CV f1_macro**: 0.3306

**结果**：
| 指标 | 值 | vs 基线 |
|------|-----|---------|
| Accuracy | 0.4092 | +0.9% |
| Macro-F1 | 0.3714 | +1.2% |
| OOB Score | 0.4301 | +0.6% |
| Rain Recall | 0.4468 | -2.0% |
| Snow Recall | 0.7279 | +1.5% |

**分析**：超参搜索提升微弱。核心问题不在超参，而在特征——日级数据缺失气压/湿度/云量。

---

### Round 2 (日级)：小时级聚合特征增强 ⭐ 关键改进

**修正内容**：
1. **新增 `_aggregate_hourly_to_daily_features()`**：从小时级清洗数据按 city+date 聚合出日级补充特征
2. 新增 9 个关键天气预测变量：
   - `pressure_mean` / `pressure_min` — 气压均值/最低值（低气压系统=恶劣天气）
   - `humidity_mean` / `humidity_max` — 湿度均值/峰值（高湿=降水/阴天）
   - `cloud_cover_mean` / `cloud_cover_max` — 云量均值/峰值（直接决定阴/晴分类）
   - `wind_speed_mean` / `gust_max` — 平均风速/最大阵风
   - `wind_direction_mean` — 主导风向（circular mean via u/v 分量）
   - `today_weather_cat` — 当天最严重天气类别（编码为整数，当前天气是最强预测特征）
3. 为新特征增加滞后(lag_1/2/3)和滑动窗口(roll7)
4. 增加气压变化率 `pressure_mean_change_1`（前一天气压差）
5. 特征数从 40 → 50

**最优参数**：`{max_depth: None, max_features: log2, min_samples_leaf: 1, n_estimators: 300}`

**CV f1_macro**: 0.3824

**结果**：
| 指标 | 值 | vs 基线 | vs R1 |
|------|-----|---------|-------|
| Accuracy | 0.4564 | +12.5% | +11.5% |
| Macro-F1 | **0.4117** | **+12.2%** | **+10.9%** |
| Weighted-F1 | 0.4576 | +12.1% | +11.2% |
| OOB Score | 0.4728 | +10.6% | +9.9% |
| Rain Recall | 0.5011 | +9.9% | +12.2% |
| Snow Recall | 0.7161 | -0.1% | -1.6% |
| Log Loss | 1.2185 | -8.8% | -8.9% |

**分析**：特征增强带来显著提升！气压/湿度/云量是最强天气预测信号，日级数据之前完全缺失这些变量是效果差的根本原因。Macro-F1 从 0.37 提升到 0.41。

---

### Round 3 (日级)：细调 min_samples_split + ccp_alpha

**修正内容**：
1. 基于 R2 最优参数，扩展搜索空间：
   - 新增 `min_samples_split: [2, 5, 10]`
   - 新增 `ccp_alpha: [0.0, 0.001, 0.005]`（后剪枝）
   - 微调 n_estimators: [300, 400]
   - 微调 max_features: [log2, 0.5]

**最优参数**：`{ccp_alpha: 0.0, max_depth: None, max_features: log2, min_samples_leaf: 1, min_samples_split: 5, n_estimators: 400}`

**CV f1_macro**: 0.3863（+1.0% vs R2）

**结果**：
| 指标 | 值 | vs R2 |
|------|-----|-------|
| Accuracy | 0.4484 | -1.8% |
| Macro-F1 | 0.4107 | -0.2% |
| OOB Score | 0.4629 | -2.1% |
| Rain Recall | **0.5216** | **+4.1%** |
| Snow Recall | **0.7466** | **+4.3%** |
| Log Loss | 1.2301 | +1.0% |

**分析**：Macro-F1 微降但少数类 Recall 显著提升。ccp_alpha=0.0 说明后剪枝无效（数据量足够，树不需要额外剪枝）。min_samples_split=5 带来更好的少数类召回。综合评分后 R3 胜出（少数类 Recall 加权）。

---

### Round 1 (小时级)：10% 子样本 Halving 搜索 + 100 树全量训练

**修正内容**：
1. 10% 系统抽样子样本粗搜参数（386 万行全量搜索太慢）
2. 全量训练用 100 树（而非 300）控制运行时间在 ~17 分钟
3. 手动少数类加权 `class_weight={Rain:5, Snow:5}`

**最优参数**：`{max_depth: None, max_features: log2, min_samples_leaf: 4, n_estimators: 300}`
**实际训练用 100 树**

**CV f1_macro**: 0.3116

**结果**：
| 指标 | 值 | vs 基线 |
|------|-----|---------|
| Accuracy | 0.7111 | +0.7% |
| Macro-F1 | 0.6524 | +1.0% |
| OOB Score | 0.7281 | +1.4% |
| Rain Recall | 0.5484 | -11.6% |
| Snow Recall | 0.7390 | -1.0% |
| Log Loss | 0.7665 | -1.4% |

**分析**：小时级基线已经较强（预测下一小时比预测明天容易），超参搜索提升有限。Rain Recall 下降是因为搜索到的 min_samples_leaf=4 比基线的 2 更保守。

---

### Round 2 (小时级)：细调 min_samples_split + max_features 扩展

**修正内容**：
1. 基于 R1 最优参数扩展搜索空间：
   - 新增 `min_samples_split: [2, 5, 10]`
   - `max_features` 扩展: [log2, 0.3, 0.5]
   - `min_samples_leaf`: [2, 4, 8]
   - `max_depth`: [None, 30]
2. 全量训练用 150 树（比 R1 多 50 树，提升精度）

**最优参数**：`{max_depth: 30, max_features: log2, min_samples_leaf: 8, min_samples_split: 2, n_estimators: 200}`
**实际训练用 150 树**

**CV f1_macro**: 0.3157（+1.3% vs R1）

**结果**：
| 指标 | 值 | vs R1 |
|------|-----|-------|
| Accuracy | 0.7093 | -0.3% |
| Macro-F1 | 0.6486 | -0.6% |
| OOB Score | **0.7315** | **+0.5%** |
| Rain Recall | **0.5914** | **+7.8%** |
| Snow Recall | **0.7494** | **+1.4%** |
| Log Loss | **0.7623** | **-0.5%** |

**分析**：OOB 提升、Log Loss 降低、少数类 Recall 全部提升。min_samples_leaf=8 更保守但泛化更好。综合评分 R2 胜出。

---

## 最终选模结果

### 日级最优：Round 3

| 项目 | 值 |
|------|-----|
| 参数 | `ccp_alpha=0.0, max_depth=None, max_features=log2, min_samples_leaf=1, min_samples_split=5, n_estimators=400` |
| 特征数 | 50（含 9 个小时级聚合特征） |
| Macro-F1 | 0.4107 |
| OOB Score | 0.4629 |
| Rain Recall | 0.5216 |
| Snow Recall | 0.7466 |
| Log Loss | 1.2301 |
| 模型路径 | `saved_models/rf_daily.pkl` |

### 小时级最优：Round 2

| 项目 | 值 |
|------|-----|
| 参数 | `max_depth=30, max_features=log2, min_samples_leaf=8, min_samples_split=2, n_estimators=200 (实际150)` |
| 特征数 | 62 |
| Macro-F1 | 0.6486 |
| OOB Score | 0.7315 |
| Rain Recall | 0.5914 |
| Snow Recall | 0.7494 |
| Log Loss | 0.7623 |
| 模型路径 | `saved_models/rf_hourly.pkl` |

---

## 调优总结

### 关键发现

1. **特征工程 > 超参调优**：日级 Round 2 的特征增强（+12.2% Macro-F1）远超超参搜索的收益（+1.2%）。日级数据缺失气压/湿度/云量是效果差的根本原因。
2. **HalvingGridSearchCV 是有效的加速方案**：81 组合在 3-5 分钟内完成搜索，比全量 GridSearch 快 5-10x。
3. **后剪枝(ccp_alpha)无效**：数据量足够时（16 万+行），树不需要额外剪枝。
4. **少数类 Recall 与 Macro-F1 存在权衡**：更保守的 min_samples_leaf 提升少数类 Recall 但略降整体 F1。
5. **小时级基线已强**：预测下一小时比预测明天容易，超参调优提升空间有限。

### 从基线到最终的提升

| 指标 | 日级(基线→最终) | 提升 | 小时级(基线→最终) | 提升 |
|------|------------------|------|-------------------|------|
| Macro-F1 | 0.3669 → 0.4107 | +11.9% | 0.6457 → 0.6486 | +0.4% |
| OOB Score | 0.4275 → 0.4629 | +8.3% | 0.7179 → 0.7315 | +1.9% |
| Rain Recall | 0.4560 → 0.5216 | +14.4% | 0.6201 → 0.5914 | -4.6% |
| Snow Recall | 0.7171 → 0.7466 | +4.1% | 0.7468 → 0.7494 | +0.3% |
| Log Loss | 1.3352 → 1.2301 | -7.9% | 0.7773 → 0.7623 | -1.9% |
