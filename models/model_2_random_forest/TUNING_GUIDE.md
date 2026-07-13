# 随机森林技术选型与调优指南 (RF Technical & Tuning Guide)

> 模型：Model 2 Random Forest  
> 项目：final2/weather — 欧洲城市天气预测系统  
> 双粒度：日级 + 小时级，均为 6 类天气分类  
> 日期：2026-07-12

---

## 一、当前技术选型

### 1.1 算法选型理由

| 决策点 | 选型 | 理由 |
|--------|------|------|
| **算法** | RandomForestClassifier | 结构化数据的"常青树"，自动捕捉非线性交互，自带 OOB 无偏评估和 feature_importances_ 可解释性 |
| **任务类型** | 多分类（6 类） | 业务要求预测天气类别（Clear/Cloudy/Overcast/Drizzle/Rain/Snow），非二分类 |
| **预测方向** | 时序偏移 shift(-1) | 用当前时刻特征预测下一时刻天气，严格防数据泄露 |
| **标准化** | 直接用标准化后数据 | RF 基于阈值分裂，对量纲不敏感；但保持管道一致性 |
| **特征工程** | 滞后 + 滑动窗口 + 气压变化率 | 天气强时序依赖，原始瞬时值不足以预测未来 |

### 1.2 已实现的技术路线

```
特征工程产出 CSV
    ↓
data_loader.py
    ├── 小时级：weather_code → 6 类 → shift(-1) 标签
    ├── 日级：小时级清洗数据按"最严重天气"聚合 → join → shift(-1)
    ├── 滞后特征（lag_1/2/3）
    ├── 滑动窗口（roll7_mean / roll7_std）
    ├── 气压变化率（pressure_change_3h）
    └── 字符串列编码（season/day_period → 整数）
    ↓
train_*.py
    ├── GridSearchCV(TimeSeriesSplit, f1_macro)  ← 超参搜索
    ├── class_weight 加权                      ← 不平衡处理
    ├── oob_score=True                         ← 无偏评估
    └── 小时级：10% 子样本粗搜 → 全量训练       ← 工程优化
    ↓
evaluate.py → outputs/{daily,hourly}/
    ├── evaluation_report.json/md  ← 指标
    ├── confusion_matrix.png       ← 混淆矩阵
    ├── feature_importance.png     ← 特征重要性
    └── probability_distribution.png
    ↓
predict.py → RFPredictor（供集成层调用）
```

### 1.3 当前超参数配置

```python
# 搜索网格
PARAM_GRID = {
    "n_estimators":     [100, 200, 300],
    "max_depth":        [15, 25, None],
    "min_samples_leaf": [1, 2, 4],
    "max_features":     ["sqrt", "log2", 0.3],
}

# 固定参数
RF_FIXED_PARAMS = {
    "random_state": 42,
    "n_jobs": -1,
    "class_weight": "balanced",
    "oob_score": True,
    "bootstrap": True,
}
```

### 1.4 当前实测结果（默认参数基线）

| 指标 | 日级 | 小时级 |
|------|------|--------|
| Accuracy | 0.4058 | 0.7063 |
| Macro-F1 | 0.3669 | 0.6457 |
| Weighted-F1 | 0.4080 | 0.7086 |
| OOB Score | 0.4275 | 0.7179 |
| Cohen's Kappa | 0.2228 | 0.5996 |
| Log Loss | 1.3352 | 0.7773 |
| Rain Recall | 0.4560 | 0.6201 |
| Snow Recall | 0.7171 | 0.7468 |

> 注：以上为默认参数（未跑 GridSearch）的基线结果。小时级已用 `class_weight={Rain:5,Snow:5}` 手动加权。

---

## 二、随机森林可调优方法全景

### 2.1 超参数调优（模型层）

随机森林的核心超参数分为 **控制树结构** 和 **控制森林多样性** 两类：

| 超参数 | 作用方向 | 调大效果 | 调小效果 | 推荐范围 | 当前值 |
|--------|----------|----------|----------|----------|--------|
| **n_estimators** | 森林规模 | 更稳定、方差更低，但收益递减、训练更慢 | 更快但不稳定 | 100-500 | 100/200/300 |
| **max_depth** | 单树深度 | 更深→拟合能力更强→过拟合风险↑ | 更浅→欠拟合但泛化好 | 10-30/None | 15/25/None |
| **min_samples_leaf** | 叶节点最小样本 | 更大→更保守→抗过拟合→但可能欠拟合 | 更小→更精细→过拟合风险↑ | 1-10 | 1/2/4 |
| **min_samples_split** | 分裂最小样本 | 更大→更保守 | 更小→更激进 | 2-20 | 未搜索 |
| **max_features** | 每次分裂特征数 | 更大→树间相关性↑→多样性↓ | 更小→多样性↑→偏差↑ | sqrt/log2/0.1-0.5 | sqrt/log2/0.3 |
| **max_leaf_nodes** | 最大叶节点数 | 限制树复杂度（替代 max_depth） | — | None/100-1000 | 未用 |
| **min_impurity_decrease** | 最小信息增益 | 更大→更保守，只保留显著分裂 | 更小→分裂更多 | 0-0.1 | 未用 |
| **criterion** | 分裂准则 | gini（默认，快）/ entropy / log_loss | — | gini/entropy | gini |
| **ccp_alpha** | 后剪枝 | >0 启用成本复杂度剪枝 | — | 0-0.02 | 未用 |

#### 调优策略建议

**第一步：粗搜 n_estimators + max_depth**
```python
# 这两个参数影响最大，先确定大方向
PARAM_GRID_COARSE = {
    "n_estimators": [100, 200, 300],
    "max_depth": [15, 25, None],
}
```

**第二步：细搜正则化参数**
```python
# 固定第一步最优，精调控制过拟合的参数
PARAM_GRID_FINE = {
    "min_samples_leaf": [1, 2, 4, 8],
    "min_samples_split": [2, 5, 10],
    "max_features": ["sqrt", "log2", 0.3, 0.5],
}
```

**第三步：后剪枝微调（可选）**
```python
# 用 cost-complexity pruning 做最后微调
PARAM_GRID_PRUNE = {
    "ccp_alpha": [0.0, 0.001, 0.005, 0.01],
}
```

### 2.2 数据层调优（特征工程）

| 方法 | 说明 | 当前状态 |
|------|------|----------|
| **滞后特征** | temp/pressure/humidity 前 1-3 步的值 | ✅ 已实现 |
| **滑动窗口统计** | roll7_mean / roll7_std | ✅ 已实现 |
| **气压变化率** | pressure_change_3h（暴雨前兆） | ✅ 已实现 |
| **交互特征** | season×city_id、hour×wind_direction | ❌ 可补充 |
| **特征选择** | 用 feature_importances_ 筛选 Top-N，剔除噪声列 | ⚠️ 可视化已有，未自动筛选 |
| **删除常量列** | 测试集中 year 可能恒为 2024 | ❌ 可补充 |
| **更丰富的滞后** | 扩展到 lag_6/12/24（小时级多步滞后） | ❌ 可补充 |
| **多窗口滑动** | 3h/12h/24h 多尺度窗口 | ❌ 可补充 |

### 2.3 类别不平衡调优（7 种方法）

> 小时级 Rain/Snow 仅 1.7%，日级聚合后 7%-15%。

| # | 方法 | 实现方式 | 当前状态 | 适用场景 |
|---|------|----------|----------|----------|
| 1 | `class_weight="balanced"` | 自动按反频率加权 | ✅ 日级使用 | 默认首选 |
| 2 | 手动加大权重 | `class_weight={Rain:5,Snow:5}` | ✅ 小时级使用 | balanced 不够时递进 |
| 3 | SMOTE 过采样 | `imblearn.over_sampling.SMOTE` | ❌ 未实现 | 极端不平衡，权重不够时 |
| 4 | 日级聚合放大 | "最严重天气"聚合策略 | ✅ 已实现 | 纯数据工程，Rain 1.8%→14.5% |
| 5 | Macro-F1 调参 | `scoring="f1_macro"` | ✅ 已使用 | 让调参过程优化少数类 |
| 6 | 子采样+全量两阶段 | 10% 粗搜 → 全量训练 | ✅ 小时级使用 | 大数据集加速 |
| 7 | 集成增益 | 3 模型加权平均互补 | ⏳ 集成层实现 | 单模型召回不足时互补 |

**不平衡调优递进路线**：
```
balanced (默认)
    ↓ Rain/Snow Recall < 0.3?
手动加权 {Rain:5, Snow:5}
    ↓ 仍 < 0.3?
SMOTE 过采样
    ↓ 仍不够?
调整 min_samples_leaf ↑（提升少数类召回）
```

### 2.4 概率校准调优

随机森林的概率输出偏向极端（接近 0 或 1），不够校准，影响集成层软投票质量：

| 方法 | 实现 | 适用 |
|------|------|------|
| **Platt Scaling** | `CalibratedClassifierCV(method="sigmoid")` | 二分类，但多分类也可用 |
| **Isotonic Regression** | `CalibratedClassifierCV(method="isotonic")` | 非参数，数据量大时更好 |
| **预拟合校准** | `cv="prefit"` + 独立验证集 | 模型已训练好，节省时间 |

```python
from sklearn.calibration import CalibratedClassifierCV
rf_calibrated = CalibratedClassifierCV(rf_final, method="sigmoid", cv=5)
# 注意：不能在测试集校准（会泄露）
```

### 2.5 集成多样性调优

为了让综合研判有价值，RF 应与其他模型产生**不同错误模式**：

| 差异化策略 | 说明 |
|------------|------|
| RF 用全部特征 | DL 用精炼特征 → 不同视角 |
| 时序增强特征 | RF 独有滞后/窗口特征，DL 用 Embedding |
| 不同的 max_features | RF 用 sqrt（多样性高），其他模型用全量 |

---

## 三、如何判断模型好坏（评估参数详解）

### 3.1 整体指标

| 指标 | 含义 | 判断标准 | 当前值(小时级) |
|------|------|----------|----------------|
| **Accuracy** | 整体预测正确率 | >0.7 良好；但类别不平衡时会误导 | 0.7063 ✅ |
| **Macro-F1** | 6 类 F1 的算术平均（各类等权） | **最重要指标**，>0.6 良好 | 0.6457 ✅ |
| **Weighted-F1** | 按各类样本量加权的 F1 | 反映整体实际表现 | 0.7086 ✅ |
| **Cohen's Kappa** | 去除随机一致性的达成度 | >0.6 良好；>0.8 优秀 | 0.5996 ⚠️ |
| **OOB Score** | 袋外无偏泛化估计 | 应接近测试集 Accuracy | 0.7179 ✅ |
| **Log Loss** | 概率预测的负对数似然 | <0.8 良好；越小越好 | 0.7773 ✅ |

> **为什么 Macro-F1 最重要**：6 类中 Rain/Snow 仅 1.7%，Accuracy 会被 Overcast(33%) 主导。Macro-F1 对每个类别等权，能反映少数类表现。

### 3.2 逐类指标（关键诊断）

| 指标 | 含义 | 诊断意义 |
|------|------|----------|
| **Precision** (精确率) | 预测为该类的准确率 | 低=误报多（模型过度预测该类） |
| **Recall** (召回率) | 该类被正确识别的比例 | 低=漏报多（模型忽略该类） |
| **F1-Score** | P 和 R 的调和平均 | 综合衡量 |
| **Support** | 测试集中该类实际样本数 | 判断类别是否过少 |

**诊断逻辑**：

```
某类 Recall 高但 Precision 低
    → 模型过度预测该类（误报）
    → 降低该类的 class_weight 或提高 min_samples_leaf

某类 Recall 低
    → 模型漏报该类
    → 增加该类权重，或降低 min_samples_leaf

某类 P 和 R 都低
    → 特征区分度不足或样本太少
    → 补充特征或使用 SMOTE
```

### 3.3 可视化诊断

| 图表 | 文件 | 诊断用途 |
|------|------|----------|
| **混淆矩阵** | `confusion_matrix.png` | 看哪些类别被混淆（如 Cloudy↔Overcast） |
| **特征重要性** | `feature_importance.png` | 哪些气象指标最重要（答辩可解释性） |
| **概率分布** | `probability_distribution.png` | 模型置信度分布，兜底阈值设定依据 |
| **OOB 误差曲线** | `oob_error_curve.png` | 模型何时收敛，最优树数量 |

### 3.4 可靠性验证项

| 验证项 | 方法 | 判断标准 |
|--------|------|----------|
| **数据泄露检查** | 训练集时间≤2023，测试集=2024 | 训练集不含 2024 数据 |
| **OOB vs 测试集** | 比较 OOB Score 和测试 Accuracy | 差距 <5% 说明泛化稳定 |
| **CV 各折稳定性** | 看 GridSearch 各折分数方差 | 方差小=模型稳定 |
| **训练/测试分布一致性** | 比较标签分布 | 分布一致=无采样偏差 |
| **少数类达成度** | Rain/Snow Recall | >0.3 为最低目标 |

---

## 四、调优决策树（实操路线）

```
当前基线结果
    │
    ├── Macro-F1 < 0.5?
    │   ├── 是 → 跑完整 GridSearchCV（当前 best_params=null）
    │   │         ↓
    │   │      仍低？→ 检查特征工程（补充交互特征、多步滞后）
    │   │
    │   └── 否 → 进入下一步
    │
    ├── 少数类 Recall < 0.3?
    │   ├── 是 → 递增加权 → SMOTE → 调 min_samples_leaf↑
    │   └── 否 → 进入下一步
    │
    ├── OOB 与测试集差距 > 5%?
    │   ├── 是 → 过拟合 → 降 max_depth / 升 min_samples_leaf
    │   └── 否 → 泛化稳定
    │
    ├── 概率校准差（Log Loss 高）?
    │   └── 加 CalibratedClassifierCV
    │
    └── Cloudy↔Overcast 混淆严重?
        └── 这两类本质相似，可考虑合并或加 cloud_cover 变化率特征
```

---

## 五、关键参数速查表

| 你想... | 调什么 | 方向 |
|---------|--------|------|
| 提升整体准确率 | n_estimators ↑ / max_depth ↑ | 增强拟合能力 |
| 减少过拟合 | max_depth ↓ / min_samples_leaf ↑ / ccp_alpha ↑ | 增强正则化 |
| 提升少数类召回 | class_weight 少数类↑ / min_samples_leaf ↑ | 加权+保守分裂 |
| 提升概率质量 | CalibratedClassifierCV | 校准概率 |
| 加速训练 | n_estimators ↓ / 子采样 | 减少计算量 |
| 增加树间多样性 | max_features ↓（如 0.1） | 每棵树看不同特征 |
| 判断何时停止加树 | OOB 误差曲线收敛点 | n_estimators = 收敛点 |

---

## 六、与项目其他模型的差异化定位

| 维度 | 模型1 逻辑回归 | **模型2 随机森林** | 模型3 深度学习 |
|------|----------------|---------------------|----------------|
| 范式 | 线性 | **非线性+集成** | 非线性+梯度优化 |
| 特征使用 | 精炼特征 | **全部特征+时序增强** | 精炼特征+Embedding |
| 不平衡处理 | class_weight | **class_weight+SMOTE+聚合** | weighted loss |
| 评估方式 | CV | **OOB+CV** | train/val/test |
| 可解释性 | 系数 | **feature_importances_** | SHAP（事后） |
| 概率质量 | 天然校准 | **需校准** | 需校准 |
| 集成角色 | 线性基准 | **非线性强规则** | 高维泛化 |

> 集成层使用 `predict.py` 的 `predict_proba`（[n,6] 概率矩阵）+ `get_weight()`（OOB Score 作为权重参考值）进行加权平均。
