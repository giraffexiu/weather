# 随机森林技术选型与调优指南 (RF Technical & Tuning Guide)

> 模型：Model 2 Random Forest  
> 项目：final2/weather — 欧洲城市天气预测系统  
> 粒度：小时级多目标回归（5 变量连续值预测）  
> 日期：2026-07-12（更新：2026-07-14，移除日级）

---

## 一、当前技术选型

### 1.1 算法选型理由

| 决策点 | 选型 | 理由 |
|--------|------|------|
| **算法** | RandomForestRegressor | 结构化数据的"常青树"，自动捕捉非线性交互，自带 OOB 无偏评估和 feature_importances_ 可解释性 |
| **任务类型** | 多目标回归（5 变量） | 预测下一小时的温度/降水/风速/体感温度/相对湿度连续值 |
| **预测方向** | 时序偏移 shift(-1) | 用当前时刻特征预测下一时刻值，严格防数据泄露 |
| **标准化** | 直接用标准化后数据 | RF 基于阈值分裂，对量纲不敏感；但保持管道一致性 |
| **特征工程** | 滞后 + 滑动窗口 + 气压变化率 | 天气强时序依赖，原始瞬时值不足以预测未来 |

### 1.2 已实现的技术路线

```
特征工程产出 CSV
    ↓
data_loader.py
    ├── 小时级：5 变量 shift(-1) 标签 = 下一小时值
    ├── 重切分：2023 数据移入测试集（train=2015~2022, test=2023~2024）
    ├── precipitation log1p 变换（长尾→近似正态）
    ├── 滞后特征（lag_1/2/3/6/12/24/48）
    ├── 多尺度滑动窗口（roll3/7/12/24/48_mean/std）
    ├── 气压变化率（pressure_change_1/3/6）
    ├── 云量趋势特征
    ├── 交互特征（temp_humidity / pressure_wind / cloud_rad_ratio）
    └── 字符串列编码（season/day_period → 整数）
    ↓
train_hourly.py
    ├── GridSearchCV(TimeSeriesSplit, neg_MSE)  ← 超参搜索（可选 --no-grid 跳过）
    ├── 10% 子样本粗搜 → 全量训练               ← 两阶段优化
    ├── oob_score=True                         ← 无偏评估
    └── joblib(compress=3) 保存                ← 压缩存储
    ↓
evaluate.py → outputs/hourly/
    ├── evaluation_report.json/md  ← 指标
    └── feature_importance.png     ← 特征重要性
    ↓
predict.py → RFPredictor（供集成层调用）
```

### 1.3 当前超参数配置

```python
# 搜索网格
PARAM_GRID = {
    "n_estimators":     [200, 300],
    "max_depth":        [20, 30, None],
    "min_samples_leaf": [2, 4],
    "max_features":     ["log2", 0.3],
}

# 固定参数（回归专用）
RF_FIXED_PARAMS = {
    "random_state": 42,
    "n_jobs": -1,
    "oob_score": True,
    "bootstrap": True,
}

# 已知最优参数（--no-grid 使用）
best_params = {
    "n_estimators": 300, "max_depth": 30,
    "min_samples_leaf": 2, "max_features": 0.3,
}
```

### 1.4 当前实测结果

| 指标 | 值 |
|------|-----|
| Overall R² | 0.8397 |
| Overall RMSE | 0.3318 |
| Overall MAE | 0.1393 |
| OOB Score | 0.9024 |

---

## 二、随机森林可调优方法全景

### 2.1 超参数调优（模型层）

| 超参数 | 作用方向 | 调大效果 | 调小效果 | 推荐范围 | 当前值 |
|--------|----------|----------|----------|----------|--------|
| **n_estimators** | 森林规模 | 更稳定、方差更低，但收益递减、训练更慢 | 更快但不稳定 | 100-500 | 200/300 |
| **max_depth** | 单树深度 | 更深→拟合能力更强→过拟合风险↑ | 更浅→欠拟合但泛化好 | 10-30/None | 20/30/None |
| **min_samples_leaf** | 叶节点最小样本 | 更大→更保守→抗过拟合 | 更小→更精细→过拟合风险↑ | 1-10 | 2/4 |
| **max_features** | 每次分裂特征数 | 更大→树间相关性↑→多样性↓ | 更小→多样性↑→偏差↑ | sqrt/log2/0.1-0.5 | log2/0.3 |

### 2.2 数据层调优（特征工程）

| 方法 | 说明 | 当前状态 |
|------|------|----------|
| **滞后特征** | temp/pressure/humidity 前 1-48 步的值 | ✅ 已实现（lag_1~48） |
| **多尺度滑动窗口** | roll3/7/12/24/48 mean+std | ✅ 已实现 |
| **气压变化率** | pressure_change_1/3/6（暴雨前兆） | ✅ 已实现 |
| **交互特征** | temp_humidity / pressure_wind / cloud_rad_ratio | ✅ 已实现 |
| **特征选择** | 用 feature_importances_ 筛选 Top-N | ⚠️ 可视化已有，未自动筛选 |

### 2.3 集成多样性调优

| 差异化策略 | 说明 |
|------------|------|
| RF 用全部特征 | DL 用精炼特征 → 不同视角 |
| 时序增强特征 | RF 独有滞后/窗口特征，DL 用 Embedding |
| 不同的 max_features | RF 用 0.3（多样性高），其他模型用全量 |

---

## 三、如何判断模型好坏（评估参数详解）

### 3.1 整体指标

| 指标 | 含义 | 判断标准 | 当前值 |
|------|------|----------|--------|
| **Overall R²** | 5 目标平均决定系数 | >0.8 良好 | 0.8397 ✅ |
| **Overall RMSE** | 5 目标平均均方根误差 | 越小越好 | 0.3318 |
| **Overall MAE** | 5 目标平均绝对误差 | 越小越好 | 0.1393 |
| **OOB Score** | 袋外无偏泛化估计 | 应接近测试 R² | 0.9024 ✅ |

### 3.2 逐目标指标

| 目标 | R² | 诊断 |
|------|-----|------|
| temperature_2m | 0.9938 | ✅ 极强 |
| apparent_temperature | 0.9937 | ✅ 极强 |
| relative_humidity_2m | 0.9554 | ✅ 良好 |
| wind_speed_10m | 0.9149 | ✅ 良好 |
| precipitation | 0.3409 | ⚠️ 弱项，长尾分布难预测 |

> precipitation R² 仅 0.34 是整体瓶颈。建议集成层对降水单独加权降权或增强兜底处理。

---

## 四、调优决策树（实操路线）

```
当前基线结果 (R²=0.8397)
    │
    ├── precipitation R² < 0.5?
    │   └── 是 → 独立训练降水模型 / 集成层单独加权 / 兜底增强
    │
    ├── OOB vs 测试 R² 差距 > 5%?
    │   ├── 是 → 过拟合 → 降 max_depth / 升 min_samples_leaf
    │   └── 否 → 泛化稳定 ✅
    │
    ├── 训练太慢?
    │   └── 用 --no-grid 跳过 GridSearch / 降 n_estimators
    │
    └── 模型体积太大?
        └── 降 n_estimators / 提高 compress / 用 mmap_mode 加载
```

---

## 五、关键参数速查表

| 你想... | 调什么 | 方向 |
|---------|--------|------|
| 提升整体 R² | n_estimators ↑ / max_depth ↑ | 增强拟合能力 |
| 减少过拟合 | max_depth ↓ / min_samples_leaf ↑ | 增强正则化 |
| 提升降水预测 | 独立训练 + 单独调参 | 降水与其他目标难度差异大 |
| 加速训练 | n_estimators ↓ / 子采样 | 减少计算量 |
| 增加树间多样性 | max_features ↓ | 每棵树看不同特征 |
| 判断何时停止加树 | OOB 误差曲线收敛点 | n_estimators = 收敛点 |
| 减小模型体积 | n_estimators ↓ / compress ↑ | 压缩存储 |

---

## 六、与项目其他模型的差异化定位

| 维度 | 模型1 线性回归 | **模型2 随机森林** | 模型3 深度学习 |
|------|----------------|---------------------|----------------|
| 范式 | 线性 | **非线性+集成** | 非线性+梯度优化 |
| 特征使用 | 精炼特征 | **全部特征+时序增强** | 精炼特征+Embedding |
| 评估方式 | CV | **OOB+CV** | train/val/test |
| 可解释性 | 系数 | **feature_importances_** | SHAP（事后） |
| 集成角色 | 线性基准 | **非线性强规则** | 高维泛化 |

> 集成层使用 `predict.py` 的 `predict()`（[n,5] 预测值）+ `get_weight()`（OOB Score 作为权重参考值）进行加权平均。
