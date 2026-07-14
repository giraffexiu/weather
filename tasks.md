# tasks.md — model_2_random_forest 开发与集成任务追踪

> 项目：final2/weather — 欧洲城市天气预测系统
> 角色：Lead Agent（主导协调）
> 创建时间：2026-07-14
> 分支策略：`main` ← 已有进度；`random_forest` ← 已推送的开发分支

---

## Phase 1：信息收集与环境搭建

### 1.1 任务审查：阅读 `任务.md`
- [x] 读取并分析 `weather/任务.md`
- [x] 确认项目范围：三模型训练（线性/随机森林/深度学习）→ 综合研判机制 → 兜底方案 → Gradio 可视化部署
- [x] 确认 model_2 定位：非线性强规则捕捉，OOB 无偏评估，feature_importances_ 可解释性

### 1.2 模型验证：测试 `random_forest` 模型
- [x] 确认已训练模型文件：`models/model_2_random_forest/saved_models/rf_hourly.pkl`
- [x] 确认模型训练参数：n_estimators=300, max_depth=30, min_samples_leaf=2, max_features=0.3
- [x] 确认 OOB Score=0.9024, Test Overall R²=0.8397（小时级多目标回归）
- [x] 记录预测执行流程（见下方"预测执行流程"）
- [x] ⚠️ **关键发现**：`rf_hourly.pkl` 体积 **21.4 GB**，加载时触发 `MemoryError`（系统 31GB RAM，可用约 21GB）

#### 预测执行流程（model_2_random_forest）

```bash
# 1. 环境准备
cd final2/weather
pip install -r requirements.txt   # sklearn>=1.3, pandas>=2.0, numpy>=1.24, joblib

# 2. 模型验证（仅加载元信息，不执行预测）
cd models/model_2_random_forest
python predict.py
# 预期输出：目标变量、log1p 状态、OOB Score、特征数、Top-5 特征
# ⚠️ 当前因模型 21.4GB 超出内存触发 MemoryError

# 3. 编程式调用（供集成层使用）
python -c "from predict import RFPredictor; p = RFPredictor(); print(p.predict_hourly(X))"
```

**注意**：当前 `rf_hourly.pkl` 因体积过大无法在本机直接加载。需用 `mmap_mode='r'` 或减小模型体积（降低 n_estimators / compress 级别）后重训。

#### 模型评估摘要（来自 outputs/hourly/evaluation_report.json）

| 目标变量 | RMSE | MAE | R² |
|---------|------|-----|-----|
| temperature_2m | 0.0784 | 0.0548 | 0.9938 |
| precipitation (log还原) | 0.9929 | 0.2276 | 0.3409 |
| wind_speed_10m | 0.3001 | 0.2142 | 0.9149 |
| apparent_temperature | 0.0796 | 0.0562 | 0.9937 |
| relative_humidity_2m | 0.2079 | 0.1436 | 0.9554 |
| **Overall** | **0.3318** | **0.1393** | **0.8397** |

> precipitation 的 R² 仅 0.34，长尾分布预测仍是弱项，建议集成层对降水单独加权或兜底处理。

---

## Phase 2：执行工作流

### Task Group A：仓库与版本控制

#### Task A1：Git 卫生 ✅ 完成
- [x] 识别大文件（见下方"大文件清单"）
- [x] 确认 `.gitignore` 已覆盖所有大文件类型（`*.pkl`, `*.csv`, `saved_models/`, `outputs/`）
- [x] 确认工作树清洁（`git status` 无未提交变更）
- [x] 确认大文件均未被 git 跟踪（`git check-ignore` 验证通过）
- [x] 提交 `tasks.md` 到本地仓库（commit 56369f4）

#### 大文件清单（>10MB）

| 文件 | 大小 | git 状态 |
|------|------|----------|
| `models/model_2_random_forest/saved_models/rf_hourly.pkl` | 21.4 GB | 已忽略 ✅ |
| `data/data_engineer/hourly_data/processed_data/train_features.csv` | 2.6 GB | 已忽略 ✅ |
| `data/data_clean/cleaned_data/weather_hourly_cleaned.csv` | 423 MB | 已忽略 ✅ |
| `data/data_engineer/hourly_data/processed_data/test_features.csv` | 294 MB | 已忽略 ✅ |
| `data/data_engineer/daily_data/processed_data/train_features.csv` | 67 MB | 已忽略 ✅ |
| `data/data_clean/cleaned_data/weather_daily_cleaned.csv` | 12.9 MB | 已忽略 ✅ |

#### Task A2：分支管理 ✅ 完成
- [x] 创建 `random_forest` 分支
- [x] 推送当前进度到 `origin/random_forest`

---

### Task Group B：代码分析与系统集成

#### Task B3：数据完整性审计 ✅ 完成

**审计方法**：委派 explore 子代理深度审查数据清洗管道与特征工程管道代码，交叉比对 `任务.md` 需求与实际产出。

**数据文件完整清单**：

| 文件 | 路径 | 大小 | 数据行数 | 列数 |
|------|------|------|---------|------|
| 小时清洗数据 | `data/data_clean/cleaned_data/weather_hourly_cleaned.csv` | 423 MB | 4,295,928 | 18 |
| 日清洗数据 | `data/data_clean/cleaned_data/weather_daily_cleaned.csv` | 12.9 MB | 178,997 | 12 |
| 小时-训练集 | `data/data_engineer/hourly_data/processed_data/train_features.csv` | 2.6 GB | 3,864,385 | 67 |
| 小时-测试集 | `data/data_engineer/hourly_data/processed_data/test_features.csv` | 294 MB | 430,416 | 67 |
| 日-训练集 | `data/data_engineer/daily_data/processed_data/train_features.csv` | 67 MB | 161,063 | 43 |
| 日-测试集 | `data/data_engineer/daily_data/processed_data/test_features.csv` | 7.5 MB | 17,934 | 43 |

**任务.md 需求逐项验证**：

| 需求 | 状态 | 说明 |
|------|------|------|
| 时间特征分解（Month/Day/Hour） | ✅ 已实现 | 超额：含 year/season/周期 sin-cos 编码/时段分类 |
| 类别特征编码（Weather_Condition） | ✅ 已实现 | weather_code → 13类 WMO 天气码 ID |
| 类别特征编码（Wind_Direction） | ⚠️ 部分实现 | 数据为数值角度(0-360°)非离散文本，转为 wind_u/wind_v 向量分量（工程合理但与任务描述不一致） |
| StandardScaler 标准化 | ✅ 已实现 | 训练集 fit_transform / 测试集 transform，无泄露 |
| 基于时间的 train/test 切分 | ✅ 已实现 | train=2015~2023 / test=2024，纯时间过滤无 shuffle |
| 数据清洗管道 | ✅ 已实现 | 噪声移除 + 缺失填充 + 异常裁剪完整 |

**发现的问题**：
1. ⚠️ 小时数据行数轻微不一致：清洗 4,295,928 行 vs 训练+测试 4,294,801 行，差异 1,127 行（0.026%），可能因少量 `time` 字段无法解析为 datetime 被排除
2. ⚠️ 风向特征处理方式与任务.md 描述不一致（数值转 uv 分量 vs nn.Embedding），工程上合理但答辩需补充说明

---

#### Task B4：逻辑验证（决策层技术合理性）✅ 完成

**审计方法**：委派 explore 子代理深度审查 `models/model_2_random_forest/` 全部代码，评估接口设计、逻辑正确性与架构一致性。

**审计结论**：小时级训练流水线在算法层面设计合理（OOB 0.9024, R² 0.8397 可信），但架构完整性存在严重缺口。

| 级别 | 发现 | 文件:行 |
|------|------|---------|
| 🔴 P0 | 日级训练流水线完全断裂：`train_daily.py` 引用 `config.DAILY_MODEL_PATH`/`DAILY_TARGET_COLUMNS`/`DAILY_OUTPUT_DIR` 和 `load_daily()` 均不存在 | `train_daily.py:17,28,80,85,93` |
| 🔴 P0 | 日级评估报告与当前回归范式不匹配：`outputs/daily/evaluation_report.json` 是分类指标(accuracy/f1)，但 `evaluate.py` 已重写为回归指标(RMSE/R²)，存在"幽灵产物" | `outputs/daily/` vs `evaluate.py` |
| 🔴 P0 | precipitation log1p 变换数学不自洽：变换用 `log1p(x-min)` 但逆变换仅 `expm1(y)` 未加回 min，且 min 未持久化到 saved 字典；train/test 各自用各自的 min（目标侧泄露） | `data_loader.py:198` vs `predict.py:51` |
| 🟡 P1 | `is_fallback_needed()` 用全局 OOB 标量(0.90) 对比阈值 0.3 做逐样本决策 → 永不触发兜底，逻辑形同虚设 | `predict.py:99-113` |
| 🟡 P1 | `get_weight()` 返回单一 OOB 标量，无法体现逐目标差异化权重（温度强/降水弱） | `predict.py:91-93` |
| 🟡 P1 | 缺少逐样本不确定性输出（RF 可提供所有树预测 std），兜底层无法获得逐样本置信度 | `predict.py` 全局 |
| 🟡 P1 | 项目缺失 model_1/model_3/集成层/兜底层/Gradio UI 全部架构组件，README.md 承诺的三模型融合体系只有 model_2 一个 | `models/` 目录 |
| 🟢 P2 | 标签构造(shift(-1))与滞后特征执行顺序无实质性泄露，但建议文档说明"原始目标列同时作为特征保留"的设计意图 | `data_loader.py:189-214` |
| 🟢 P2 | 多目标回归用单一 RF multi-output，所有目标共享超参数，但降水(R²=0.34)与温度(R²=0.99)难度差异巨大，独立模型可能更优 | `train_hourly.py:124-125` |

---

#### Task B5：模型集成策略 ✅ 完成（方案制定）

**现状**：任务.md 要求"三模型综合研判 + 兜底方案"，但项目中 model_1/model_3/集成层均不存在。以下为集成策略技术方案。

##### 目标架构

```
用户输入（气象指标）
    │
    ▼
特征工程管道（data_engineer 标准化 + 时序特征）
    │
    ├──► model_1 (LinearRegression)     → preds_1 [n,5], weight_1
    ├──► model_2 (RandomForest)          → preds_2 [n,5], weight_2
    └──► model_3 (Wide&Deep PyTorch)     → preds_3 [n,5], weight_3
                │
                ▼
        加权平均（逐目标权重）
                │
                ▼
        物理范围约束（裁剪不合理值）
                │
                ▼
        逐样本置信度检查（预测方差 > 阈值?）
           ├── 否 → 输出综合预测
           └── 是 → 触发兜底（历史同期均值/气象规则）
                │
                ▼
        最终输出（5个气象变量 + 置信度 + 兜底标志）
```

##### model_2 对接接口契约

```python
from models.model_2_random_forest.predict import RFPredictor

rf = RFPredictor()                    # 加载小时级模型
preds = rf.predict(X)                 # [n, 5] 回归值（precipitation 已还原）
weight = rf.get_weight()              # OOB=0.9024（标量，建议改为逐目标）
importance = rf.get_feature_importance()  # {feature: importance}
fallback = rf.is_fallback_needed(X)   # 布尔数组（当前实现有缺陷，见 P1-1）
```

##### 统一推理接口设计（Inference Pipeline）

```python
class WeatherInferencePipeline:
    """三模型统一推理 + 综合研判 + 兜底"""
    
    def predict(self, X) -> dict:
        # 1. 三模型分别预测
        preds = {
            "model_1": self.linear.predict(X),
            "model_2": self.rf.predict(X),
            "model_3": self.dnn.predict(X),
        }
        # 2. 逐目标加权平均
        weights = self._get_target_weights()  # [5] 逐目标权重
        ensemble_pred = sum(preds[m] * weights[m] for m in preds)
        # 3. 物理范围约束
        ensemble_pred = self._clip_to_physical_range(ensemble_pred)
        # 4. 逐样本置信度检查
        uncertainty = self._compute_uncertainty(preds)  # 三模型预测的 std
        needs_fallback = uncertainty > self.fallback_threshold
        # 5. 兜底
        if needs_fallback.any():
            ensemble_pred[needs_fallback] = self._fallback(X[needs_fallback])
        return {"prediction": ensemble_pred, "uncertainty": uncertainty, "fallback": needs_fallback}
```

##### 逐目标权重分配方案

| 目标 | model_2 R² | 权重建议 | 说明 |
|------|-----------|---------|------|
| temperature_2m | 0.9938 | model_2 主导 | RF 在温度预测上极强 |
| apparent_temperature | 0.9937 | model_2 主导 | 与温度高度相关 |
| relative_humidity_2m | 0.9554 | 均衡 | 三模型均可能表现良好 |
| wind_speed_10m | 0.9149 | 均衡 | RF 略优 |
| precipitation | 0.3409 | model_2 降权 | RF 降水弱项，依赖兜底 |

##### 前置修复项（集成前必须完成）

1. **P0-3 修复**：precipitation log1p 变换需保存 `precip_min` 到 saved 字典，逆变换需 `expm1(y) + precip_min`
2. **P0-1 决策**：补齐日级配置或删除 `train_daily.py`，消除架构歧义
3. **P1-1 修复**：`is_fallback_needed` 改用逐样本预测方差（对所有树预测求 std）作置信度代理
4. **P1-2 增强**：`get_weight()` 改为返回 `[n_targets]` 逐目标权重向量

---

#### Task B6：健壮性与错误分析 ✅ 完成（含集成层实测）

**更新**：合并远程 main 后，集成层 `ensemble/` 已存在。以下分析基于实际代码。

##### hourly_ensemble 实测结果（ensemble/outputs/results/hourly_evaluation.txt）

| 目标 | MSE | MAE | R² |
|------|-----|-----|-----|
| temperature_2m | 0.6272 | 0.5947 | 0.9908 |
| precipitation | 0.1583 | 0.1047 | 0.3943 |
| wind_speed_10m | 4.7879 | 1.6059 | 0.9113 |
| apparent_temperature | 1.0063 | 0.7542 | 0.9899 |
| relative_humidity_2m | 13.3057 | 2.6459 | 0.9530 |

> hourly_ensemble 结果健康，但仅集成 Model 3 单模型，非真正多模型集成。

##### daily_ensemble 实测结果（所有 R² 为负值）

| 目标 | Model1 R² | Model3 R² | Ensemble R² |
|------|-----------|-----------|-------------|
| temp_mean | -0.9394 | -0.9008 | -0.8923 |
| precipitation | -0.1871 | -0.0854 | -0.1052 |
| wind | -0.8536 | -0.3665 | -0.5632 |

> 🔴 daily_ensemble 结果全坏——所有回归 R² 为负值，AUC≈0.49≈随机。根因：缺逆标准化 + 缺物理约束 + Model 1 pkl 全缺失。

##### 风险分析

| 风险 | 状态 | 说明 |
|------|------|------|
| hourly post_process 物理约束 | ✅ 已实现 | `post_process.py:43-60` 裁剪降水/风速非负、湿度[0,100]，但温度无约束 |
| daily 后处理缺失 | 🔴 缺失 | daily_ensemble 无逆标准化、无物理约束 → R² 全负 |
| 概率转换用错空间 | 🟡 bug | `model_wrapper.py:112` 用标准化值转概率，freezing 阈值 0°C 在 z-score 空间无意义 |
| 兜底机制 | ❌ 不存在 | ensemble 层无预测值异常检测/阈值触发兜底 |
| Model 2 被排除 | 🟡 未集成 | ensemble 层零引用 model_2（rf_hourly.pkl 存在但未接入） |

##### 物理边界约束（已由 hourly_ensemble post_process 实现）

```python
# hourly_ensemble/post_process.py:43-60 已实现：
result[:,1] = max(.,0)        # 降水非负
result[:,2] = max(.,0)        # 风速非负
result[:,4] = clip(.,0,100)  # 湿度[0,100]
# ⚠️ 温度和体感温度未做约束
```

---

#### Task B7：前端兼容性 ✅ 完成（含实际 UI 分析）

**更新**：合并远程 main 后，Web UI 已存在（Vue3 + FastAPI），非 Gradio。以下为实际分析。

##### 现状：Vue3 + FastAPI 全栈架构

| 组件 | 技术栈 | 文件 |
|------|--------|------|
| 后端 | FastAPI + Pydantic | `backend/app/main.py`, `routers/predict.py`, `services/predictor.py`, `schemas.py` |
| 前端 | Vue3 + TypeScript + Element Plus + Vite | `frontend/src/App.vue`, `views/*.vue`, `components/PredictionForm.vue` |
| API 通信 | REST (axios) | `frontend/src/api/index.ts` → POST `/api/predict` |

##### 前端页面

| 页面 | 文件 | 功能 |
|------|------|------|
| Dashboard | `views/Dashboard.vue` | 总览视图 |
| Hourly | `views/Hourly.vue` | 24小时逐小时预测 |
| Daily | `views/Daily.vue` | 7天每日预测 |
| ModelExplanation | `views/ModelExplanation.vue` | 模型可解释性 |
| PredictionForm | `components/PredictionForm.vue` | 城市选择+日期输入表单 |

##### 🔴 关键发现：后端 predictor.py 是假数据生成器

`backend/app/services/predictor.py:185-318` 中的 `_generate_daily_forecast` 和 `_generate_hourly_forecast` **完全不调用任何模型**，而是用 `random.uniform()` + 纬度/季节启发式规则生成假数据：

```python
# predictor.py:226 — 降水是随机数
precipitation = max(0, random.uniform(0, 8) if is_coastal else random.uniform(0, 3))
# predictor.py:180 — 置信度是随机数
"confidence": round(random.uniform(0.78, 0.95), 2)
```

虽然 `_load_daily_model()` / `_load_hourly_model()` 定义了模型加载逻辑（predictor.py:87-141），但 `predict()` 方法（predictor.py:145-181）**从未调用它们**，直接走假数据路径。model_weights 也是硬编码的假数据（predictor.py:345-352）。

##### 缺失分析

| 任务.md 要求 | 现状 |
|-------------|------|
| Gradio 网页界面 | ⚠️ 用 Vue3+FastAPI 替代（更专业，但未用 Gradio） |
| 左侧输入 | ✅ PredictionForm 组件（城市+日期） |
| 右侧输出（三模型预测+综合研判+兜底） | ❌ 后端不调用任何模型，输出假数据 |
| 模型集成 | ❌ predictor.py 未调用 ensemble 层 |
| 兜底状态展示 | ❌ 不存在 |

##### 部署前置条件

1. 🔴 **后端 predictor.py 需重写**：将 `_generate_*_forecast` 替换为实际调用 `ensemble/` 的代码
2. model_2 模型需可加载（解决 21.4GB OOM 问题）
3. model_1 的 .pkl 模型文件需训练生成（当前全部缺失）
4. ensemble hourly_ensemble 需接入 Model 2（当前仅 Model 3 单模型）
5. 前端 `npm install` + `npm run dev`，后端 `pip install -r backend/requirements.txt` + `python backend/run.py`

---

## 任务状态总览

| 任务 | 状态 | 负责方 | 备注 |
|------|------|--------|------|
| Phase 1.1 任务审查 | ✅ 完成 | Lead Agent | `任务.md` 已分析 |
| Phase 1.2 模型验证 | ✅ 完成 | Lead Agent | 模型可训练，但加载 OOM |
| Task A1 Git 卫生 | ✅ 完成 | Lead Agent | 大文件已忽略，tasks.md 已提交 |
| Task A2 分支管理 | ✅ 完成 | Lead Agent | 已推送到 origin/random_forest |
| 日级内容清理 | ✅ 完成 | Lead Agent | 删除 train_daily.py + 3个历史文档 |
| 合并远程 main | ✅ 完成 | Lead Agent | --allow-unrelated-histories，冲突以远程为准 |
| Task B3 数据审计 | ✅ 完成 | explore 子代理 | 6项需求5项满足，1项部分满足 |
| Task B4 逻辑验证 | ✅ 完成 | explore 子代理 | 3项P0 + 4项P1 + 2项P2 |
| Task B5 集成策略 | ✅ 完成 | Lead Agent | 方案已制定，含4项前置修复 |
| Task B6 健壮性分析 | ✅ 完成 | explore 子代理 | hourly 集成结果健康，daily 全坏 |
| Task B7 前端兼容 | ✅ 完成 | explore 子代理 | Vue3+FastAPI 已存在，但 predictor 输出假数据 |
| model_1 分析 | ✅ 完成 | explore 子代理 | Ridge/Logistic，代码完整但 .pkl 全缺 |
| model_3 分析 | ✅ 完成 | explore 子代理 | Wide&Deep PyTorch，hourly 主版本已训练 |
| ensemble 分析 | ✅ 完成 | explore 子代理 | hourly 单模型封装，daily 双模型但不可用 |

---

## 各组件分析摘要（合并远程 main 后）

### model_1_linear（线性回归）

| 维度 | 状态 | 说明 |
|------|------|------|
| 算法 | ✅ Ridge(回归) + Logistic(分类) | 10个子模型：3分类(rain/snow/severe) + 7回归(temp_*/precip/wind) |
| 代码 | ✅ 完整 | config.py/train.py/evaluate.py/predict.py 齐全 |
| 训练模型 | 🔴 全缺 | `models/` 下仅有 feature_names.json，10个 .pkl 全部不存在 |
| 接口 | ✅ 有 predict | 但返回 dict 格式与 model_2 的 RFPredictor 不兼容 |
| 数据 | ✅ 同源 | 读取同一份 train/test_features.csv |
| 切分 | ⚠️ 不一致 | 用上游默认切分(含2023)，model_2 重切到 train<2023 |

### model_3_deep_learning（深度学习）

| 维度 | 状态 | 说明 |
|------|------|------|
| 算法 | ✅ Wide & Deep (PyTorch) | 非 LSTM，forward 只取最后一步 `[:, -1, :]` |
| 主版本 | ✅ hourly_train | 5目标与 model_2 一致；daily_train 是遗留(9目标) |
| 训练 | ✅ 已完成 | best_model.pt (2.65MB)，best_val_loss=0.2189 |
| Embedding | ✅ 7路 | city/country/weather/month/hour/season/day_period |
| 接口 | ⚠️ 不兼容 | 需 CSV/DataLoader 输入，无 get_weight/is_fallback_needed |
| 切分 | ⚠️ 不一致 | 同 model_1，用上游默认切分 |
| 降水口径 | ⚠️ 不一致 | model_3 用原始 precipitation，model_2 用 log1p 变换 |
| 时序对齐 | 🔴 bug | `INCLUDE_CURRENT_HOUR` 是死参数，`__getitem__` 未引用 |

### ensemble（集成研判层）

| 维度 | hourly_ensemble | daily_ensemble |
|------|----------------|----------------|
| 集成对象 | 仅 Model 3（单模型封装） | Model 1 + Model 3（软投票） |
| 策略 | 无集成 | 加权平均（performance_based，但 Model3 性能全 None→等权） |
| 后处理 | ✅ 逆标准化+物理约束 | ❌ 缺失（R² 全负的根因） |
| 兜底 | ❌ 不存在 | ❌ 不存在 |
| Model 2 | ❌ 未引用 | ❌ 未引用 |
| 评估结果 | ✅ 健康(R² 0.39~0.99) | 🔴 全坏(R² 全负) |
| 可用性 | ✅ 可运行 | ❌ Model1 pkl 全缺 + 缺后处理 |

### Web UI（backend + frontend）

| 维度 | 状态 | 说明 |
|------|------|------|
| 前端 | ✅ 完整 | Vue3 + TypeScript + Element Plus + Vite，4个页面 |
| 后端 | ⚠️ 脚手架 | FastAPI 路由/schemas 完整，但 predictor 输出假数据 |
| 模型集成 | 🔴 未接入 | predictor.py 用 random.uniform() 生成假数据，不调用任何模型 |
| API | ✅ /api/predict + /api/cities | REST + CORS |
| 数据流 | ❌ 断裂 | 前端→后端API→predictor(假数据)→前端，模型未参与 |

---

## 关键发现汇总（合并远程 main 后更新）

### 🔴 P0 级（阻断性，必须修复）

1. **P0 — 后端输出假数据**：`predictor.py` 用 `random.uniform()` 生成天气值，不调用任何模型。整个 Web UI 是"空壳"
2. **P0 — 模型加载 OOM**：`rf_hourly.pkl` 21.4GB 超出内存（已有瘦身提交 fe84908，降为 n_estimators=100 预计 2-4GB）
3. **P0 — daily_ensemble 全坏**：所有 R² 为负值，缺逆标准化 + Model 1 pkl 全缺
4. **P0 — Model 1 模型全缺**：10 个 .pkl 文件不存在，daily_ensemble 无法运行
5. **P0 — precipitation log1p 不自洽**：model_2 变换 `log1p(x-min)` vs 逆变换 `expm1(y)` 未加回 min

### 🟡 P1 级（设计缺陷，影响集成）

6. **P1 — Model 2 未接入 ensemble**：rf_hourly.pkl 存在但 ensemble 层零引用
7. **P1 — 时序对齐 bug**：`INCLUDE_CURRENT_HOUR` 是死参数，model_2(t→t+1) 与 model_3(t-24:t-1→t) 差 1 小时
8. **P1 — 切分不一致**：model_2 重切到 train<2023，model_1/model_3 用上游默认切分(含2023)
9. **P1 — 降水口径不一致**：model_2 用 log1p，model_3 用原始值，集成时尺度不匹配
10. **P1 — 概率转换用错空间**：hourly model_wrapper 用标准化值转概率，0°C 阈值在 z-score 空间无意义
11. **P1 — 接口不统一**：model_1/model_2/model_3 的 predict 接口签名、返回格式完全不同
12. **P1 — hourly_ensemble 非真集成**：仅 Model 3 单模型封装，无加权投票

### 🟢 P2 级（改进建议）

13. **P2 — 降水预测弱项**：R²=0.34，建议独立训练或集成层单独加权
14. **P2 — model_3 forward 只用最后一步**：24步序列输入但丢弃前23步，等价单点 DNN
15. **P2 — hourly evaluate 不落盘逐目标指标**：集成层无法读取 R² 用作加权权重
16. **P2 — daily_train 是遗留版本**：9目标与 model_2 不兼容，建议清理

---

## 推荐执行顺序（更新）

1. **P0-2**（model_2 瘦身重训）→ 使 predict.py 可运行（已有提交 fe84908）
2. **P0-1**（后端 predictor 接入真实模型）→ 让 Web UI 输出真实预测
3. **P0-4**（Model 1 训练生成 .pkl）→ 使 daily_ensemble 可运行
4. **P0-3**（daily_ensemble 后处理修复）→ 逆标准化 + 物理约束
5. **P0-5**（precipitation log1p 修复）→ 变换-逆变换对称性
6. **P1-6**（Model 2 接入 ensemble）→ 三模型真正融合
7. **P1-7**（时序对齐修复）→ `INCLUDE_CURRENT_HOUR` 改为 True 并修复 `__getitem__`
8. **P1-8/9**（切分+降水口径对齐）→ 统一 train<2023 + log1p 口径
9. **P1-11**（统一模型接口）→ 三模型 predict 签名一致
10. **P2 项** → 精度与工程规范优化
