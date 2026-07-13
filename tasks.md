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

#### Task B6：健壮性与错误分析 ✅ 完成（分析）

**测试条件说明**：由于 model_2 模型文件 21.4GB 无法加载（OOM），且 model_1/model_3/集成层代码不存在，无法执行实际的多模型组合测试。以下为基于代码审查的理论分析与风险预案。

##### 风险 1：多模型组合产生异常值

| 风险场景 | 触发条件 | 影响 | 预案 |
|---------|---------|------|------|
| 降水负值 | log1p 逆变换不自洽(P0-3) + 多模型加权 | 输出负降水量 | 集成层输出后强制 `clip(0, None)` |
| 温度极端值 | 某模型在未见数据上外推失败 | 预测温度超出物理范围 | 裁剪到 [-60°C, 60°C] |
| 湿度越界 | 标准化逆变换后超出 [0%, 100%] | 不合理的湿度值 | 裁剪到 [0, 100] |
| 风速负值 | 回归模型可能输出负值 | 不合理风速 | 裁剪到 [0, None] |
| 三模型分歧过大 | 某模型在边缘数据点严重偏离 | 加权后结果被拖偏 | 用预测 std 做异常检测，分歧 > 2σ 触发兜底 |

##### 物理边界约束设计

```python
PHYSICAL_RANGES = {
    "temperature_2m":       (-60, 60),    # °C
    "precipitation":        (0, 200),     # mm/h
    "wind_speed_10m":       (0, 120),     # km/h
    "apparent_temperature": (-70, 70),    # °C
    "relative_humidity_2m": (0, 100),     # %
}

def clip_to_physical_range(preds, target_columns):
    for i, col in enumerate(target_columns):
        lo, hi = PHYSICAL_RANGES[col]
        preds[:, i] = np.clip(preds[:, i], lo, hi)
    return preds
```

##### 兜底触发逻辑设计

当前 `is_fallback_needed` 的缺陷：用全局 OOB(0.90) 对比阈值 0.3 → 永不触发。

**改进方案**：基于三模型预测一致性 + RF 逐样本不确定性：

```python
def is_fallback_needed(self, X, models_preds, threshold=2.0):
    """
    逐样本兜底判断：
    1. 三模型预测的标准差 > threshold → 分歧大，触发兜底
    2. RF 各树预测的 std > threshold → 不确定性高，触发兜底
    """
    # 三模型预测分歧（跨模型 std）
    stacked = np.stack(list(models_preds.values()))  # [3, n, 5]
    cross_model_std = stacked.std(axis=0)            # [n, 5]
    
    # RF 逐样本不确定性（对所有树预测求 std）
    tree_preds = np.array([tree.predict(X) for tree in self.model.estimators_])  # [n_trees, n, 5]
    rf_uncertainty = tree_preds.std(axis=0)  # [n, 5]
    
    # 综合判断
    needs_fallback = (cross_model_std > threshold).any(axis=1) | (rf_uncertainty > threshold).any(axis=1)
    return needs_fallback
```

##### 单元测试清单（集成层就绪后执行）

- [ ] 测试正常输入 → 三模型均输出合理值，加权结果在物理范围内
- [ ] 测试极端输入（高温/极端降水）→ 物理约束裁剪生效
- [ ] 测试三模型分歧场景 → 兜底正确触发
- [ ] 测试降水负值 → clip(0, None) 生效
- [ ] 测试空输入/NaN → 优雅降级而非崩溃

---

#### Task B7：前端兼容性 ✅ 完成（评估）

**现状**：当前项目**无任何 Web UI 代码**。任务.md 第五步要求的 Gradio 界面未创建。

##### 缺失分析

| 任务.md 要求 | 现状 |
|-------------|------|
| Gradio 网页界面 | ❌ 不存在 |
| 左侧输入（气温/湿度/气压/风向滑块） | ❌ 不存在 |
| 右侧输出（三模型预测 + 综合研判 + 兜底状态） | ❌ 不存在 |

##### Gradio 界面方案设计

```python
import gradio as gr

def predict_weather(temperature, humidity, pressure, wind_dir, ...):
    # 1. 构造特征向量
    X = build_features(temperature, humidity, pressure, wind_dir, ...)
    # 2. 三模型预测 + 综合研判
    result = pipeline.predict(X)
    # 3. 格式化输出
    return {
        "model_1_预测": result["model_1"],
        "model_2_预测": result["model_2"],
        "model_3_预测": result["model_3"],
        "综合研判": result["ensemble"],
        "兜底触发": "是" if result["fallback"] else "否",
    }

with gr.Blocks() as demo:
    gr.Markdown("# 天气预测综合研判系统")
    with gr.Row():
        with gr.Column():
            temp = gr.Slider(-40, 50, label="温度 (°C)")
            hum = gr.Slider(0, 100, label="湿度 (%)")
            pres = gr.Slider(900, 1100, label="气压 (hPa)")
            wind = gr.Dropdown(["北","东北","东","东南","南","西南","西","西北"], label="风向")
            btn = gr.Button("预测")
        with gr.Column():
            out1 = gr.Textbox(label="模型一（线性回归）")
            out2 = gr.Textbox(label="模型二（随机森林）")
            out3 = gr.Textbox(label="模型三（深度学习）")
            out_ens = gr.Textbox(label="综合研判结论")
            out_fb = gr.Textbox(label="兜底状态")
    btn.click(predict_weather, [temp, hum, pres, wind], [out1, out2, out3, out_ens, out_fb])

demo.launch()
```

##### 前端输入字段与 model_2 特征对齐

model_2 小时级特征共 143 列（含 67 基础 + 76 滞后/滚动/交互特征）。Gradio 界面无法让用户手动输入全部 143 列，需：
1. **简化输入**：只让用户输入核心气象变量（温度/湿度/气压/风速/风向/时间）
2. **自动补全**：滞后/滚动特征由系统从历史数据自动构造
3. **特征工程复用**：调用 `data_engineer` 的 `feature_creator.py` 完成剩余特征构造

##### 部署前置条件

1. model_2 模型需可加载（解决 21.4GB OOM 问题）
2. model_1/model_3 需训练完成
3. 集成层需开发完成
4. Gradio 依赖需添加到 `requirements.txt`（当前未包含）

---

## 任务状态总览

| 任务 | 状态 | 负责方 | 备注 |
|------|------|--------|------|
| Phase 1.1 任务审查 | ✅ 完成 | Lead Agent | `任务.md` 已分析 |
| Phase 1.2 模型验证 | ✅ 完成 | Lead Agent | 模型可训练，但加载 OOM |
| Task A1 Git 卫生 | ✅ 完成 | Lead Agent | 大文件已忽略，tasks.md 已提交 |
| Task A2 分支管理 | ✅ 完成 | Lead Agent | 已推送到 origin/random_forest |
| Task B3 数据审计 | ✅ 完成 | explore 子代理 | 6项需求5项满足，1项部分满足 |
| Task B4 逻辑验证 | ✅ 完成 | explore 子代理 | 3项P0 + 4项P1 + 2项P2 |
| Task B5 集成策略 | ✅ 完成 | Lead Agent | 方案已制定，含4项前置修复 |
| Task B6 健壮性分析 | ✅ 完成 | Lead Agent | 5类风险预案 + 兜底逻辑改进方案 |
| Task B7 前端兼容 | ✅ 完成 | Lead Agent | Web UI 不存在，方案已设计 |

---

## 关键发现汇总（需优先处理）

### 🔴 P0 级（阻断性，必须修复）

1. **P0 — 模型加载 OOM**：`rf_hourly.pkl` 21.4GB 超出本机内存，`predict.py` 无法运行。需 mmap_mode 或重训压缩（降 n_estimators 或提高 compress）
2. **P0 — 日级训练断裂**：`train_daily.py` 引用 `config.py` 中不存在的 `DAILY_MODEL_PATH`/`DAILY_TARGET_COLUMNS`/`DAILY_OUTPUT_DIR` 和 `data_loader.load_daily()`。需补齐配置或删除脚本
3. **P0 — 日级评估幽灵产物**：`outputs/daily/evaluation_report.json` 是分类指标(accuracy/f1)，与当前回归范式 `evaluate.py`(RMSE/R²) 不匹配。需清理或重新生成
4. **P0 — precipitation log1p 不自洽**：变换 `log1p(x-min)` vs 逆变换 `expm1(y)` 未加回 min，且 min 未持久化。需修复变换-逆变换对称性

### 🟡 P1 级（设计缺陷，影响集成）

5. **P1 — 无集成研判层**：任务.md 第三步要求的综合研判模块（加权平均/Stacking）未创建
6. **P1 — 无 Web UI**：任务.md 第五步要求的 Gradio 界面未创建
7. **P1 — 兜底逻辑失效**：`is_fallback_needed` 用全局 OOB(0.90) 对比阈值 0.3，永不触发。需改为逐样本预测方差
8. **P1 — 权重接口不足**：`get_weight()` 返回标量，无法体现逐目标差异化（温度强/降水弱）
9. **P1 — 缺少逐样本不确定性**：RF 可提供所有树预测 std，但接口未暴露
10. **P1 — model_1/model_3 不存在**：README 承诺三模型，实际只有 model_2

### 🟢 P2 级（改进建议）

11. **P2 — 降水预测弱项**：R²=0.34，建议独立训练降水模型或集成层单独加权
12. **P2 — 多目标共享超参**：5 目标用同一套超参数，降水与温度难度差异大
13. **P2 — 小时数据行数差异**：清洗 4,295,928 行 vs 训练+测试 4,294,801 行，差 1,127 行(0.026%)
14. **P2 — 风向处理方式**：转 uv 分量 vs 任务.md 的 nn.Embedding，答辩需补充说明

---

## 推荐执行顺序

1. **P0-4**（precipitation 变换修复）→ 影响输出正确性
2. **P0-1/P0-2/P0-3**（日级流水线：补齐或删除）→ 消除架构歧义
3. **P0-1**（模型 OOM：重训压缩）→ 使 predict.py 可运行
4. **P1-7/P1-8/P1-9**（兜底逻辑 + 权重 + 不确定性接口）→ 支撑集成层
5. **P1-5/P1-10**（集成层 + model_1/model_3）→ 满足任务书完整交付
6. **P1-6**（Gradio UI）→ 可视化部署
7. **P2 项**→ 精度与工程规范优化
