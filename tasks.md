# tasks.md — model_2_random_forest 开发与集成任务追踪

> 项目：final2/weather — 欧洲城市天气预测系统
> 角色：Lead Agent（主导协调）
> 创建时间：2026-07-14
> 分支策略：`main` ← 已有进度；`random_forest` ← 待推送的模型开发分支

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
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt

# 2. 模型验证（仅加载元信息，不执行预测）
cd models/model_2_random_forest
python predict.py
# 输出：目标变量、log1p 状态、OOB Score、特征数、Top-5 特征

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

#### Task A1：Git 卫生
- [x] 识别大文件（见下方"大文件清单"）
- [x] 确认 `.gitignore` 已覆盖所有大文件类型（`*.pkl`, `*.csv`, `saved_models/`, `outputs/`）
- [x] 确认工作树清洁（`git status` 无未提交变更）
- [x] 确认大文件均未被 git 跟踪（`git check-ignore` 验证通过）
- [ ] 提交 `tasks.md` 到本地仓库

#### 大文件清单（>10MB）

| 文件 | 大小 | git 状态 |
|------|------|----------|
| `models/model_2_random_forest/saved_models/rf_hourly.pkl` | 21.4 GB | 已忽略 ✅ |
| `data/data_engineer/hourly_data/processed_data/train_features.csv` | 2.6 GB | 已忽略 ✅ |
| `data/data_clean/cleaned_data/weather_hourly_cleaned.csv` | 423 MB | 已忽略 ✅ |
| `data/data_engineer/hourly_data/processed_data/test_features.csv` | 294 MB | 已忽略 ✅ |
| `data/data_engineer/daily_data/processed_data/train_features.csv` | 67 MB | 已忽略 ✅ |
| `data/data_clean/cleaned_data/weather_daily_cleaned.csv` | 12.9 MB | 已忽略 ✅ |

#### Task A2：分支管理
- [ ] 创建 `random_forest` 分支
- [ ] 推送当前进度到 `origin/random_forest`

---

### Task Group B：代码分析与系统集成

#### Task B3：数据完整性审计
- [x] 从 `main` 分支拉取最新代码（已是 main 最新状态，commit 361cbb0）
- [ ] 对照 `任务.md` 检查数据集完整性
  - [ ] 原始数据：`data/dataset/` — 采集脚本输出
  - [ ] 清洗数据：`data/data_clean/cleaned_data/weather_hourly_cleaned.csv` (423MB) ✅
  - [ ] 清洗数据：`data/data_clean/cleaned_data/weather_daily_cleaned.csv` (12.9MB) ✅
  - [ ] 特征工程产出：`data/data_engineer/hourly_data/processed_data/train_features.csv` (2.6GB) ✅
  - [ ] 特征工程产出：`data/data_engineer/hourly_data/processed_data/test_features.csv` (294MB) ✅
  - [ ] 特征工程产出：`data/data_engineer/daily_data/processed_data/train_features.csv` (67MB) ✅
- [ ] 确认 `任务.md` 第一步要求的特征工程（时间分解/类别编码/标准化）已在 data_engineer 中实现
- [ ] 确认 train/test 按时间切分（train=2015~2022, test=2023~2024）

#### Task B4：逻辑验证（决策层技术合理性）
- [ ] 审查现有决策逻辑架构
- [ ] 确认 model_2 的 `predict.py` 接口设计（`predict_hourly`, `get_weight`, `is_fallback_needed`）
- [ ] ⚠️ **发现**：当前无独立"综合研判层"代码（ensemble/decision 模块未创建）
- [ ] ⚠️ **发现**：`predict.py` 的 `is_fallback_needed` 用 OOB Score 作置信度参考，逻辑待完善（OOB 是全局值，非逐样本）
- [ ] ⚠️ **发现**：`train_daily.py` 引用 `config.DAILY_MODEL_PATH` 等，但 `config.py` 已删除日级配置 → 日级训练脚本损坏

#### Task B5：模型集成策略
- [ ] 制定 model_2 集成到决策管线的技术方案（见下方"集成策略草案"）
- [ ] 确认 model_1（线性）与 model_3（深度学习）的接口契约
- [ ] 设计统一推理接口（Inference Pipeline）签名
- [ ] 设计加权平均策略权重分配方案

#### Task B6：健壮性与错误分析
- [ ] 测试多模型组合是否产生异常值
- [ ] 分析降水预测 R²=0.34 的异常风险（可能输出负值或不合理极端值）
- [ ] 设计输出范围约束（物理边界裁剪：温度、湿度、风速合理区间）
- [ ] 设计兜底触发逻辑（置信度 < 阈值 → 启发式规则）
- [ ] 编写集成层单元测试

#### Task B7：前端兼容性
- [ ] ⚠️ **发现**：当前项目无 Web UI 代码（`任务.md` 第五步要求的 Gradio 界面未创建）
- [ ] 设计 Gradio 界面方案
- [ ] 确认前端输入字段与 model_2 特征列对齐
- [ ] 确认前端输出展示（单模型预测 + 综合研判 + 兜底状态）

---

## 集成策略草案（Task B5 预研）

### 目标架构
```
用户输入 → 特征工程管道 → ┌─ model_1 (线性回归) ──┐
                           ├─ model_2 (随机森林) ──┤ → 加权平均 → 范围约束 → 兜底检查 → 最终输出
                           └─ model_3 (Wide&Deep) ─┘
```

### model_2 对接接口
```python
from models.model_2_random_forest.predict import RFPredictor

rf = RFPredictor()                           # 加载小时级模型
preds = rf.predict(X)                        # [n, 5] 回归值
weight = rf.get_weight()                      # OOB=0.9024，供加权平均
fallback = rf.is_fallback_needed(X)           # 布尔数组，兜底触发判断
```

### 待解决问题
1. **模型体积**：21.4GB 无法在常规环境加载 → 需 mmap_mode 或重训压缩
2. **权重分配**：model_2 OOB=0.9024 vs model_1/model_3 待确认，需统一验证集对比
3. **降水弱项**：R²=0.34，集成层需对降水单独处理或加权降权
4. **日级配置缺失**：`config.py` 删除了 DAILY_* 常量，`train_daily.py` 无法运行

---

## 任务状态总览

| 任务 | 状态 | 负责方 | 备注 |
|------|------|--------|------|
| Phase 1.1 任务审查 | ✅ 完成 | Lead Agent | `任务.md` 已分析 |
| Phase 1.2 模型验证 | ✅ 完成 | Lead Agent | 模型可训练，但加载 OOM |
| Task A1 Git 卫生 | 🔄 进行中 | Lead Agent | 大文件已忽略，待提交 tasks.md |
| Task A2 分支管理 | ⬜ 待开始 | Lead Agent | 待创建 random_forest 分支 |
| Task B3 数据审计 | 🔄 进行中 | Lead Agent | 数据集齐全，待深度核验 |
| Task B4 逻辑验证 | ⬜ 待开始 | Lead Agent | 发现 3 项架构缺陷 |
| Task B5 集成策略 | ⬜ 待开始 | Lead Agent | 草案已列，待细化 |
| Task B6 健壮性分析 | ⬜ 待开始 | Lead Agent | 待集成层就绪后测试 |
| Task B7 前端兼容 | ⬜ 待开始 | Lead Agent | Web UI 尚未创建 |

---

## 关键发现汇总（需优先处理）

1. 🔴 **P0 — 模型加载 OOM**：`rf_hourly.pkl` 21.4GB 超出本机内存，`predict.py` 无法运行
2. 🔴 **P0 — 日级训练损坏**：`train_daily.py` 引用 `config.py` 中不存在的 `DAILY_*` 常量
3. 🟡 **P1 — 无集成研判层**：`任务.md` 第三步要求的综合研判模块未创建
4. 🟡 **P1 — 无 Web UI**：`任务.md` 第五步要求的 Gradio 界面未创建
5. 🟡 **P1 — 兜底逻辑不完善**：`is_fallback_needed` 用全局 OOB 而非逐样本置信度
6. 🟢 **P2 — 降水预测弱项**：R²=0.34，集成层需特殊处理
