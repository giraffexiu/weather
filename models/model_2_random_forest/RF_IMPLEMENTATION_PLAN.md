# 随机森林模型实现计划 (Model 2: Random Forest)

> 负责人：曹子恒  
> 日期：2026-07-12  
> 项目：final2/weather — 欧洲城市天气预测系统  
> 定位：非线性强规则捕捉模型，为综合研判层提供概率输出

---

## 一、预测目标定义

当前特征工程产出（`train_features.csv` / `test_features.csv`）包含 43 列特征但**无标签列**。随机森林模块需要自行构造预测目标。

### 推荐方案：双任务并行

| 任务 | 目标变量 | 类型 | 构造方式 | 业务意义 |
|------|----------|------|----------|----------|
| **任务A（主）** | `rain_tomorrow` | 二分类 | `is_rainy` 按城市分组后偏移1行 | 预测明天是否下雨 |
| **任务B（辅）** | `temp_tomorrow` | 回归 | `temperature_2m_mean` 按城市分组后偏移1行 | 预测明天平均温度 |

> **推荐优先实现任务A（分类）**，因为：
> 1. 任务.md 明确提到"明天是否有雨/雪（分类）"
> 2. 二分类天然支持概率输出，适配综合研判层的软投票/加权平均机制
> 3. 可使用 `class_weight="balanced"` 处理类别不平衡
> 4. OOB 评估和 feature_importances_ 在分类场景下更直观

---

## 二、目录结构

```
models/model_2_random_forest/
├── README.md                    # 模型说明文档
├── config.py                    # 超参数与路径配置
├── data_loader.py               # 数据加载与标签构造
├── train.py                     # 训练主脚本（含 GridSearchCV + OOB）
├── evaluate.py                  # 评估脚本（混淆矩阵/ROC/特征重要性）
├── predict.py                   # 统一推理接口（供集成层调用）
├── utils.py                     # 可视化与辅助函数
├── requirements.txt             # 依赖
├── saved_models/                # 模型产物（.gitignore 忽略）
│   ├── rf_classifier.pkl        # 分类模型
│   ├── rf_regressor.pkl         # 回归模型（可选）
│   └── feature_config.json      # 特征列配置
└── outputs/                     # 评估图表与报告（.gitignore 忽略）
    ├── confusion_matrix.png
    ├── roc_curve.png
    ├── feature_importance.png
    ├── oob_error_curve.png
    └── evaluation_report.md
```

---

## 三、实施步骤（5个阶段）

### 阶段1：数据加载与标签构造 — `data_loader.py`

**目标**：从特征工程产出中加载数据，构造时序预测标签，处理数据泄露。

**核心逻辑**：
```python
# 按城市分组，构造"明天"的标签
df = df.sort_values(['city', 'time'])
df['rain_tomorrow'] = df.groupby('city')['is_rainy'].shift(-1)
df = df.dropna(subset=['rain_tomorrow'])  # 删除最后一天（无明天数据）

# 特征矩阵 X：排除标签列和未来信息列
# 标签 y：rain_tomorrow (0/1)
feature_cols = [c for c in df.columns if c not in 
    ['city', 'country', 'time', 'rain_tomorrow', 'is_rainy']]
```

**关键防泄露规则**：
- `is_rainy` 是"今天是否下雨"，而 `rain_tomorrow` 是"明天是否下雨"。`is_rainy` 本身可以作为特征（今天的天气状况是预测明天的输入），**不构成泄露**
- 但必须删除每个城市最后一条记录（没有"明天"的数据）
- 严格使用已有的时间切分（train: 2015-2023, test: 2024），不重新随机切分

**数据质量处理**：
- 检查 `rain_tomorrow` 的类别分布，预期正样本率约 30-40%（欧洲城市平均雨天比例）
- 如果类别极度不平衡（正样本 <10%），记录但不立即处理——交给 `class_weight` 参数

**输出**：
- `X_train, y_train, X_test, y_test, feature_names` — numpy 数组 + 列名列表
- 保存 `feature_config.json` 记录使用的特征列顺序

---

### 阶段2：超参数搜索与训练 — `train.py`

**目标**：使用 GridSearchCV 搜索最优超参数，启用 OOB 评估，训练最终模型。

**策略：两阶段调优（先粗搜后细搜）**

#### 阶段2a：粗粒度 GridSearchCV
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

rf_base = RandomForestClassifier(
    random_state=42,
    n_jobs=-1,
    class_weight='balanced',        # 处理类别不平衡
    oob_score=True,                 # 启用 OOB 评估（任务.md 要求）
)

# 粗搜网格（12种组合）
param_grid_coarse = {
    'n_estimators': [100, 200, 300],
    'max_depth': [15, 25, None],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt'],       # 固定为 sqrt（分类推荐默认）
}
```

#### 阶段2b：细粒度搜索（基于粗搜结果缩小范围）
```python
# 假设粗搜最优：n_estimators=200, max_depth=25, min_samples_leaf=2
param_grid_fine = {
    'n_estimators': [150, 200, 250, 300],
    'max_depth': [20, 25, 30, None],
    'min_samples_leaf': [1, 2, 3],
    'max_features': ['sqrt', 'log2', 0.3, 0.5],   # 搜索特征采样比例
    'min_samples_split': [2, 5, 10],              # 新增：节点分裂最小样本数
}
```

**交叉验证策略**：
```python
# 使用 TimeSeriesSplit 而非 KFold（时序数据专用）
tscv = TimeSeriesSplit(n_splits=5)
grid = GridSearchCV(rf_base, param_grid_fine, cv=tscv, 
                     scoring='f1', n_jobs=-1, verbose=1, refit=True)
```

> **为什么用 TimeSeriesSplit 而非默认 KFold**：
> 天气数据有时序依赖性，随机 KFold 切分会导致训练集中包含未来数据、测试集中包含过去数据，造成数据泄露。TimeSeriesSplit 保证每折的训练数据始终在测试数据之前。

**OOB 评估**（任务.md 明确要求）：
```python
# 最终模型使用 oob_score=True
rf_final = RandomForestClassifier(
    **grid.best_params_,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced',
    oob_score=True,
    bootstrap=True,                 # OOB 要求 bootstrap=True
)
rf_final.fit(X_train, y_train)
print(f"OOB Score: {rf_final.oob_score_:.4f}")
# OOB Score 是无偏泛化估计，无需额外验证集
```

**模型持久化**：
```python
import pickle
with open('saved_models/rf_classifier.pkl', 'wb') as f:
    pickle.dump(rf_final, f)
```

---

### 阶段3：全面评估 — `evaluate.py`

**目标**：生成全套评估指标和可视化图表。

**评估指标矩阵**：

| 维度 | 指标 | 说明 |
|------|------|------|
| 整体 | Accuracy | 准确率 |
| 整体 | Macro-F1 | 多类别均衡 F1（本任务为二分类，等同 F1） |
| 整体 | ROC-AUC | 概率排序能力（综合研判层需要） |
| 整体 | OOB Score | 袋外无偏估计（任务.md 要求） |
| 分类 | Precision / Recall | 按类别输出 |
| 分类 | Classification Report | 完整报告 |
| 校准 | Brier Score | 概率校准度（影响软投票质量） |

**可视化产出**：

1. **混淆矩阵** — `confusion_matrix.png`
   - 标注绝对值 + 百分比
   - 中文标签（下雨/不下雨）

2. **ROC 曲线** — `roc_curve.png`
   - 标注 AUC 值
   - 对角线参考线

3. **特征重要性 Top-20** — `feature_importance.png`
   - 水平条形图，按重要性降序
   - **这是任务.md 明确要求的可解释性输出**——答辩时展示"哪些气象指标对预测下雨最重要"

4. **OOB 误差曲线** — `oob_error_curve.png`
   - X 轴：n_estimators 数量（50→500）
   - Y 轴：OOB 错误率
   - 展示模型何时收敛，辅助判断最优树数量

5. **概率分布直方图** — `probability_distribution.png`
   - 测试集预测概率分布
   - 按真实标签着色
   - 直观展示模型置信度分布（辅助兜底阈值设定）

**评估报告** — `evaluation_report.md`：
```markdown
# 随机森林评估报告

## 最优超参数
- n_estimators: 200
- max_depth: 25
- ...

## 性能指标
| 指标 | 值 |
|------|-----|
| Accuracy | 0.85 |
| F1-Score | 0.78 |
| ROC-AUC | 0.91 |
| OOB Score | 0.83 |

## Top-10 重要特征
1. humidity (0.15)
2. pressure (0.12)
...

## 结论
- 模型在测试集上 F1=0.78，OOB=0.83（无偏估计）
- 湿度和气压是最重要的预测因子
- 建议集成层使用加权平均策略
```

---

### 阶段4：统一推理接口 — `predict.py`

**目标**：封装为可被集成研判层（Ensemble Layer）调用的统一接口。

**接口设计**：
```python
class RFPredictor:
    """随机森林预测器 - 供集成层调用"""
    
    def __init__(self, model_path='saved_models/rf_classifier.pkl'):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
    
    def predict_proba(self, X) -> np.ndarray:
        """返回概率输出 [n_samples, 2] — 供软投票/Stacking使用"""
        return self.model.predict_proba(X)
    
    def predict(self, X) -> np.ndarray:
        """返回类别预测 [n_samples]"""
        return self.model.predict(X)
    
    def get_feature_importance(self) -> dict:
        """返回特征重要性字典 — 供可解释性展示"""
        return dict(zip(self.feature_names, self.model.feature_importances_))
    
    def get_oob_score(self) -> float:
        """返回 OOB 分数"""
        return self.model.oob_score_
```

**与集成层对接的约定**：
- **必须输出概率**（`predict_proba`），而非硬标签
- 概率格式：`[P(不下雨), P(下雨)]` — 与模型1/模型3保持一致
- 提供模型权重参考值：`weight = F1_score`（供加权平均策略使用）

---

### 阶段5：配置与依赖 — `config.py` + `requirements.txt`

**`config.py`**：
```python
import os
from pathlib import Path

# ==================== 路径配置 ====================
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "data"

# 特征工程产出路径（日级数据）
TRAIN_DATA_PATH = DATA_ROOT / "data_engineer" / "daily_data" / "processed_data" / "train_features.csv"
TEST_DATA_PATH = DATA_ROOT / "data_engineer" / "daily_data" / "processed_data" / "test_features.csv"

# 模型保存路径
MODEL_DIR = Path(__file__).parent / "saved_models"
MODEL_PATH = MODEL_DIR / "rf_classifier.pkl"
REGRESSOR_PATH = MODEL_DIR / "rf_regressor.pkl"
FEATURE_CONFIG_PATH = MODEL_DIR / "feature_config.json"

# 输出路径
OUTPUT_DIR = Path(__file__).parent / "outputs"

# ==================== 超参数配置 ====================
RANDOM_SEED = 42

# 标签配置
TARGET_COLUMN = 'rain_tomorrow'       # 预测目标
TARGET_CONSTRUCTION = 'shift'          # 构造方式：按城市偏移
SHIFT_PERIODS = -1                     # 负数表示"明天"
GROUP_COLUMN = 'city'                  # 分组列

# 排除列（不作为特征使用的列）
EXCLUDE_COLUMNS = ['city', 'country', 'time', 'rain_tomorrow']

# RandomForest 超参数网格
PARAM_GRID_COARSE = {
    'n_estimators': [100, 200, 300],
    'max_depth': [15, 25, None],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt'],
}

PARAM_GRID_FINE = {
    'n_estimators': [150, 200, 250, 300],
    'max_depth': [20, 25, 30, None],
    'min_samples_leaf': [1, 2, 3],
    'max_features': ['sqrt', 'log2', 0.3, 0.5],
    'min_samples_split': [2, 5, 10],
}

# 固定参数
RF_FIXED_PARAMS = {
    'random_state': RANDOM_SEED,
    'n_jobs': -1,
    'class_weight': 'balanced',
    'oob_score': True,
    'bootstrap': True,
}

# 交叉验证
CV_SPLITS = 5
CV_SCORING = 'f1'         # 二分类 F1

# OOB 曲线配置
OOB_N_ESTIMATORS_RANGE = list(range(50, 501, 50))
```

**`requirements.txt`**：
```
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

---

## 四、效果优化策略（追求最优）

### 4.1 特征工程增强（在 data_loader.py 中）

1. **滞后特征（Lag Features）**
   ```python
   # 使用前1-3天的天气数据作为特征
   for lag in [1, 2, 3]:
       df[f'temp_lag_{lag}'] = df.groupby('city')['temperature_2m_mean'].shift(lag)
       df[f'rain_lag_{lag}'] = df.groupby('city')['is_rainy'].shift(lag)
       df[f'pressure_lag_{lag}'] = df.groupby('city')['pressure_msl'].shift(lag)  # 小时级有
   ```

2. **滑动窗口统计**
   ```python
   # 过去7天的统计量
   df['temp_7d_mean'] = df.groupby('city')['temperature_2m_mean'].transform(
       lambda x: x.rolling(7).mean())
   df['rain_7d_sum'] = df.groupby('city')['is_rainy'].transform(
       lambda x: x.rolling(7).sum())  # 过去7天降雨天数
   df['pressure_7d_std'] = df.groupby('city')['pressure_msl'].transform(
       lambda x: x.rolling(7).std())  # 气压波动性
   ```

3. **气压变化率**（气象学强信号）
   ```python
   # 气压骤降是暴雨的前兆
   df['pressure_change_3d'] = df.groupby('city')['pressure_msl'].transform(
       lambda x: x.shift(3) - x)  # 3天气压变化
   ```

   > **注意**：日级数据中无 `pressure_msl` 列（仅有温度/降水/风速/辐射），需从小时级数据聚合或改用小时级特征工程产出。如果需要气压特征，建议切换到小时级数据并聚合为日级（取日均值）。

4. **季节×城市交互特征**
   ```python
   df['season_city'] = df['season'].astype(str) + '_' + df['city_id'].astype(str)
   ```

### 4.2 类别不平衡处理

| 策略 | 适用场景 | 实现 |
|------|----------|------|
| `class_weight='balanced'` | 默认 | 自动按反频率加权 |
| `class_weight={0:1, 1:3}` | 正样本极少时 | 手动加大正样本权重 |
| SMOTE 过采样 | 极端不平衡 | `imblearn.over_sampling.SMOTE`（需额外依赖） |

> 推荐先用 `class_weight='balanced'`，如果 Recall < 0.5 再考虑 SMOTE。

### 4.3 概率校准（提升软投票质量）

```python
from sklearn.calibration import CalibratedClassifierCV

# 随机森林的概率通常不够校准，可用 Platt Scaling 校准
rf_calibrated = CalibratedClassifierCV(rf_final, cv='prefit', method='sigmoid')
rf_calibrated.fit(X_val, y_val)  # 需要验证集
```

> 校准后的概率更适合软投票/加权平均，因为综合研判层依赖各模型概率的准确性。但需注意：校准需要额外验证集，不能在测试集上校准（会泄露）。如果训练集足够大，可从训练集中再切分出一个校准集。

### 4.4 集成内多样性（与其他模型差异化）

为了使综合研判有价值，随机森林应与其他两个模型产生**不同的错误模式**：

| 差异化策略 | 说明 |
|------------|------|
| 使用不同特征子集 | 随机森林可以天然处理特征选择，但可故意加入更多派生特征 |
| 使用原始未标准化特征 | 随机森林对单调变换不敏感，可用未标准化数据（树模型不需要标准化） |
| 使用全部 43 列 vs 模型3 仅用关键特征 | 让 RF 看到更多特征，DL 看到更精炼的特征 |

> **关键决策：随机森林是否使用标准化后的数据？**
> - 标准化对树模型**无实质影响**（树基于阈值分裂，不受量纲影响）
> - 但使用标准化数据可保持与模型1/模型3的管道一致性
> - **推荐**：使用未标准化的原始数值 + 派生特征 + 编码后的类别ID，排除标准化列（`*_sin`, `*_cos` 可保留，因为它们是周期编码而非标准化）

---

## 五、多人协作接口约定

### 5.1 与模型1（逻辑回归）的接口
- 共享相同的 `X_train, y_train, X_test, y_test`（由 data_loader.py 统一输出）
- 标签构造逻辑一致（`rain_tomorrow` shift）
- 建议共同维护 `data_loader.py` 或各自复制一份

### 5.2 与模型3（Wide & Deep）的接口
- 模型3 使用 Embedding，需要 `city_id` / `country_id` 作为索引
- 随机森林可直接使用 `city_id` / `country_id` 作为数值特征
- 两者的 `predict_proba` 输出格式必须一致：`[P(class_0), P(class_1)]`

### 5.3 与集成层的接口
```python
# 集成层调用方式示例
from models.model_2_random_forest.predict import RFPredictor

rf = RFPredictor()
proba = rf.predict_proba(X_test)     # → [n, 2] 概率矩阵
weight = rf.get_weight()              # → F1 score float
oob = rf.get_oob_score()             # → OOB float
importance = rf.get_feature_importance()  # → {feature: importance}
```

### 5.4 与兜底层的接口
- 随机森林需暴露 `predict_proba` 的原始概率
- 兜底层检查 `max(proba) < 0.55` 时触发规则兜底
- 随机森林可提供"置信度"指标：`max_proba = np.max(proba, axis=1)`

---

## 六、执行顺序与时间估算

| 步骤 | 文件 | 预估时间 | 依赖 |
|------|------|----------|------|
| 1 | `config.py` | 15 min | 无 |
| 2 | `data_loader.py` | 45 min | 特征工程产出（✅已完成） |
| 3 | `train.py` | 60 min | data_loader.py |
| 4 | `evaluate.py` | 45 min | train.py 产出模型 |
| 5 | `predict.py` | 30 min | train.py 产出模型 |
| 6 | `utils.py` | 30 min | 无（可视化函数） |
| 7 | `README.md` | 15 min | 全部完成 |
| 8 | 端到端测试 | 30 min | 全部完成 |
| **合计** | | **~4.5 小时** | |

---

## 七、验证清单

- [ ] `data_loader.py` 能正确加载 train/test CSV 并构造 `rain_tomorrow` 标签
- [ ] 标签无数据泄露（训练集不含2024年数据，测试集不含2015-2023年数据）
- [ ] `train.py` 完成两阶段 GridSearchCV，输出最优参数
- [ ] OOB Score 已记录并展示
- [ ] `evaluate.py` 生成 5 张图表 + 评估报告
- [ ] `feature_importance.png` 展示 Top-20 特征（答辩可解释性）
- [ ] `predict.py` 的 `predict_proba` 返回 `[n, 2]` 概率矩阵
- [ ] 概率输出格式与模型1/模型3一致
- [ ] 模型已 pickle 保存到 `saved_models/rf_classifier.pkl`
- [ ] 所有数据产物已被 `.gitignore` 忽略
