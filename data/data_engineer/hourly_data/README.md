# Hourly Weather Data - 特征工程模块

## 概述

本模块对清洗后的小时级天气数据（`weather_hourly_cleaned.csv`）进行特征工程处理，生成可直接用于机器学习模型的特征数据。

### 输入
- **数据源**: `data/data_clean/cleaned_data/weather_hourly_cleaned.csv`
- **数据规模**: 4,295,928 条记录（49个欧洲城市，2015-2024年逐小时）
- **原始特征**: 18列（城市、国家、经纬度、时间、weather_code、13个气象特征）

### 输出
- **训练集**: `processed_data/train_features.csv` (2015-2023)
- **测试集**: `processed_data/test_features.csv` (2024)
- **预处理对象**: `processed_data/preprocessors/` (scaler、映射字典)

---

## 特征工程策略（与日数据的关键区别）

### 1. 时间特征分解（新增小时维度）

#### 基础时间特征
- `year`, `month`, `day`, `hour`, `day_of_year`, `day_of_week`
- `quarter`, `week_of_year`, `is_weekend`
- `day_period`: 时段分类（morning/forenoon/afternoon/evening/night）

#### 周期性编码（三角函数）
- `month_sin`, `month_cos`: 月份周期 (12)
- `day_of_year_sin`, `day_of_year_cos`: 年周期 (365)
- `day_of_week_sin`, `day_of_week_cos`: 周周期 (7)
- ★ `hour_sin`, `hour_cos`: **小时周期 (24)** — 小时数据特有

### 2. 类别特征编码

比日数据多了 **weather_code**（WMO天气编码，约12种）：
- `city` → `city_id` (49个城市)
- `country` → `country_id` (~28个国家)
- ★ `weather_code` → `weather_code_id` (12种天气现象)

### 3. 派生特征（小时数据专属）

#### 温度相关
- `wind_chill`: 风寒指数（温度≤10°C时生效）
- `heat_index`: 酷暑指数（温度>27°C时生效）
- `is_freezing`, `is_hot`, `temperature_level`

#### 降水相关
- `is_rainy`, `precipitation_intensity`, `solid_precip`, `is_snowy`, `is_snow`

#### 风速风向相关
- `wind_level`, `is_windy`, `is_strong_wind`
- ★ `gust_factor`: 阵风因子（阵风/持续风速比）
- ★ `wind_u`, `wind_v`: 风向矢量分量（北向和东向）

#### 湿度/气压
- `humidity_level`, `is_dry`, `is_humid`
- `pressure_level`

#### 云量/辐射
- `cloud_level`, `is_clear`, `is_overcast`

#### 综合
- `severe_weather_index`, `is_severe_weather`
- `is_high_latitude`, `is_mediterranean`

### 4. 数值特征标准化

使用 **StandardScaler**（Z-score标准化），仅在训练集上 fit。

---

## 运行方式

```bash
cd data/data_engineer/hourly_data
python main.py
```

**注意**：小时数据量大（~430万条），完整运行约需 2-5 分钟。

## 目录结构

```
hourly_data/
├── feature_engineer/
│   ├── __init__.py
│   ├── categorical_encoder.py   # 类别编码（city/country/weather_code）
│   ├── feature_creator.py       # 派生特征创建（小时适配版）
│   ├── numerical_scaler.py      # 数值标准化
│   ├── pipeline.py              # 特征工程流水线
│   └── time_features.py         # 时间特征（含小时周期编码）
├── processed_data/
│   ├── preprocessors/           # scaler.pkl + 映射JSON
│   ├── train_features.csv
│   └── test_features.csv
├── config.py
└── main.py
```
