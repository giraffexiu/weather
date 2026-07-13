# 天气数据获取工具 - 技术文档

欧洲城市历史天气数据获取脚本的详细技术说明。

## 📝 脚本说明

### fetch_weather_data.py（主脚本）
**完整数据集** - 50个城市，25年数据，~1GB

**配置:**
- 时间: 2000-2024（25年）
- 城市: 50个欧洲主要城市
- 小时级变量: 38个
- 每日级变量: 19个
- 请求延迟: 3秒（避免429错误）
- 重试机制: 最多5次
- 断点续传: 支持

**运行:**
```bash
python fetch_weather_data.py
```

**输出到:** `../` (即 `data/` 目录)
- `weather_hourly_2000-01-01_to_2024-12-31.csv` (~600-700MB)
- `weather_daily_2000-01-01_to_2024-12-31.csv` (~60-80MB)

### fetch_weather_data_demo.py（演示脚本）
**小型数据集** - 15个城市，5年数据，~30-50MB

**配置:**
- 时间: 2020-2024（5年）
- 城市: 15个欧洲主要城市
- 变量: 10个小时级
- 请求延迟: 1秒

**运行:**
```bash
python fetch_weather_data_demo.py
```

**输出到:** `../` (即 `data/` 目录)
- `weather_data_2020-01-01_to_2024-12-31.csv` (~30-50MB)

## 📋 环境要求

- **Python版本**: Python 3.8+（推荐 3.9+）
- **依赖包**: `requests >= 2.31.0`, `pandas >= 2.0.0`

## 🚀 使用步骤

## 🚀 使用步骤

### 1. 确保虚拟环境已激活

```bash
# 从项目根目录
source venv/bin/activate  # macOS/Linux
```

### 2. 运行数据获取脚本

```bash
# 进入fetchdata目录
cd data/fetchdata

# 运行主脚本（1GB数据集）
python fetch_weather_data.py

# 或运行演示脚本（小数据集）
python fetch_weather_data_demo.py
```

## 🌍 城市列表

### 主脚本（50个城市）
包含西欧、北欧、东欧、南欧、巴尔干半岛等地区的主要城市。
详见 [DATA_COLLECTION_PLAN.md](DATA_COLLECTION_PLAN.md)

### 演示脚本（15个城市）

| 城市 | 纬度 | 经度 |
|------|------|------|
| London | 51.5072 | -0.1276 |
| Paris | 48.8566 | 2.3522 |
| Berlin | 52.5200 | 13.4050 |
| Madrid | 40.4168 | -3.7038 |
| Rome | 41.9028 | 12.4964 |
| Amsterdam | 52.3676 | 4.9041 |
| Brussels | 50.8503 | 4.3517 |
| Vienna | 48.2082 | 16.3738 |
| Zurich | 47.3769 | 8.5417 |
| Stockholm | 59.3293 | 18.0686 |
| Oslo | 59.9139 | 10.7522 |
| Copenhagen | 55.6761 | 12.5683 |
| Helsinki | 60.1699 | 24.9384 |
| Lisbon | 38.7223 | -9.1393 |
| Milan | 45.4642 | 9.1900 |

## 📊 天气数据变量

### 主脚本变量（57个）
- **小时级**: 38个（温度、湿度、气压、降水、云量、风、辐射、土壤温湿度等）
- **每日级**: 19个（日最高/最低/平均温度、降水量、日照时长等）

详见 [DATA_COLLECTION_PLAN.md](DATA_COLLECTION_PLAN.md)

### 演示脚本变量（10个）

| 变量名 | 描述 | 单位 |
|--------|------|------|
| temperature_2m | 2米高度气温 | °C |
| relative_humidity_2m | 2米高度相对湿度 | % |
| pressure_msl | 海平面气压 | hPa |
| precipitation | 降水量 | mm |
| wind_speed_10m | 10米高度风速 | km/h |
| wind_direction_10m | 10米高度风向 | ° |
| cloud_cover | 云量 | % |
| shortwave_radiation | 短波辐射 | W/m² |
| uv_index | 紫外线指数 | - |
| visibility | 能见度 | m |

## 📁 输出文件

CSV文件包含以下列：
- `city` - 城市名称
- `country` - 国家（仅主脚本）
- `latitude` - 纬度
- `longitude` - 经度
- `data_type` - 数据类型：hourly/daily（仅主脚本）
- `time` - 时间戳（ISO 8601格式）
- 各天气变量列

## 🔧 解决429错误

主脚本包含完整的429错误处理机制：
- 请求间隔：3秒
- 最大重试：5次
- 指数退避策略
- 断点续传支持

详见 [DATA_COLLECTION_PLAN.md](DATA_COLLECTION_PLAN.md)

## 🛠️ 故障排除

### ModuleNotFoundError
确保已激活虚拟环境并安装依赖：
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### externally-managed-environment (macOS)
必须使用虚拟环境（见项目根目录README.md）

### 429 Too Many Requests
主脚本已内置处理机制。演示脚本如遇到，手动增加延迟时间。

## 📚 相关文档

- [项目主README](../../README.md) - 项目概述和环境配置
- [DATA_COLLECTION_PLAN.md](DATA_COLLECTION_PLAN.md) - 详细数据采集方案
- [Open-Meteo API文档](https://open-meteo.com/en/docs/historical-weather-api)

---

**最后更新**: 2026-07-12
