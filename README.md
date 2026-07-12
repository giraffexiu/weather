# 欧洲城市天气数据集

从Open-Meteo API获取欧洲城市历史天气数据的项目。

## 📋 环境要求

- **Python**: 3.8+（推荐 3.9+）
- **依赖**: `requests`, `pandas`

## 🚀 快速开始

### 1. 创建虚拟环境

```bash
# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 获取数据

```bash
cd data/fetchdata
python fetch_weather_data.py
```

## 📁 项目结构

```
weather/
├── README.md                    # 项目说明
├── requirements.txt             # Python依赖
├── .gitignore                   # Git忽略配置
└── data/
    ├── *.csv                    # 生成的数据集（已忽略）
    └── fetchdata/
        ├── README.md            # 详细使用说明
        ├── DATA_COLLECTION_PLAN.md  # 数据采集方案
        ├── fetch_weather_data.py    # 主脚本（50城市，25年）
        └── fetch_weather_data_demo.py  # 演示脚本（15城市，5年）
```

## 📊 数据说明

### 主脚本 (fetch_weather_data.py)
- **城市**: 50个欧洲主要城市
- **时间跨度**: 2000-2024（25年）
- **变量**: 38个小时级 + 19个每日级
- **预计大小**: ~1GB

### 演示脚本 (fetch_weather_data_demo.py)
- **城市**: 15个欧洲主要城市
- **时间跨度**: 2020-2024（5年）
- **变量**: 10个小时级
- **预计大小**: ~30-50MB

## ⚠️ 注意事项

1. **虚拟环境**: 必须使用虚拟环境（macOS Homebrew Python限制）
2. **CSV文件**: 生成的数据集不会提交到Git
3. **执行时间**: 主脚本需要10-20分钟
4. **网络要求**: 需要稳定的网络连接

## 📚 详细文档

- [数据获取详细说明](data/fetchdata/README.md)
- [数据采集方案](data/fetchdata/DATA_COLLECTION_PLAN.md)

## 🔗 数据来源

- [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)
- 基于ERA5再分析数据集

---

**最后更新**: 2026-07-12
