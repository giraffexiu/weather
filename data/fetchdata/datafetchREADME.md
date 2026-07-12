# 欧洲城市天气数据获取工具

从Open-Meteo API获取欧洲15个主要城市的历史天气数据（2020-2024年）。

## 城市列表

- London (伦敦)
- Paris (巴黎)
- Berlin (柏林)
- Madrid (马德里)
- Rome (罗马)
- Amsterdam (阿姆斯特丹)
- Brussels (布鲁塞尔)
- Vienna (维也纳)
- Zurich (苏黎世)
- Stockholm (斯德哥尔摩)
- Oslo (奥斯陆)
- Copenhagen (哥本哈根)
- Helsinki (赫尔辛基)
- Lisbon (里斯本)
- Milan (米兰)

## 数据变量

- `temperature_2m` - 2米高度气温 (°C)
- `relative_humidity_2m` - 2米高度相对湿度 (%)
- `pressure_msl` - 海平面气压 (hPa)
- `precipitation` - 降水量 (mm)
- `wind_speed_10m` - 10米高度风速 (km/h)
- `wind_direction_10m` - 10米高度风向 (°)
- `cloud_cover` - 云量 (%)
- `shortwave_radiation` - 短波辐射 (W/m²)
- `uv_index` - 紫外线指数
- `visibility` - 能见度 (m)

## 安装依赖

```bash
pip install -r requirements.txt
```

或者单独安装：

```bash
pip install requests pandas
```

## 使用方法

直接运行脚本：

```bash
python fetch_weather_data.py
```

脚本会：
1. 依次获取15个城市的天气数据
2. 将所有数据合并到一个CSV文件
3. 保存到 `data/fetchdata/` 目录

## 输出文件

生成的CSV文件包含以下列：

- `city` - 城市名称
- `latitude` - 纬度
- `longitude` - 经度
- `time` - 时间戳（ISO格式）
- `temperature_2m` - 气温
- `relative_humidity_2m` - 相对湿度
- `pressure_msl` - 气压
- `precipitation` - 降水量
- `wind_speed_10m` - 风速
- `wind_direction_10m` - 风向
- `cloud_cover` - 云量
- `shortwave_radiation` - 短波辐射
- `uv_index` - 紫外线指数
- `visibility` - 能见度

## 注意事项

- 数据量较大，完整下载需要几分钟时间
- 脚本包含1秒延迟以避免API限流
- 最终CSV文件大小约为几百MB
- 如需修改时间范围或城市，请编辑脚本中的配置部分

## 数据来源

数据来自 [Open-Meteo ERA5 Archive API](https://open-meteo.com/en/docs/historical-weather-api)
