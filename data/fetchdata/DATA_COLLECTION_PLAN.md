# 天气数据采集方案 - 1GB目标

## 📊 数据规模预估

### 当前数据（原脚本）
- **城市**: 5个（实际获取）
- **时间**: 2020-2024 (5年)
- **变量**: 10个
- **文件大小**: 16 MB

### 增强版数据（新脚本）
- **城市**: 50个欧洲主要城市
- **时间**: 1990-2024 (35年)
- **小时级变量**: 38个
- **每日级变量**: 19个
- **预计文件大小**: **800MB - 1.2GB**

## 🎯 数据量计算

```
小时级数据:
- 50城市 × 35年 × 365天 × 24小时 = 15,330,000 条记录
- 15,330,000条 × (38变量 + 5元数据列) ≈ 约 800-900 MB

每日数据:
- 50城市 × 35年 × 365天 = 638,750 条记录  
- 638,750条 × (19变量 + 5元数据列) ≈ 约 100-150 MB

总计: 约 900MB - 1.05GB ✅
```

## 🌍 新增城市列表（50个）

### 西欧 (11个)
London, Paris, Berlin, Madrid, Rome, Amsterdam, Brussels, Vienna, Zurich, Lisbon, Milan

### 北欧 (5个)
Stockholm, Oslo, Copenhagen, Helsinki, Reykjavik

### 东欧 (6个)
Warsaw, Prague, Budapest, Bucharest, Sofia, Athens

### 南欧 (6个)
Barcelona, Valencia, Seville, Naples, Turin, Florence

### 德国城市 (4个)
Munich, Hamburg, Frankfurt, Cologne

### 英国城市 (3个)
Edinburgh, Manchester, Birmingham

### 法国城市 (4个)
Lyon, Marseille, Toulouse, Nice

### 波罗的海 (3个)
Tallinn, Riga, Vilnius

### 巴尔干半岛 (3个)
Belgrade, Zagreb, Ljubljana

### 其他 (5个)
Dublin, Porto, Geneva, Basel, Luxembourg

## 📈 天气变量详情

### 小时级变量 (38个)

#### 温度 (3个)
- temperature_2m - 2米气温
- apparent_temperature - 体感温度
- dew_point_2m - 露点温度

#### 湿度和气压 (4个)
- relative_humidity_2m - 相对湿度
- pressure_msl - 海平面气压
- surface_pressure - 地面气压
- vapour_pressure_deficit - 蒸汽压差

#### 降水 (4个)
- precipitation - 总降水量
- rain - 降雨量
- snowfall - 降雪量
- snow_depth - 积雪深度

#### 云量 (4个)
- cloud_cover - 总云量
- cloud_cover_low - 低云量
- cloud_cover_mid - 中云量
- cloud_cover_high - 高云量

#### 风 (5个)
- wind_speed_10m - 10米风速
- wind_speed_100m - 100米风速
- wind_direction_10m - 10米风向
- wind_direction_100m - 100米风向
- wind_gusts_10m - 10米阵风

#### 太阳辐射 (5个)
- shortwave_radiation - 短波辐射
- direct_radiation - 直接辐射
- direct_normal_irradiance - 直接法向辐照度
- diffuse_radiation - 漫射辐射
- sunshine_duration - 日照时长

#### 土壤温度 (4个)
- soil_temperature_0_to_7cm
- soil_temperature_7_to_28cm
- soil_temperature_28_to_100cm
- soil_temperature_100_to_255cm

#### 土壤湿度 (4个)
- soil_moisture_0_to_7cm
- soil_moisture_7_to_28cm
- soil_moisture_28_to_100cm
- soil_moisture_100_to_255cm

#### 其他 (2个)
- et0_fao_evapotranspiration - 蒸散发
- weather_code - 天气代码

### 每日级变量 (19个)

- temperature_2m_max/min/mean - 最高/最低/平均气温
- apparent_temperature_max/min/mean - 体感温度
- precipitation_sum - 总降水量
- rain_sum - 降雨量
- snowfall_sum - 降雪量
- precipitation_hours - 降水小时数
- sunrise/sunset - 日出日落时间
- sunshine_duration - 日照时长
- daylight_duration - 日光时长
- wind_speed_10m_max - 最大风速
- wind_gusts_10m_max - 最大阵风
- wind_direction_10m_dominant - 主导风向
- shortwave_radiation_sum - 辐射总量
- et0_fao_evapotranspiration - 蒸散发

## 🔧 解决429错误的关键措施

### 问题原因
Open-Meteo API 有请求频率限制，快速连续请求会触发 429 (Too Many Requests) 错误。

### 解决方案

1. **增加请求间隔**
   ```python
   REQUEST_DELAY = 3  # 每次请求间隔3秒
   ```

2. **智能重试机制**
   ```python
   MAX_RETRIES = 5  # 最大重试5次
   RETRY_DELAY = 15  # 重试时等待15秒
   ```

3. **指数退避**
   - 第1次重试: 等待15秒
   - 第2次重试: 等待30秒
   - 第3次重试: 等待45秒
   - 以此类推

4. **断点续传**
   - 自动保存进度到 `progress_checkpoint.json`
   - 中断后重新运行会从断点继续
   - 完成后自动删除检查点文件

5. **更长的超时时间**
   ```python
   TIMEOUT = 120  # 120秒超时
   ```

## 🚀 使用说明

### 运行增强版脚本

```bash
# 确保虚拟环境已激活
source venv/bin/activate

# 运行增强版脚本
python fetch_weather_data_enhanced.py
```

### 预计执行时间

```
50个城市 × (3秒请求 + 3秒延迟) = 300秒 ≈ 5分钟（无错误情况）
考虑重试和网络延迟: 10-20分钟
```

### 输出文件

1. **weather_hourly_1990-01-01_to_2024-12-31.csv**
   - 小时级数据
   - 约 800-900 MB

2. **weather_daily_1990-01-01_to_2024-12-31.csv**
   - 每日聚合数据
   - 约 100-150 MB

3. **progress_checkpoint.json** (临时文件)
   - 进度检查点
   - 完成后自动删除

## 📝 数据字段说明

### CSV列结构

#### 元数据列 (5列)
- `city` - 城市名称
- `country` - 国家
- `latitude` - 纬度
- `longitude` - 经度
- `data_type` - 数据类型 (hourly/daily)

#### 时间列
- `time` - 时间戳 (ISO 8601格式)

#### 天气变量列
- 根据hourly/daily类型包含相应的天气变量

## ⚠️ 注意事项

1. **网络稳定性**: 确保网络连接稳定，整个过程需要10-20分钟
2. **磁盘空间**: 确保至少有 2GB 可用空间
3. **不要中断**: 如果中断，重新运行会从断点继续
4. **API限制**: 遵守Open-Meteo的使用条款
5. **数据质量**: 
   - ERA5数据从1990年开始质量较好
   - 土壤数据在某些地区可能不完整
   - 积雪深度可能被高估

## 🔍 数据验证

运行完成后验证：

```bash
# 查看文件大小
ls -lh data/fetchdata/*.csv

# 统计记录数
wc -l data/fetchdata/weather_hourly_*.csv
wc -l data/fetchdata/weather_daily_*.csv

# 查看城市覆盖
cut -d',' -f1 data/fetchdata/weather_hourly_*.csv | sort | uniq -c
```

## 📚 API文档

- [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)
- [ERA5 Dataset](https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5)

## 🎓 数据使用建议

### 适合的分析场景
- 气候变化趋势分析
- 城市间气候对比
- 极端天气事件研究
- 机器学习天气预测模型
- 能源消耗与天气关系分析

### 数据预处理建议
- 检查缺失值并适当填充
- 处理异常值
- 时区标准化（已设置为GMT）
- 根据需要进行数据聚合

---

**创建日期**: 2026-07-12  
**预计数据量**: 900MB - 1.05GB  
**预计执行时间**: 10-20分钟
