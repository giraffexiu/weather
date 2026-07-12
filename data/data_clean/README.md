# 天气数据清洗系统

## 目录结构
```
data_clean/
├── main.py              # 主入口脚本
├── config.py            # 配置文件
├── missing_handler.py   # 缺失值处理模块
├── noise_remover.py     # 噪声列移除模块
├── outlier_handler.py   # 异常值处理模块
└── cleaned_data/        # 清洗后数据输出目录
```

## 功能模块

### 1. 缺失值处理 (missing_handler.py)
- 分析并统计缺失值
- 温度字段：线性插值
- 降水字段：填充为0
- 其他字段：前向/后向填充

### 2. 噪声列移除 (noise_remover.py)
- 识别乱码列
- 移除单一值列
- 删除全空列
- 去除重复列

### 3. 异常值处理 (outlier_handler.py)
- 负值截断（非负字段）
- 极值裁剪（温度、降水等）
- 业务逻辑修正

## 使用方法

```bash
cd data/data_clean
python3 main.py
```

## 输出结果
- `cleaned_data/weather_daily_cleaned.csv` - 日数据（清洗后）
- `cleaned_data/weather_hourly_cleaned.csv` - 小时数据（清洗后）
