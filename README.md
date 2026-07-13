# 欧洲城市天气预测平台

基于集成学习（Ensemble Learning）的欧洲城市天气预测系统，包含数据管道、模型训练和 Web 预测平台。

## 项目架构

```
weather/
├── data/                          # 数据层
│   ├── fetch_data/                #   数据采集（Open-Meteo API）
│   ├── data_clean/                #   数据清洗
│   └── data_engineer/             #   特征工程（daily + hourly）
├── models/                        # 模型层
│   ├── model_1_linear/            #   线性基准模型（Logistic/Ridge）
│   └── model_3_deep_learning/     #   Wide & Deep 深度学习（daily + hourly）
├── ensemble/                      # 集成层
│   ├── daily_ensemble/            #   日级软投票集成
│   └── hourly_ensemble/           #   小时级集成
├── backend/                       # 后端 API（FastAPI）
│   ├── app/
│   │   ├── main.py                #   FastAPI 入口
│   │   ├── schemas.py             #   Pydantic 数据模型
│   │   ├── routers/predict.py     #   API 路由
│   │   └── services/predictor.py  #   预测服务核心
│   ├── run.py                     #   启动脚本
│   └── requirements.txt
├── frontend/                      # 前端（Vue3 + TypeScript）
│   ├── src/
│   │   ├── App.vue                #   主布局
│   │   ├── main.ts                #   入口
│   │   ├── api/index.ts           #   API 客户端
│   │   ├── types/index.ts         #   TypeScript 类型
│   │   ├── router/index.ts        #   路由配置
│   │   ├── components/            #   公共组件
│   │   │   └── PredictionForm.vue #   预测表单
│   │   └── views/                 #   页面
│   │       ├── Dashboard.vue      #   仪表盘
│   │       ├── Hourly.vue         #   24小时预测
│   │       ├── Daily.vue          #   7天预测
│   │       └── ModelExplanation.vue # 模型解释
│   ├── index.html
│   ├── vite.config.ts
│   └── package.json
└── README.md
```

## 技术栈

### 后端
- **FastAPI** - 高性能 Python Web 框架
- **Pydantic** - 数据验证
- **Uvicorn** - ASGI 服务器
- **NumPy / Pandas** - 数据处理

### 前端
- **Vue 3** - 渐进式前端框架
- **TypeScript** - 类型安全
- **Vite** - 构建工具
- **Element Plus** - UI 组件库
- **ECharts** - 数据可视化
- **Axios** - HTTP 客户端
- **Vue Router** - 路由管理

### 模型
- **Scikit-learn** - 线性模型（Logistic Regression / Ridge）
- **PyTorch** - 深度学习（Wide & Deep 架构）
- **软投票集成** - 多模型融合

## 环境要求

- Python 3.9+
- Node.js 18+
- npm 9+

## 快速开始

### 1. 后端启动

```bash
cd backend

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 启动后端（默认 http://localhost:8000）
python run.py
```

API 文档：
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### 2. 前端启动

```bash
cd frontend

# 安装依赖
npm install

# 启动开发服务器（默认 http://localhost:3000）
npm run dev

# 生产构建
npm run build
```

### 3. 访问

打开浏览器访问 **http://localhost:3000**，输入城市名称和日期开始预测。

## API 接口

### POST /api/predict

天气预测接口。

**请求：**
```json
{
  "city": "London",
  "target_time": "2024-06-15"
}
```

**响应：**
```json
{
  "city": "London",
  "country": "UK",
  "latitude": 51.5072,
  "longitude": -0.1276,
  "target_time": "2024-06-15",
  "current": {
    "temperature": 18.5,
    "humidity": 72.0,
    "wind_speed": 12.3,
    "rain_probability": 0.35,
    "weather": "Cloudy"
  },
  "daily": [ ... ],
  "hourly": [ ... ],
  "explanation": { ... },
  "confidence": 0.85
}
```

### GET /api/cities

获取支持的城市列表。

### GET /api/health

健康检查。

## 支持的城市（49个）

| 区域 | 城市 |
|------|------|
| 西欧 | London, Paris, Berlin, Amsterdam, Brussels, Dublin, Luxembourg |
| 南欧 | Madrid, Barcelona, Rome, Milan, Lisbon, Athens, Porto, Valencia, Seville, Naples, Turin, Florence |
| 中欧 | Vienna, Zurich, Munich, Prague, Budapest, Geneva, Cologne, Frankfurt, Hamburg, Lyon, Marseille, Toulouse, Nice |
| 北欧 | Stockholm, Oslo, Copenhagen, Helsinki, Reykjavik |
| 东欧 | Warsaw, Bucharest, Sofia, Zagreb, Ljubljana, Belgrade |
| 波罗的海 | Riga, Vilnius, Tallinn |
| 英国 | Edinburgh, Manchester, Birmingham |

## 前端页面

| 页面 | 路由 | 说明 |
|------|------|------|
| Dashboard | `/` | 当前天气、核心指标卡片、7天概览、特征重要性 |
| Hourly | `/hourly` | 24小时温度折线图、降雨概率柱状图、风速湿度图 |
| Daily | `/daily` | 7天详细预报表格、温度范围对比图 |
| Explanation | `/explanation` | 模型置信度、影响因素、特征重要性、集成权重 |

## 模型集成策略

- **Daily 预测**：Model 1（线性模型）+ Model 3（Wide & Deep）软投票集成，权重基于验证集 R²/F1 动态分配
- **Hourly 预测**：单一 Wide & Deep 模型，同时处理回归和分类任务
- **后处理**：概率转换（回归值 → 分类概率）、物理约束校正
