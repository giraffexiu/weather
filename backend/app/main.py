"""FastAPI 天气预测平台 - 后端入口"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers.predict import router as predict_router

app = FastAPI(
    title="天气预测平台 API",
    description="基于集成学习的欧洲城市天气预测系统",
    version="1.0.0",
)

# CORS 配置：允许前端跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(predict_router)


@app.get("/")
async def root():
    return {
        "message": "天气预测平台 API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
    }
