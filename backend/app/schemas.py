"""Pydantic 数据模型定义"""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


# ==================== 请求模型 ====================

class PredictRequest(BaseModel):
    """预测请求"""
    city: str = Field(
        ...,
        description="城市名称，如 London, Paris, Berlin 等",
        examples=["London"]
    )
    target_time: Optional[str] = Field(
        default=None,
        description="目标预测时间，格式: YYYY-MM-DD，默认为明天",
        examples=["2024-06-15"]
    )

    class Config:
        json_schema_extra = {
            "example": {
                "city": "London",
                "target_time": "2024-06-15"
            }
        }


class CityListResponse(BaseModel):
    """城市列表响应"""
    cities: List[str]
    count: int


# ==================== 响应模型 ====================

class DailyPrediction(BaseModel):
    """每日天气预测"""
    date: str
    temperature_max: float
    temperature_min: float
    temperature_mean: float
    apparent_temperature: float
    precipitation: float
    rain_amount: float
    snow_amount: float
    wind_speed: float
    humidity: float
    weather: str
    rain_probability: float
    snow_probability: float
    severe_probability: float


class HourlyPrediction(BaseModel):
    """逐小时天气预测"""
    time: str
    temperature: float
    apparent_temperature: float
    humidity: float
    precipitation: float
    wind_speed: float
    rain_probability: float
    weather: str


class ModelExplanation(BaseModel):
    """模型可解释性信息"""
    feature_importance: List[dict]
    model_confidence: float
    weather_summary: str
    contributing_factors: List[str]
    model_weights: dict


class PredictionResponse(BaseModel):
    """完整预测响应"""
    city: str
    country: str
    latitude: float
    longitude: float
    target_time: str
    current: HourlyPrediction
    daily: List[DailyPrediction]
    hourly: List[HourlyPrediction]
    explanation: ModelExplanation
    confidence: float

    class Config:
        json_schema_extra = {
            "example": {
                "city": "London",
                "country": "UK",
                "latitude": 51.5072,
                "longitude": -0.1276,
                "target_time": "2024-06-15",
                "current": {
                    "time": "2024-06-15T12:00:00",
                    "temperature": 18.5,
                    "apparent_temperature": 17.2,
                    "humidity": 72.0,
                    "precipitation": 0.5,
                    "wind_speed": 12.3,
                    "rain_probability": 0.35,
                    "weather": "Cloudy"
                },
                "daily": [],
                "hourly": [],
                "explanation": {
                    "feature_importance": [],
                    "model_confidence": 0.85,
                    "weather_summary": "",
                    "contributing_factors": [],
                    "model_weights": {}
                },
                "confidence": 0.85
            }
        }
