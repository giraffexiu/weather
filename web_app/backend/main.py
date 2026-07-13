"""
天气预测 Web API 后端
- 样本集49个城市选择
- Open-Meteo API 获取未来预报
- Hourly + Daily Wide & Deep 模型预测
"""
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parent.parent
sys.path.insert(0, str(_PROJECT))
sys.path.insert(0, str(_PROJECT / "ensemble"))

from weather_service import predict, CITY_NAMES, CITIES_CONFIG

app = FastAPI(title="WeatherAI", version="3.0")
app.mount("/static", StaticFiles(directory=_HERE / ".." / "frontend" / "static"), name="static")


@app.get("/")
async def root():
    return HTMLResponse((_HERE / ".." / "frontend" / "templates" / "index.html").read_text())


# ==================== API ====================

class PredictRequest(BaseModel):
    city: str
    target_time: str  # "2025-07-15T12:00"


@app.post("/api/predict")
async def predict_weather(req: PredictRequest):
    """根据城市和目标时间预测天气（Hourly + Daily）"""
    try:
        result = await predict(req.city, req.target_time)
        return result
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"预测失败: {str(e)}")


@app.get("/api/cities")
async def list_cities():
    """返回样本集49个城市列表"""
    return {
        "cities": [
            {"name": c, "country": CITIES_CONFIG[c]["country"],
             "lat": CITIES_CONFIG[c]["latitude"], "lon": CITIES_CONFIG[c]["longitude"]}
            for c in CITY_NAMES
        ]
    }


@app.get("/api/statistics")
async def get_statistics():
    return {
        "model": "Wide & Deep (Hourly + Daily)",
        "cities": len(CITY_NAMES),
        "hourly_targets": {
            "temperature_2m": {"unit": "°C", "mae": 0.59, "r2": 0.991},
            "apparent_temperature": {"unit": "°C", "mae": 0.75, "r2": 0.990},
            "relative_humidity_2m": {"unit": "%", "mae": 2.65, "r2": 0.953},
            "wind_speed_10m": {"unit": "m/s", "mae": 1.61, "r2": 0.911},
            "precipitation": {"unit": "mm", "mae": 0.10, "r2": 0.394},
        },
        "daily_targets": {
            "temperature_2m_mean": {"unit": "°C", "mae": 0.59},
            "precipitation_sum": {"unit": "mm", "mae": 0.10},
            "wind_speed_10m_max": {"unit": "m/s", "mae": 1.61},
        },
    }


if __name__ == "__main__":
    import uvicorn
    print(f"城市 ({len(CITY_NAMES)}): {', '.join(CITY_NAMES[:5])}...")
    uvicorn.run(app, host="0.0.0.0", port=8080)
