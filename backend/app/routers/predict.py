"""预测相关 API 路由"""
from fastapi import APIRouter, HTTPException
from ..schemas import (
    PredictRequest,
    PredictionResponse,
    CityListResponse,
)
from ..services.predictor import get_predictor

router = APIRouter(prefix="/api", tags=["预测"])


@router.post("/predict", response_model=PredictionResponse)
async def predict_weather(request: PredictRequest):
    """天气预测接口

    接受城市名称和目标时间，返回完整的天气预测结果，
    包括当前天气、未来7天每日预测、未来24小时逐小时预测和模型解释。

    Args:
        request: 预测请求，包含 city 和可选的 target_time

    Returns:
        PredictionResponse: 完整预测结果
    """
    try:
        predictor = get_predictor()
        result = predictor.predict(
            city=request.city,
            target_time=request.target_time
        )
        return PredictionResponse(**result)
    except KeyError:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的城市: {request.city}。请使用 /api/cities 查看支持的城市列表。"
        )
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"日期格式错误: {str(e)}。请使用 YYYY-MM-DD 格式。"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"预测服务内部错误: {str(e)}"
        )


@router.get("/cities", response_model=CityListResponse)
async def list_cities():
    """获取支持的城市列表"""
    predictor = get_predictor()
    cities = predictor.get_cities()
    return CityListResponse(cities=cities, count=len(cities))


@router.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "ok", "service": "weather-prediction-api"}
