"""
Hourly Ensemble - Wide & Deep 集成模块

为小时级天气预测提供模型包装和集成预测
"""

__version__ = "1.0.0"
__author__ = "Weather Prediction Team"

from .config import (
    HOUR_TARGET_COLUMNS,
    MODEL3_TARGET_INDICES,
    DEVICE,
)

from .probability_converter import ProbabilityConverter
from .model_wrapper import HourModelWrapper

__all__ = [
    'ProbabilityConverter',
    'HourModelWrapper',
    'HOUR_TARGET_COLUMNS',
    'MODEL3_TARGET_INDICES',
    'DEVICE',
]
