"""
Daily Ensemble - 软投票集成模块

集成 Model 1 (Linear) 和 Model 3 (Wide & Deep) 的预测结果
"""

__version__ = "1.0.0"
__author__ = "Weather Prediction Team"

from .config import (
    MODEL1_CLASSIFICATION_TASKS,
    MODEL1_REGRESSION_TASKS,
    ENSEMBLE_TASKS,
    DEVICE
)

from .probability_converter import ProbabilityConverter
from .model_wrapper import Model1Wrapper, Model3Wrapper
from .soft_voting_ensemble import SoftVotingEnsemble

__all__ = [
    'ProbabilityConverter',
    'Model1Wrapper',
    'Model3Wrapper',
    'SoftVotingEnsemble',
    'MODEL1_CLASSIFICATION_TASKS',
    'MODEL1_REGRESSION_TASKS',
    'ENSEMBLE_TASKS',
    'DEVICE'
]
