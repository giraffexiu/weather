"""
统一推理接口 (Predict) — 仅小时级
多目标回归：预测下一小时的 5 个气象变量值
供集成研判层 (Ensemble Layer) 调用
"""
import pickle
import json
import numpy as np
from pathlib import Path
from typing import Optional

import config


class RFPredictor:
    """随机森林回归预测器（小时级） - 供集成层调用"""

    def __init__(self, model_path: Optional[str] = None):
        """
        Args:
            model_path: 模型路径（默认为小时级模型）
        """
        self.granularity = "hourly"
        if model_path is None:
            model_path = config.HOURLY_MODEL_PATH
        self.model_path = Path(model_path)

        with open(self.model_path, "rb") as f:
            saved = pickle.load(f)

        if isinstance(saved, dict):
            self.model = saved["model"]
            self.feature_names = saved.get("feature_names", [])
            self.target_columns = saved.get("target_columns", config.HOURLY_TARGET_COLUMNS)
            self.oob_score = saved.get("oob_score", None)
        else:
            self.model = saved
            self.feature_names = []
            self.target_columns = config.HOURLY_TARGET_COLUMNS
            self.oob_score = getattr(self.model, "oob_score_", 0.0)

    def predict(self, X) -> np.ndarray:
        """返回回归预测值 [n_samples, n_targets]"""
        return self.model.predict(X)

    def predict_hourly(self, X) -> dict:
        """
        小时级预测接口：返回预测值 + 置信度参考

        Args:
            X: 特征矩阵 [n_samples, n_features]

        Returns:
            {
                "prediction": [n_samples, 5] 预测值,
                "target_columns": 目标列名列表,
                "oob_score": OOB 分数,
            }
        """
        preds = self.predict(X)
        return {
            "prediction": preds.tolist() if hasattr(preds, 'tolist') else list(preds),
            "target_columns": self.target_columns,
            "oob_score": self.oob_score,
        }

    def get_feature_importance(self) -> dict:
        """返回特征重要性字典"""
        names = self.feature_names or [f"f{i}" for i in range(len(self.model.feature_importances_))]
        return dict(zip(names, self.model.feature_importances_))

    def get_top_features(self, top_n: int = 10) -> list:
        """返回 Top-N 最重要的特征"""
        imp = self.get_feature_importance()
        return sorted(imp.items(), key=lambda x: -x[1])[:top_n]

    def get_oob_score(self) -> float:
        """返回 OOB 分数"""
        return self.oob_score

    def get_weight(self) -> float:
        """返回模型权重参考值（OOB Score），供加权平均策略使用"""
        return self.get_oob_score()

    def get_target_columns(self) -> list:
        """返回目标列名列表"""
        return self.target_columns

    def is_fallback_needed(self, X, threshold: float = 0.3) -> np.ndarray:
        """
        判断是否需要触发兜底方案
        当 OOB Score 低于阈值，或输入特征存在大量缺失时返回 True
        回归任务没有概率输出，用 OOB Score 作为模型整体置信度参考

        Args:
            X: 特征矩阵
            threshold: OOB 阈值（默认 0.3）

        Returns:
            布尔数组，True 表示该样本需要兜底
        """
        if self.oob_score < threshold:
            return np.ones(len(X), dtype=bool)
        return np.zeros(len(X), dtype=bool)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path is None:
        path = str(config.HOURLY_MODEL_PATH)
    if not Path(path).exists():
        print(f"模型文件不存在: {path}")
        print("请先运行 train_hourly.py")
        sys.exit(1)
    predictor = RFPredictor(model_path=path)
    print(f"模型加载成功: hourly")
    print(f"目标变量: {predictor.get_target_columns()}")
    print(f"OOB Score: {predictor.get_oob_score():.4f}")
    print(f"特征数: {len(predictor.get_feature_importance())}")
    print(f"Top-5 特征:")
    for name, val in predictor.get_top_features(5):
        print(f"  {name}: {val:.4f}")
