"""
统一推理接口 (Predict) — 仅小时级
供集成研判层 (Ensemble Layer) 调用
"""
import pickle
import json
import numpy as np
from pathlib import Path
from typing import Optional

import config


class RFPredictor:
    """随机森林预测器（小时级） - 供集成层调用"""

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
            self.label_encoder = saved.get("label_encoder", None)
            self.classes = saved.get("classes", config.WEATHER_CATEGORIES)
        else:
            self.model = saved
            self.feature_names = []
            self.label_encoder = None
            self.classes = list(self.model.classes_)

    def predict_proba(self, X) -> np.ndarray:
        """返回概率输出 [n_samples, 6] — 供软投票/Stacking使用"""
        return self.model.predict_proba(X)

    def predict(self, X) -> np.ndarray:
        """返回类别预测 [n_samples]（类别名字符串）"""
        return self.model.predict(X)

    def predict_hourly(self, X) -> dict:
        """
        小时级预测接口：返回预测类别 + 概率 + 置信度

        Args:
            X: 特征矩阵 [n_samples, n_features]

        Returns:
            {
                "prediction": 类别名列表,
                "probabilities": [n, 6] 概率矩阵,
                "confidence": 每条样本最大概率,
                "classes": 类别列表,
            }
        """
        proba = self.predict_proba(X)
        preds = self.predict(X)
        confidence = np.max(proba, axis=1)

        return {
            "prediction": preds.tolist() if hasattr(preds, 'tolist') else list(preds),
            "probabilities": proba.tolist() if hasattr(proba, 'tolist') else list(proba),
            "confidence": confidence.tolist() if hasattr(confidence, 'tolist') else list(confidence),
            "classes": self.classes,
        }

    def get_feature_importance(self) -> dict:
        """返回特征重要性字典 — 供可解释性展示"""
        names = self.feature_names or [f"f{i}" for i in range(len(self.model.feature_importances_))]
        return dict(zip(names, self.model.feature_importances_))

    def get_top_features(self, top_n: int = 10) -> list:
        """返回 Top-N 最重要的特征"""
        imp = self.get_feature_importance()
        return sorted(imp.items(), key=lambda x: -x[1])[:top_n]

    def get_oob_score(self) -> float:
        """返回 OOB 分数"""
        return getattr(self.model, "oob_score_", 0.0)

    def get_weight(self) -> float:
        """返回模型权重参考值（OOB Score），供加权平均策略使用"""
        return self.get_oob_score()

    def get_max_confidence(self, X) -> np.ndarray:
        """返回每条样本的最大概率（供兜底层检查阈值）"""
        return np.max(self.predict_proba(X), axis=1)

    def get_classes(self) -> list:
        """返回类别列表"""
        return self.classes

    def is_fallback_needed(self, X, threshold: float = 0.55) -> np.ndarray:
        """
        判断是否需要触发兜底方案
        当所有样本的最大概率都低于阈值时返回 True

        Args:
            X: 特征矩阵
            threshold: 置信度阈值（默认 0.55）

        Returns:
            布尔数组，True 表示该样本需要兜底
        """
        return self.get_max_confidence(X) < threshold


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
    print(f"类别: {predictor.get_classes()}")
    print(f"OOB Score: {predictor.get_oob_score():.4f}")
    print(f"特征数: {len(predictor.get_feature_importance())}")
    print(f"Top-5 特征:")
    for name, val in predictor.get_top_features(5):
        print(f"  {name}: {val:.4f}")
