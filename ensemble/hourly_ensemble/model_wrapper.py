"""
模型包装器：为小时级Wide & Deep模型提供统一预测+Cron作业安全接口
"""
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import torch

from config import (
    MODEL3_CHECKPOINT,
    MODEL3_DIR,
    MODEL3_TARGET_INDICES,
    HOUR_TARGET_COLUMNS,
    DEVICE,
    DATA_DIR,
)
from probability_converter import ProbabilityConverter
from post_process import post_process as _post_process


class HourModelWrapper:
    """
    小时Wide & Deep模型包装器
    提供简洁的预测接口 + Cron/Pipeline安全的懒加载
    """

    def __init__(self, probability_converter: Optional[ProbabilityConverter] = None):
        self.model = None
        self.device = DEVICE
        self.probability_converter = probability_converter

        # 路径配置
        if str(DATA_DIR) not in sys.path:
            sys.path.insert(0, str(DATA_DIR))
        if str(MODEL3_DIR.parent) not in sys.path:
            sys.path.insert(0, str(MODEL3_DIR.parent))
        if str(MODEL3_DIR) not in sys.path:
            sys.path.insert(0, str(MODEL3_DIR))

    def _ensure_model(self):
        """懒加载模型（Cron安全）"""
        if self.model is not None:
            return
        self._load_model()

    def _load_model(self):
        """加载Wide & Deep模型"""
        print("加载小时级 Wide & Deep 模型...")

        if not MODEL3_CHECKPOINT.exists():
            raise FileNotFoundError(f"Checkpoint not found: {MODEL3_CHECKPOINT}")

        # 从 hourly_train 导入模型工厂
        from model import WideAndDeep, create_model

        self.model = create_model(device=self.device)

        ckpt = torch.load(MODEL3_CHECKPOINT, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()

        print(f"  Epoch: {ckpt.get('epoch', '?')}, Val MSE: {ckpt.get('val_loss', 0):.6f}")
        print(f"  设备: {self.device}")

    @torch.no_grad()
    def predict(self, data_loader) -> Dict[str, np.ndarray]:
        """
        执行预测

        Args:
            data_loader: PyTorch DataLoader

        Returns:
            预测结果字典:
            {
                'regression': {
                    'temperature_2m': array (N,),
                    'precipitation': array (N,),
                    ...
                },
                'classification': {  # 需要 probability_converter
                    'rain': {'probability': array},
                    ...
                }
            }
        """
        self._ensure_model()
        self.model.eval()
        all_preds = []

        for batch in data_loader:
            batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            pred = self.model(batch)
            all_preds.append(pred.cpu().numpy())

        all_preds = np.concatenate(all_preds, axis=0)  # (N, 5)

        results = {'regression': {}}
        for col, idx in MODEL3_TARGET_INDICES.items():
            results['regression'][col] = all_preds[:, idx]

        # === 后处理：逆标准化 + 物理约束 ===
        real_preds = _post_process(all_preds)
        for i, col in enumerate(HOUR_TARGET_COLUMNS):
            results['regression'][col] = real_preds[:, i]

        # 若提供概率转换器则生成分类概率（用逆标准化后的值）
        if self.probability_converter is not None:
            results['classification'] = self.probability_converter.convert_all(all_preds)

        return results


def test_wrapper():
    """测试包装器"""
    print("=" * 70)
    print("测试 Hourly Model Wrapper")
    print("=" * 70)

    from config import PROBABILITY_CONVERSION_CONFIG
    converter = ProbabilityConverter(PROBABILITY_CONVERSION_CONFIG)
    wrapper = HourModelWrapper(probability_converter=converter)

    from dataset_loader import get_dataloader
    loader = get_dataloader(split='test', batch_size=128, shuffle=False, num_workers=0)

    results = wrapper.predict(loader)
    for col in HOUR_TARGET_COLUMNS:
        vals = results['regression'][col]
        print(f"  {col}: mean={vals.mean():.4f}, std={vals.std():.4f}, "
              f"range=[{vals.min():.4f}, {vals.max():.4f}]")

    print("\n  测试通过！")


if __name__ == "__main__":
    test_wrapper()
