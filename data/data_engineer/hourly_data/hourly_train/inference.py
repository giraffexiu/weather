"""
Wide & Deep 模型推理脚本
加载训练好的模型进行预测
"""
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

_HERE = Path(__file__).parent
_PARENT = _HERE.parent
sys.path.insert(0, str(_PARENT))

from dataset_loader import get_dataloader, config as dl_config
from hourly_train.model import WideAndDeep, create_model


CHECKPOINT_PATH = _HERE / "output" / "checkpoints" / "best_model.pt"


class Predictor:
    """Wide & Deep 推理器"""

    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        device: Optional[torch.device] = None,
    ):
        self.checkpoint_path = checkpoint_path or CHECKPOINT_PATH
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.model = self._load_model()

    def _load_model(self) -> WideAndDeep:
        model = create_model(device=self.device)
        if self.checkpoint_path.exists():
            ckpt = torch.load(self.checkpoint_path, map_location=self.device,
                              weights_only=True)
            model.load_state_dict(ckpt['model_state_dict'])
            val_loss = ckpt.get('val_loss', 'unknown')
            epoch = ckpt.get('epoch', 'unknown')
            print(f"加载模型: {self.checkpoint_path}")
            print(f"  Epoch: {epoch}, Val MSE: {val_loss:.6f}")
        else:
            print(f"警告: 未找到 checkpoint {self.checkpoint_path}")
        model.eval()
        return model

    @torch.no_grad()
    def predict_batch(self, batch: dict) -> np.ndarray:
        """对单个 batch 进行推理"""
        batch = {k: v.to(self.device) for k, v in batch.items()
                 if isinstance(v, torch.Tensor)}
        pred = self.model(batch)               # (B, 1)
        return pred.cpu().numpy()

    def predict_dataloader(self, dataloader) -> np.ndarray:
        """对整个 DataLoader 进行推理"""
        preds = []
        for batch in dataloader:
            p = self.predict_batch(batch)
            preds.append(p)
        return np.concatenate(preds, axis=0)

    def predict_from_csv(
        self, csv_path: Path, batch_size: int = 1024
    ) -> pd.DataFrame:
        """从 CSV 文件推理并返回带预测结果的 DataFrame"""
        loader = get_dataloader(
            split='test',
            data_path=csv_path,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        preds = self.predict_dataloader(loader)  # (N, num_targets)

        df = pd.read_csv(csv_path)
        for i, col in enumerate(dl_config.TARGET_COLUMNS):
            df[f'pred_{col}'] = preds[:, i]
        return df

    def evaluate(self, dataloader) -> dict:
        """
        评估模型，返回多目标指标
        """
        preds = self.predict_dataloader(dataloader)  # (N, num_targets)

        all_targets = []
        for batch in dataloader:
            t = batch['target']
            if t.dim() == 1:
                t = t.unsqueeze(1)
            all_targets.append(t.numpy())
        targets = np.concatenate(all_targets, axis=0)  # (N, num_targets)

        target_cols = dl_config.TARGET_COLUMNS
        per_column = {}
        for i, col in enumerate(target_cols):
            y_true = targets[:, i]
            y_pred = preds[:, i]
            mse = np.mean((y_pred - y_true) ** 2)
            mae = np.mean(np.abs(y_pred - y_true))
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            per_column[col] = {'MSE': float(mse), 'MAE': float(mae), 'R2': float(r2)}

        total_mse = np.mean((preds - targets) ** 2)
        total_mae = np.mean(np.abs(preds - targets))

        return {
            'total_MSE': float(total_mse),
            'total_MAE': float(total_mae),
            'per_column': per_column,
            'n_samples': len(targets),
        }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Wide & Deep 推理")
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT_PATH,
                        help='模型路径')
    parser.add_argument('--batch_size', type=int, default=1024, help='批次大小')
    parser.add_argument('--device', type=str, default='auto', help='设备')
    parser.add_argument('--output', type=Path, default=None,
                        help='输出预测 CSV 路径')
    args = parser.parse_args()

    device = torch.device(
        'cuda' if args.device == 'auto' and torch.cuda.is_available()
        else args.device
    )

    predictor = Predictor(checkpoint_path=args.checkpoint, device=device)

    # 测试集评估
    test_loader = get_dataloader(
        split='test',
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    metrics = predictor.evaluate(test_loader)
    print(f"\n{'='*50}")
    print("测试集评估结果")
    print(f"{'='*50}")
    print(f"  样本数: {metrics['n_samples']:,}")
    print(f"  总体 MSE: {metrics['total_MSE']:.6f}")
    print(f"  总体 MAE: {metrics['total_MAE']:.6f}")
    print(f"\n  各目标指标:")
    for col, m in metrics['per_column'].items():
        print(f"    {col}:")
        print(f"      MSE={m['MSE']:.6f}, MAE={m['MAE']:.6f}, R²={m['R2']:.6f}")
    print(f"{'='*50}\n")

    # 可选：保存预测结果
    if args.output:
        from dataset_loader import config as dl_config
        df = predictor.predict_from_csv(
            dl_config.TEST_DATA_PATH, batch_size=args.batch_size
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"预测结果已保存: {args.output}")


if __name__ == '__main__':
    main()
