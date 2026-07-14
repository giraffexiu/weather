"""
模型评估脚本
加载 checkpoint 对测试集评估并输出各目标指标
"""
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).parent
_PROJ_ROOT = _HERE.parent.parent.parent  # weather/
sys.path.insert(0, str(_PROJ_ROOT / "data" / "data_engineer" / "hourly_data"))
sys.path.insert(0, str(_HERE.parent))  # model_3_deep_learning/

from dataset_loader import get_dataloader, config as dl_config
from hourly_train.model import create_model
from hourly_train.config import CHECKPOINT_PATH


def evaluate(checkpoint_path: Path = None, batch_size: int = 1024, device: str = None):
    checkpoint_path = checkpoint_path or CHECKPOINT_PATH
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))

    # 加载模型
    model = create_model(device=device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"模型: {checkpoint_path}")
    print(f"  Epoch: {ckpt.get('epoch','?')}, Val MSE: {ckpt.get('val_loss',0):.6f}\n")

    # 测试集
    loader = get_dataloader(split='test', batch_size=batch_size, shuffle=False, num_workers=0)

    # 推理
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            pred = model(batch)
            target = batch['target']
            if target.dim() == 1:
                target = target.unsqueeze(1)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(target.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)       # (N, num_targets)
    targets = np.concatenate(all_targets, axis=0)    # (N, num_targets)

    target_cols = dl_config.TARGET_COLUMNS

    # 逐列指标
    print(f"{'='*55}")
    print(f"  测试集评估 (N={len(targets):,})")
    print(f"{'='*55}")
    print(f"{'目标':<28} {'MSE':>8} {'MAE':>8} {'R²':>8}")
    print(f"{'-'*55}")

    for i, col in enumerate(target_cols):
        y_t, y_p = targets[:, i], preds[:, i]
        mse = float(np.mean((y_p - y_t) ** 2))
        mae = float(np.mean(np.abs(y_p - y_t)))
        ss_res = np.sum((y_t - y_p) ** 2)
        ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
        r2 = float(1 - ss_res / ss_tot if ss_tot > 0 else 0)
        print(f"  {col:<26} {mse:8.4f} {mae:8.4f} {r2:8.4f}")

    # 总体
    total_mse = float(np.mean((preds - targets) ** 2))
    total_mae = float(np.mean(np.abs(preds - targets)))
    print(f"{'-'*55}")
    print(f"  {'OVERALL':<26} {total_mse:8.4f} {total_mae:8.4f}")
    print(f"{'='*55}")

    return {'total_MSE': total_mse, 'total_MAE': total_mae}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Wide & Deep 模型评估")
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT_PATH)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()
    evaluate(args.checkpoint, args.batch_size, args.device)
