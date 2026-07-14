"""
小时级集成模型评估脚本
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    TEST_DATA_PATH,
    RESULTS_DIR,
    DEVICE,
    setup_paths,
    validate_config,
    HOUR_TARGET_COLUMNS,
    MODEL3_PERFORMANCE,
)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = float(np.mean(np.abs(y_pred - y_true)))
    mse = float(np.mean((y_pred - y_true) ** 2))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / ss_tot if ss_tot > 0 else 0)
    return {'mse': mse, 'mae': mae, 'r2': r2}


def evaluate():
    print("\n" + "=" * 70)
    print("Hourly Ensemble 模型评估")
    print("=" * 70)

    setup_paths()
    validate_config()
    print(f"设备: {DEVICE}")

    from dataset_loader import get_dataloader
    loader = get_dataloader(split='test', batch_size=512, shuffle=False, num_workers=0)

    from post_process import inverse_transform
    from model_wrapper import HourModelWrapper
    model = HourModelWrapper()
    results = model.predict(loader)  # 预测已是真实物理单位

    # 加载真实值（标准化空间 → 转回真实单位）
    targets = []
    for batch in loader:
        t = batch['target']
        if t.dim() == 1:
            t = t.unsqueeze(1)
        targets.append(t.numpy())
    targets = np.concatenate(targets, axis=0)  # (N, 5) z-score
    targets = inverse_transform(targets)      # 转回真实物理单位

    print(f"\n{'='*55}")
    print(f"  评估结果 (N={len(targets):,})")
    print(f"{'='*55}")
    print(f"{'目标':<28} {'MSE':>8} {'MAE':>8} {'R²':>8}")
    print(f"{'-'*55}")

    all_metrics = {}
    for i, col in enumerate(HOUR_TARGET_COLUMNS):
        y_t, y_p = targets[:, i], results['regression'][col]
        m = calculate_metrics(y_t, y_p)
        all_metrics[col] = m
        print(f"  {col:<26} {m['mse']:8.4f} {m['mae']:8.4f} {m['r2']:8.4f}")

    total_mse = float(np.mean((targets - np.column_stack(
        [results['regression'][c] for c in HOUR_TARGET_COLUMNS])) ** 2))
    total_mae = float(np.mean(np.abs(targets - np.column_stack(
        [results['regression'][c] for c in HOUR_TARGET_COLUMNS]))))
    print(f"{'-'*55}")
    print(f"  {'OVERALL':<26} {total_mse:8.4f} {total_mae:8.4f}")
    print(f"{'='*55}")

    # 保存
    output_file = RESULTS_DIR / "hourly_evaluation.txt"
    with open(output_file, 'w') as f:
        f.write(f"Hourly Ensemble Evaluation (N={len(targets):,})\n")
        f.write("=" * 55 + "\n")
        f.write(f"{'Target':<28} {'MSE':>8} {'MAE':>8} {'R²':>8}\n")
        for col in HOUR_TARGET_COLUMNS:
            m = all_metrics[col]
            f.write(f"  {col:<26} {m['mse']:8.4f} {m['mae']:8.4f} {m['r2']:8.4f}\n")
    print(f"\n结果已保存: {output_file}")

    return all_metrics


if __name__ == "__main__":
    try:
        evaluate()
    except KeyboardInterrupt:
        print("\n评估被中断")
    except Exception as e:
        print(f"\n评估出错: {e}")
        import traceback
        traceback.print_exc()
