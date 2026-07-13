"""
统一评估模块 (Evaluate)
多目标回归指标：RMSE / MAE / R² / 解释方差
特征重要性可视化
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional
import json

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import config
from utils import plot_feature_importance


def evaluate_model(model, X_test, y_test, feature_names: List[str],
                   output_dir: Path, granularity: str,
                   target_columns: List[str]) -> dict:
    """
    多目标回归评估

    Args:
        model: 训练好的 RandomForestRegressor
        X_test, y_test: 测试集 (y_test shape = [n_samples, n_targets])
        feature_names: 特征列名
        output_dir: 输出目录
        granularity: "daily" 或 "hourly"
        target_columns: 目标列名列表

    Returns:
        评估结果字典
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_pred = model.predict(X_test)

    n_targets = y_test.shape[1] if len(y_test.shape) > 1 else 1

    per_target = {}
    all_rmse, all_mae, all_r2 = [], [], []

    for i in range(n_targets):
        col_name = target_columns[i] if i < len(target_columns) else f"target_{i}"
        y_t = y_test[:, i] if n_targets > 1 else y_test
        y_p = y_pred[:, i] if n_targets > 1 else y_pred

        rmse = np.sqrt(mean_squared_error(y_t, y_p))
        mae = mean_absolute_error(y_t, y_p)
        r2 = r2_score(y_t, y_p)

        per_target[col_name] = {
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
            "mean_actual": float(np.mean(y_t)),
            "std_actual": float(np.std(y_t)),
            "mean_pred": float(np.mean(y_p)),
            "std_pred": float(np.std(y_p)),
        }
        all_rmse.append(rmse)
        all_mae.append(mae)
        all_r2.append(r2)

    overall_rmse = float(np.mean(all_rmse))
    overall_mae = float(np.mean(all_mae))
    overall_r2 = float(np.mean(all_r2))

    oob_score = getattr(model, "oob_score_", None)

    print(f"\n{'='*60}")
    print(f"{granularity.upper()} 模型评估（多目标回归）")
    print(f"{'='*60}")
    print(f"Overall RMSE: {overall_rmse:.4f}")
    print(f"Overall MAE:  {overall_mae:.4f}")
    print(f"Overall R²:   {overall_r2:.4f}")
    if oob_score is not None:
        print(f"OOB Score:    {oob_score:.4f}")

    print(f"\n各目标详情:")
    print(f"{'目标':<25} {'RMSE':>10} {'MAE':>10} {'R²':>10}")
    print("-" * 60)
    for col_name, m in per_target.items():
        print(f"{col_name:<25} {m['rmse']:>10.4f} {m['mae']:>10.4f} {m['r2']:>10.4f}")

    plot_feature_importance(model.feature_importances_, feature_names,
                             output_dir / "feature_importance.png",
                             top_n=min(20, len(feature_names)),
                             title=f"{granularity.capitalize()} - Feature Importance (Top-{min(20, len(feature_names))})")

    results = {
        "granularity": granularity,
        "overall_rmse": overall_rmse,
        "overall_mae": overall_mae,
        "overall_r2": overall_r2,
        "oob_score": float(oob_score) if oob_score is not None else None,
        "best_params": getattr(model, "best_params_", None),
        "target_columns": target_columns,
        "per_target": per_target,
    }

    with open(output_dir / "evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    _write_md_report(results, output_dir / "evaluation_report.md", granularity)

    return results


def _write_md_report(results: dict, path: Path, granularity: str):
    """生成 Markdown 评估报告"""
    lines = [
        f"# 随机森林 {granularity.capitalize()} 评估报告（多目标回归）",
        "",
        "## 整体指标",
        "",
        "| 指标 | 值 |",
        "|------|-----|",
        f"| Overall RMSE | {results['overall_rmse']:.4f} |",
        f"| Overall MAE | {results['overall_mae']:.4f} |",
        f"| Overall R² | {results['overall_r2']:.4f} |",
    ]
    if results.get("oob_score") is not None:
        lines.append(f"| OOB Score | {results['oob_score']:.4f} |")

    lines += ["", "## 各目标详情", "",
              "| 目标 | RMSE | MAE | R² | 真实均值 | 真实标准差 |",
              "|------|------|-----|----|---------|-----------|"]
    for col_name, m in results["per_target"].items():
        lines.append(
            f"| {col_name} | {m['rmse']:.4f} | {m['mae']:.4f} | {m['r2']:.4f} | {m['mean_actual']:.4f} | {m['std_actual']:.4f} |")

    lines += ["", "## 结论",
              f"- 模型在测试集上 Overall R²={results['overall_r2']:.4f}",
              f"- 建议集成层使用权重 (R²): {results['overall_r2']:.4f}"]

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  评估报告已保存: {path}")
