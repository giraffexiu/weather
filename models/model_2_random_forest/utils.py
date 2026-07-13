"""
可视化与辅助函数模块 (Utils)
特征重要性、OOB 误差曲线（多目标回归版）
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Optional

import config


def plot_feature_importance(importances: np.ndarray, feature_names: List[str],
                            output_path: Path, top_n: int = 20,
                            title: str = "Feature Importance (Top-20)"):
    """绘制特征重要性水平条形图"""
    indices = np.argsort(importances)[::-1][:top_n]
    top_importances = importances[indices]
    top_names = [feature_names[i] for i in indices]

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(range(len(top_importances)), top_importances,
                   color="steelblue", edgecolor="navy")
    ax.set_yticks(range(len(top_importances)))
    ax.set_yticklabels(top_names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    for bar, imp in zip(bars, top_importances):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{imp:.3f}", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  特征重要性图已保存: {output_path}")


def plot_oob_error_curve(X_train, y_train, output_path: Path,
                          fixed_params: dict, n_estimators_range: List[int],
                          title: str = "OOB Error Curve"):
    """
    绘制 OOB 误差曲线：X=n_estimators, Y=1-OOB R²
    用于判断模型何时收敛、最优树数量
    """
    from sklearn.ensemble import RandomForestRegressor

    oob_errors = []
    for n_est in n_estimators_range:
        params = {**fixed_params, "n_estimators": n_est, "oob_score": True,
                  "bootstrap": True}
        rf = RandomForestRegressor(**params)
        rf.fit(X_train, y_train)
        oob_err = 1 - rf.oob_score_
        oob_errors.append(oob_err)
        print(f"    n_estimators={n_est}: OOB error={oob_err:.4f}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(n_estimators_range, oob_errors, "o-", color="steelblue",
            linewidth=2, markersize=6)
    ax.set_xlabel("n_estimators", fontsize=12)
    ax.set_ylabel("OOB Error (1 - R²)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    best_n = n_estimators_range[int(np.argmin(oob_errors))]
    best_err = min(oob_errors)
    ax.axvline(best_n, color="red", linestyle="--", alpha=0.5)
    ax.annotate(f"Best: {best_n} trees\n(OOB err={best_err:.4f})",
                xy=(best_n, best_err), fontsize=9,
                xytext=(best_n + 30, best_err + 0.002),
                arrowprops=dict(arrowstyle="->", color="red"))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  OOB 误差曲线已保存: {output_path}")
    return best_n
