"""
小时级多目标回归训练脚本 (Train Hourly)
策略：10% 子样本粗搜参数 -> 全量训练 + OOB -> 评估
预测目标：下一小时的 5 个气象变量（温度/降水/风速/体感温度/相对湿度）

优化措施：
  1. 数据重切分：2023 年移入测试集（train=2015~2022, test=2023~2024）
  2. precipitation 目标 log1p 变换（长尾→近似正态）
  3. lag_48 + rolling_48 特征（捕捉两天周期）
  4. 精简 GridSearch 网格（24 组 vs 原 81 组）
"""
import sys
import time
import pickle
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

import config
from data_loader import load_hourly, save_feature_config
from evaluate import evaluate_model


def _print_grid_results(grid):
    """输出 GridSearchCV 每组参数的 CV 分数 + 耗时"""
    results = grid.cv_results_
    n_combos = len(results["params"])
    print(f"\n{'='*90}")
    print(f"GridSearch 参数效率表（{n_combos} 组参数 × {config.CV_SPLITS} CV = {n_combos * config.CV_SPLITS} 次拟合）")
    print(f"{'='*90}")
    header = f"{'#':>3} | {'n_est':>5} {'depth':>6} {'leaf':>4} {'feat':>6} | {'CV MSE(neg)':>12} {'CV std':>8} | {'fit_time':>9}"
    print(header)
    print("-" * 90)
    for i in range(n_combos):
        p = results["params"][i]
        mean = results["mean_test_score"][i]
        std = results["std_test_score"][i]
        ft = results["mean_fit_time"][i]
        depth = str(p.get("max_depth", "-"))
        feat = str(p.get("max_features", "-"))
        print(f"{i+1:>3} | {p['n_estimators']:>5} {depth:>6} {p['min_samples_leaf']:>4} {feat:>6} | {mean:>12.4f} {std:>8.4f} | {ft:>8.1f}s")
    best_i = grid.best_index_
    print(f"{'='*90}")
    print(f"最优: #{best_i+1}  参数={grid.best_params_}")
    print(f"最优 CV {config.CV_SCORING}: {grid.best_score_:.4f}  fit_time={results['mean_fit_time'][best_i]:.1f}s")
    print(f"{'='*90}")


def train_hourly(use_grid_search: bool = True):
    """小时级训练主函数"""
    print("=" * 60)
    print("随机森林 - 小时级多目标回归训练")
    print(f"数据切分: train < {config.TRAIN_CUTOFF}, test >= {config.TEST_START}")
    print(f"log1p(precipitation): {config.USE_LOG_TRANSFORM_PRECIP}")
    print(f"lag 特征: {config.LAG_PERIODS_HOURLY}")
    print("=" * 60)

    # 1. 子采样数据粗搜参数
    print(f"\n[阶段1] 子采样 {config.HOURLY_SUBSAMPLE_FRAC:.0%} 粗搜参数...")
    X_sub, y_sub, _, _, feat_sub = load_hourly(
        subset_frac=config.HOURLY_SUBSAMPLE_FRAC)

    if use_grid_search:
        print(f"\n{'='*60}")
        print("GridSearchCV 超参数搜索（子样本）")
        print(f"{'='*60}")

        rf_base = RandomForestRegressor(**config.RF_FIXED_PARAMS)

        tscv = TimeSeriesSplit(n_splits=config.CV_SPLITS)
        grid = GridSearchCV(
            rf_base, config.PARAM_GRID,
            cv=tscv, scoring=config.CV_SCORING,
            n_jobs=-1, verbose=1, refit=True,
        )
        grid.fit(X_sub, y_sub)

        _print_grid_results(grid)
        best_params = grid.best_params_
    else:
        best_params = {
            "n_estimators": 300, "max_depth": 30,
            "min_samples_leaf": 4,
            "max_features": "log2",
        }

    # 2. 全量训练集加载
    print(f"\n[阶段2] 加载全量训练集...")
    X_train, y_train, X_test, y_test, feature_names = load_hourly(subset_frac=1.0)

    print(f"\n训练集: {X_train.shape[0]:,} 行 × {X_train.shape[1]} 特征")
    print(f"测试集: {X_test.shape[0]:,} 行")
    print(f"目标数: {y_train.shape[1]}")

    # 3. 全量训练（用子样本搜索到的最优参数）
    print(f"\n{'='*60}")
    print("全量训练（OOB 评估）")
    print(f"{'='*60}")

    final_params = {
        "n_estimators": best_params["n_estimators"],
        "max_depth": best_params["max_depth"],
        "min_samples_leaf": best_params["min_samples_leaf"],
        "max_features": best_params["max_features"],
        "random_state": config.RANDOM_SEED,
        "n_jobs": -1,
        "oob_score": True,
        "bootstrap": True,
    }
    print(f"最终参数: {final_params}")

    t0 = time.time()
    rf_final = RandomForestRegressor(**final_params)
    rf_final.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"全量训练耗时: {train_time:.1f}s ({train_time/60:.1f} min)")

    oob_score = rf_final.oob_score_
    print(f"OOB Score (全量): {oob_score:.4f}")

    # 4. 保存模型
    print(f"\n{'='*60}")
    print("保存模型")
    print(f"{'='*60}")

    saved = {
        "model": rf_final,
        "feature_names": feature_names,
        "target_columns": config.HOURLY_TARGET_COLUMNS,
        "best_params": best_params,
        "oob_score": oob_score,
        "granularity": "hourly",
        "subsample_frac": config.HOURLY_SUBSAMPLE_FRAC,
        "train_time_sec": train_time,
        "use_log_precip": config.USE_LOG_TRANSFORM_PRECIP,
        "train_cutoff": config.TRAIN_CUTOFF,
    }
    with open(config.HOURLY_MODEL_PATH, "wb") as f:
        pickle.dump(saved, f)
    print(f"模型已保存: {config.HOURLY_MODEL_PATH}")

    save_feature_config(feature_names, "hourly")

    # 5. 评估
    results = evaluate_model(rf_final, X_test, y_test, feature_names,
                             config.HOURLY_OUTPUT_DIR, "hourly",
                             target_columns=config.HOURLY_TARGET_COLUMNS)
    results["train_time_sec"] = train_time

    return rf_final, results


if __name__ == "__main__":
    rf, res = train_hourly(use_grid_search=True)
    print(f"\n{'='*60}")
    print(f"小时级训练完成!")
    print(f"{'='*60}")
    print(f"OOB Score: {res['oob_score']:.4f}")
    print(f"Overall RMSE: {res['overall_rmse']:.4f}")
    print(f"Overall R²: {res['overall_r2']:.4f}")
    print(f"训练耗时: {res['train_time_sec']:.1f}s ({res['train_time_sec']/60:.1f} min)")
