"""
小时级天气分类训练脚本 (Train Hourly)
策略：10% 子样本粗搜参数 -> 全量训练 + OOB -> 评估
（386万行全量 GridSearch 太慢，先子采样定位参数范围）
"""
import sys
import pickle
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder

import config
from data_loader import load_hourly, save_feature_config
from evaluate import evaluate_model


def train_hourly(use_grid_search: bool = True):
    """小时级训练主函数"""
    print("=" * 60)
    print("随机森林 - 小时级天气分类训练")
    print("=" * 60)

    # 1. 子采样数据粗搜参数（仅训练集子采样，测试集用全量评估）
    print(f"\n[阶段1] 子采样 {config.HOURLY_SUBSAMPLE_FRAC:.0%} 粗搜参数...")
    X_sub, y_sub, _, _, feat_sub = load_hourly(
        subset_frac=config.HOURLY_SUBSAMPLE_FRAC)

    le = LabelEncoder()
    y_sub_enc = le.fit_transform(y_sub)

    if use_grid_search:
        print(f"\n{'='*60}")
        print("GridSearchCV 超参数搜索（子样本）")
        print(f"{'='*60}")

        rf_base = RandomForestClassifier(**config.RF_FIXED_PARAMS)

        tscv = TimeSeriesSplit(n_splits=config.CV_SPLITS)
        grid = GridSearchCV(
            rf_base, config.PARAM_GRID,
            cv=tscv, scoring=config.CV_SCORING,
            n_jobs=-1, verbose=1, refit=True,
        )
        grid.fit(X_sub, y_sub_enc)

        best_params = grid.best_params_
        print(f"\n最优参数(子样本): {best_params}")
        print(f"最优 CV {config.CV_SCORING}: {grid.best_score_:.4f}")
    else:
        best_params = {
            "n_estimators": 200, "max_depth": 30,
            "min_samples_leaf": 8, "min_samples_split": 2,
            "max_features": "log2",
        }

    # 2. 全量训练集加载
    print(f"\n[阶段2] 加载全量训练集...")
    X_train, y_train, X_test, y_test, feature_names = load_hourly(subset_frac=1.0)

    # 重置标签编码器（全量可能有不同标签分布，但类别应一致）
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    print(f"标签编码: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # 3. 全量训练（用子样本搜索到的最优参数 + 手动少数类加权）
    print(f"\n{'='*60}")
    print("全量训练（OOB 评估 + 少数类加权）")
    print(f"{'='*60}")

    # 小时级少数类(Rain/Snow)仅1.7%，使用手动加权
    class_weight = {}
    for cls, idx in zip(le.classes_, le.transform(le.classes_)):
        if cls in ("Rain", "Snow"):
            class_weight[int(idx)] = 5
        else:
            class_weight[int(idx)] = 1

    final_params = {
        "n_estimators": best_params["n_estimators"],
        "max_depth": best_params["max_depth"],
        "min_samples_leaf": best_params["min_samples_leaf"],
        "max_features": best_params["max_features"],
        "random_state": config.RANDOM_SEED,
        "n_jobs": -1,
        "class_weight": class_weight,
        "oob_score": True,
        "bootstrap": True,
    }
    print(f"最终参数: {final_params}")

    rf_final = RandomForestClassifier(**final_params)
    rf_final.fit(X_train, y_train_enc)

    oob_score = rf_final.oob_score_
    print(f"OOB Score (全量): {oob_score:.4f}")

    # 4. 保存模型
    print(f"\n{'='*60}")
    print("保存模型")
    print(f"{'='*60}")

    saved = {
        "model": rf_final,
        "feature_names": feature_names,
        "label_encoder": le,
        "classes": config.WEATHER_CATEGORIES,
        "best_params": best_params,
        "oob_score": oob_score,
        "granularity": "hourly",
        "subsample_frac": config.HOURLY_SUBSAMPLE_FRAC,
    }
    with open(config.HOURLY_MODEL_PATH, "wb") as f:
        pickle.dump(saved, f)
    print(f"模型已保存: {config.HOURLY_MODEL_PATH}")

    save_feature_config(feature_names, "hourly")

    # 5. 评估
    results = evaluate_model(rf_final, X_test, y_test_enc, feature_names,
                             config.HOURLY_OUTPUT_DIR, "hourly",
                             label_encoder=le)

    return rf_final, results


if __name__ == "__main__":
    rf, res = train_hourly(use_grid_search=True)
    print(f"\n{'='*60}")
    print(f"小时级训练完成!")
    print(f"{'='*60}")
    print(f"OOB Score: {res['oob_score']:.4f}")
    print(f"Macro-F1: {res['macro_f1']:.4f}")
