"""
日级天气分类训练脚本 (Train Daily)
流程：load -> GridSearchCV(TimeSeriesSplit, f1_macro) -> fit -> OOB -> save -> evaluate
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
from data_loader import load_daily, save_feature_config
from evaluate import evaluate_model


def train_daily(use_grid_search: bool = True):
    """日级训练主函数"""
    print("=" * 60)
    print("随机森林 - 日级天气分类训练")
    print("=" * 60)

    # 1. 加载数据
    X_train, y_train, X_test, y_test, feature_names = load_daily()

    # 2. 标签编码（字符串 -> 整数）
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    print(f"\n标签编码: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # 3. 超参数搜索
    if use_grid_search:
        print(f"\n{'='*60}")
        print("GridSearchCV 超参数搜索")
        print(f"{'='*60}")

        rf_base = RandomForestClassifier(**config.RF_FIXED_PARAMS)

        tscv = TimeSeriesSplit(n_splits=config.CV_SPLITS)
        grid = GridSearchCV(
            rf_base, config.PARAM_GRID,
            cv=tscv, scoring=config.CV_SCORING,
            n_jobs=-1, verbose=1, refit=True,
        )
        grid.fit(X_train, y_train_enc)

        best_params = grid.best_params_
        print(f"\n最优参数: {best_params}")
        print(f"最优 CV {config.CV_SCORING}: {grid.best_score_:.4f}")
    else:
        best_params = {
            "n_estimators": 400, "max_depth": None,
            "min_samples_leaf": 1, "min_samples_split": 5,
            "max_features": "log2",
        }

    # 4. 最终模型训练（用最优参数 + OOB）
    print(f"\n{'='*60}")
    print("最终模型训练（OOB 评估）")
    print(f"{'='*60}")

    final_params = {**config.RF_FIXED_PARAMS, **best_params}
    rf_final = RandomForestClassifier(**final_params)
    rf_final.fit(X_train, y_train_enc)

    oob_score = rf_final.oob_score_
    print(f"OOB Score: {oob_score:.4f}")

    # 5. 保存模型
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
        "granularity": "daily",
    }
    with open(config.DAILY_MODEL_PATH, "wb") as f:
        pickle.dump(saved, f)
    print(f"模型已保存: {config.DAILY_MODEL_PATH}")

    save_feature_config(feature_names, "daily")

    # 6. 评估
    results = evaluate_model(rf_final, X_test, y_test_enc, feature_names,
                             config.DAILY_OUTPUT_DIR, "daily",
                             label_encoder=le)

    return rf_final, results


if __name__ == "__main__":
    rf, res = train_daily(use_grid_search=True)
    print(f"\n{'='*60}")
    print("日级训练完成!")
    print(f"{'='*60}")
    print(f"OOB Score: {res['oob_score']:.4f}")
    print(f"Macro-F1: {res['macro_f1']:.4f}")
