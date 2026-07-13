"""
统一评估模块 (Evaluate)
分类报告、混淆矩阵、OOB、特征重要性、概率校准
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional

from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, f1_score, cohen_kappa_score,
                              log_loss)
from sklearn.preprocessing import LabelEncoder

import config
from utils import (plot_confusion_matrix, plot_feature_importance,
                   plot_probability_distribution)


def evaluate_model(model, X_test, y_test, feature_names: List[str],
                   output_dir: Path, granularity: str,
                   label_encoder: Optional[LabelEncoder] = None) -> dict:
    """
    统一评估函数

    Args:
        model: 训练好的 RandomForestClassifier
        X_test, y_test: 测试集
        feature_names: 特征列名
        output_dir: 输出目录
        granularity: "daily" 或 "hourly"
        label_encoder: 标签编码器（如有）

    Returns:
        评估结果字典
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    classes = config.WEATHER_CATEGORIES

    # 预测
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # 解码标签为类别字符串（model.classes_ / label_encoder.classes_ 是字母序）
    model_classes = list(model.classes_)
    if label_encoder is not None:
        decode = label_encoder.inverse_transform
        y_test_labels = decode(y_test) if isinstance(np.asarray(y_test)[0],
                                                      (int, np.integer)) else np.asarray(y_test)
        y_pred_labels = decode(y_pred) if isinstance(np.asarray(y_pred)[0],
                                                     (int, np.integer)) else np.asarray(y_pred)
    else:
        y_test_labels = np.asarray(y_test)
        y_pred_labels = np.asarray(y_pred)

    # y_proba 的列顺序 = model.classes_（字母序）
    proba_labels = [label_encoder.inverse_transform([c])[0]
                    if label_encoder is not None else c
                    for c in model_classes]

    # 指标
    accuracy = accuracy_score(y_test_labels, y_pred_labels)
    macro_f1 = f1_score(y_test_labels, y_pred_labels, average="macro")
    weighted_f1 = f1_score(y_test_labels, y_pred_labels, average="weighted")
    kappa = cohen_kappa_score(y_test_labels, y_pred_labels)

    # OOB
    oob_score = getattr(model, "oob_score_", None)

    # 概率相关（labels 须与 y_proba 列顺序一致）
    try:
        proba_log_loss = log_loss(y_test_labels, y_proba, labels=proba_labels)
    except Exception:
        proba_log_loss = None

    report = classification_report(y_test_labels, y_pred_labels,
                                  labels=classes, output_dict=True,
                                  zero_division=0)

    # 可视化
    print(f"\n{'='*60}")
    print(f"{granularity.upper()} 模型评估")
    print(f"{'='*60}")
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Macro-F1:    {macro_f1:.4f}")
    print(f"Weighted-F1: {weighted_f1:.4f}")
    print(f"Cohen's Kappa: {kappa:.4f}")
    if oob_score is not None:
        print(f"OOB Score:   {oob_score:.4f}")
    if proba_log_loss is not None:
        print(f"Log Loss:    {proba_log_loss:.4f}")

    print(f"\n分类报告:")
    print(classification_report(y_test_labels, y_pred_labels,
                                labels=classes, zero_division=0))

    # 少数类 Recall
    for cls in ["Rain", "Snow"]:
        if cls in report:
            print(f"  {cls} Recall: {report[cls]['recall']:.4f}")

    # 图表
    plot_confusion_matrix(y_test_labels, y_pred_labels, classes,
                          output_dir / "confusion_matrix.png",
                          title=f"{granularity.capitalize()} - Confusion Matrix")

    plot_feature_importance(model.feature_importances_, feature_names,
                             output_dir / "feature_importance.png",
                             top_n=min(20, len(feature_names)),
                             title=f"{granularity.capitalize()} - Feature Importance (Top-{min(20, len(feature_names))})")

    plot_probability_distribution(y_test_labels, y_proba, classes,
                                   output_dir / "probability_distribution.png",
                                   title=f"{granularity.capitalize()} - Probability Distribution")

    # 保存评估报告
    results = {
        "granularity": granularity,
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "cohen_kappa": float(kappa),
        "oob_score": float(oob_score) if oob_score is not None else None,
        "log_loss": float(proba_log_loss) if proba_log_loss is not None else None,
        "best_params": getattr(model, "best_params_", None),
        "per_class": report,
    }

    import json
    with open(output_dir / "evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    # Markdown 报告
    _write_md_report(results, output_dir / "evaluation_report.md", granularity)

    return results


def _write_md_report(results: dict, path: Path, granularity: str):
    """生成 Markdown 评估报告"""
    lines = [
        f"# 随机森林 {granularity.capitalize()} 评估报告",
        "",
        "## 性能指标",
        "",
        "| 指标 | 值 |",
        "|------|-----|",
        f"| Accuracy | {results['accuracy']:.4f} |",
        f"| Macro-F1 | {results['macro_f1']:.4f} |",
        f"| Weighted-F1 | {results['weighted_f1']:.4f} |",
        f"| Cohen's Kappa | {results['cohen_kappa']:.4f} |",
    ]
    if results.get("oob_score") is not None:
        lines.append(f"| OOB Score | {results['oob_score']:.4f} |")
    if results.get("log_loss") is not None:
        lines.append(f"| Log Loss | {results['log_loss']:.4f} |")

    if results.get("best_params"):
        lines += ["", "## 最优超参数", "", "```", str(results["best_params"]), "```"]

    lines += ["", "## 各类别表现", "",
              "| 类别 | Precision | Recall | F1 | Support |",
              "|------|-----------|--------|----|---------|"]
    for cls in config.WEATHER_CATEGORIES:
        if cls in results["per_class"]:
            c = results["per_class"][cls]
            lines.append(f"| {cls} | {c['precision']:.4f} | {c['recall']:.4f} | {c['f1-score']:.4f} | {int(c['support'])} |")

    lines += ["", "## 结论",
              f"- 模型在测试集上 Macro-F1={results['macro_f1']:.4f}",
              f"- 建议集成层使用权重 (F1): {results['macro_f1']:.4f}"]

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  评估报告已保存: {path}")
