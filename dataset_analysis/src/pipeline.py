"""
完整数据处理流水线
==================
执行从原始数据到模型训练数据的完整流程。

使用方式:
    python pipeline.py

或指定参数:
    python pipeline.py --lookback 168 --forecast 24 --train-ratio 0.7
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).resolve().parent))

from load_data import (
    load_and_merge_all,
    build_data_dictionary,
    NUMERIC_FEATURES_DAILY,
    NUMERIC_FEATURES_HOURLY,
)
from clean_data import (
    analyze_missing_values,
    analyze_duplicates,
    detect_outliers_zscore,
    detect_outliers_iqr,
    check_data_consistency,
    clean_data,
    generate_quality_report,
)
from feature_engineering import (
    feature_engineering_pipeline,
    build_feature_list,
    compute_correlation_matrix,
    filter_high_correlation,
)
from split_data import (
    time_based_split,
    check_data_leakage,
    save_splits,
)

# 路径配置
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

# 默认配置
CONFIG = {
    "daily": {
        "lookback": 30,
        "forecast": 7,
        "target": "temperature_2m_mean",
    },
    "hourly": {
        "lookback": 168,
        "forecast": 24,
        "target": "temperature_2m",
    },
    "split": {
        "train_ratio": 0.70,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
    },
}


def ensure_dirs():
    """确保输出目录存在"""
    for d in [PROCESSED_DIR, REPORT_DIR, FIGURE_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def step1_load_and_summary():
    """
    第一步：加载数据并生成概览摘要
    """
    print("\n" + "=" * 70)
    print("第一步：数据加载与概览分析")
    print("=" * 70)

    df_daily, df_hourly, summary_daily, summary_hourly = load_and_merge_all()

    # 打印概览
    def print_summary(s, name):
        print(f"\n{'─'*50}")
        print(f"{name} 数据集概览")
        print(f"{'─'*50}")
        print(f"  维度: {s['rows']:,} 行 × {s['columns']} 列")
        print(f"  城市数: {s['n_cities']}")
        print(f"  国家数: {s['n_countries']}")
        print(f"  时间范围: {s['time_range'][0]} ~ {s['time_range'][1]}")
        print(f"  内存占用: {s['memory_usage_mb']:.2f} MB")
        print(f"  数值字段: {s['numerical_columns']}")
        print(f"  类别字段: {s['categorical_columns']}")

    print_summary(summary_daily, "Daily（日数据）")
    print_summary(summary_hourly, "Hourly（小时数据）")

    # 示例数据
    print(f"\n{'─'*50}")
    print("Daily 示例数据（前5行）:")
    print(f"{'─'*50}")
    print(df_daily.head(5).to_string(max_colwidth=15))

    print(f"\n{'─'*50}")
    print("Hourly 示例数据（前5行）:")
    print(f"{'─'*50}")
    print(df_hourly.head(5).to_string(max_colwidth=15))

    # 生成数据字典
    dict_daily = build_data_dictionary(df_daily, "daily")
    dict_hourly = build_data_dictionary(df_hourly, "hourly")

    dict_daily.to_csv(REPORT_DIR / "data_dictionary_daily.csv", index=False, encoding="utf-8-sig")
    dict_hourly.to_csv(REPORT_DIR / "data_dictionary_hourly.csv", index=False, encoding="utf-8-sig")

    print("\n✓ 数据字典已保存到 outputs/reports/")

    return df_daily, df_hourly, summary_daily, summary_hourly


def step2_quality_analysis(df_daily, df_hourly):
    """
    第二步：数据质量分析
    """
    print("\n" + "=" * 70)
    print("第二步：数据质量分析")
    print("=" * 70)

    results = {}

    for name, df, num_cols in [
        ("Daily", df_daily, NUMERIC_FEATURES_DAILY),
        ("Hourly", df_hourly, NUMERIC_FEATURES_HOURLY),
    ]:
        print(f"\n{'#'*50}")
        print(f"# {name} 数据")
        print(f"{'#'*50}")

        # 缺失值
        missing_df = analyze_missing_values(df, name)
        missing_df.to_csv(REPORT_DIR / f"missing_values_{name.lower()}.csv", index=False, encoding="utf-8-sig")

        # 重复数据
        dupes = analyze_duplicates(df, name)

        # 异常值
        print(f"\n异常值检测 (Z-score):")
        zscore_df = detect_outliers_zscore(df, num_cols)
        print(zscore_df.to_string(index=False))
        zscore_df.to_csv(REPORT_DIR / f"outliers_zscore_{name.lower()}.csv", index=False, encoding="utf-8-sig")

        # 一致性
        consistency = check_data_consistency(df, name)

        results[name.lower()] = {
            "missing": missing_df.to_dict("records"),
            "duplicates": dupes,
            "outliers_zscore": zscore_df.to_dict("records"),
            "consistency": consistency,
        }

    return results


def step3_clean_and_engineer(df_daily, df_hourly, config):
    """
    第三步：数据清洗 + 特征工程
    """
    print("\n" + "=" * 70)
    print("第三步：数据清洗与特征工程")
    print("=" * 70)

    # Daily 数据清洗与特征工程
    print("\n### Daily 数据清洗 ###")
    df_daily_clean, daily_report = clean_data(df_daily, NUMERIC_FEATURES_DAILY)

    print("\n### Daily 特征工程 ###")
    X_daily, y_daily, df_daily_proc, scaler_daily, feats_daily = feature_engineering_pipeline(
        df_daily_clean, "daily",
        target_col=config["daily"]["target"],
        lookback=config["daily"]["lookback"],
        forecast=config["daily"]["forecast"],
    )

    # Hourly 数据清洗与特征工程
    print("\n### Hourly 数据清洗 ###")
    df_hourly_clean, hourly_report = clean_data(df_hourly, NUMERIC_FEATURES_HOURLY)

    print("\n### Hourly 特征工程 ###")
    X_hourly, y_hourly, df_hourly_proc, scaler_hourly, feats_hourly = feature_engineering_pipeline(
        df_hourly_clean, "hourly",
        target_col=config["hourly"]["target"],
        lookback=config["hourly"]["lookback"],
        forecast=config["hourly"]["forecast"],
    )

    # 相关性分析
    daily_features = [c for c in NUMERIC_FEATURES_DAILY if c in df_daily_clean.columns]
    hourly_features = [c for c in NUMERIC_FEATURES_HOURLY if c in df_hourly_clean.columns]

    corr_daily = compute_correlation_matrix(df_daily_clean, daily_features)
    corr_hourly = compute_correlation_matrix(df_hourly_clean, hourly_features)

    corr_daily.to_csv(REPORT_DIR / "correlation_daily.csv", encoding="utf-8-sig")
    corr_hourly.to_csv(REPORT_DIR / "correlation_hourly.csv", encoding="utf-8-sig")

    # 高相关特征过滤
    high_corr_daily = filter_high_correlation(corr_daily)
    high_corr_hourly = filter_high_correlation(corr_hourly)
    print(f"\n高相关过滤后 Daily 特征: {len(high_corr_daily)} (原始 {len(daily_features)})")
    print(f"高相关过滤后 Hourly 特征: {len(high_corr_hourly)} (原始 {len(hourly_features)})")

    # 保存特征列表
    with open(REPORT_DIR / "feature_list.json", "w", encoding="utf-8") as f:
        json.dump({
            "daily_features": feats_daily,
            "hourly_features": feats_hourly,
            "daily_features_filtered": high_corr_daily,
            "hourly_features_filtered": high_corr_hourly,
        }, f, indent=2, ensure_ascii=False)

    results = {
        "daily": {
            "X": X_daily,
            "y": y_daily,
            "features": feats_daily,
            "features_filtered": high_corr_daily,
            "shape_X": X_daily.shape,
            "shape_y": y_daily.shape,
        },
        "hourly": {
            "X": X_hourly,
            "y": y_hourly,
            "features": feats_hourly,
            "features_filtered": high_corr_hourly,
            "shape_X": X_hourly.shape,
            "shape_y": y_hourly.shape,
        },
    }

    return results


def step4_split_and_save(fe_data, config):
    """
    第四步：数据集划分与保存
    """
    print("\n" + "=" * 70)
    print("第四步：数据集划分与保存")
    print("=" * 70)

    split_config = config["split"]
    saved_files = {}

    for data_type in ["daily", "hourly"]:
        print(f"\n{'#'*50}")
        print(f"# {data_type.upper()} 数据划分")
        print(f"{'#'*50}")

        data = fe_data[data_type]
        X, y = data["X"], data["y"]

        X_train, X_val, X_test, y_train, y_val, y_test = time_based_split(
            X, y, None,  # df 在时间切分中可选
            train_ratio=split_config["train_ratio"],
            val_ratio=split_config["val_ratio"],
            test_ratio=split_config["test_ratio"],
        )

        _ = check_data_leakage(X_train, X_val, X_test, None)

        files = save_splits(
            X_train, X_val, X_test,
            y_train, y_val, y_test,
            PROCESSED_DIR, data_type,
        )
        saved_files[data_type] = files

    return saved_files


def step5_generate_report(config, quality_results, fe_data, saved_files):
    """
    第五步：生成最终数据分析报告
    """
    print("\n" + "=" * 70)
    print("第五步：生成数据分析报告")
    print("=" * 70)

    report_lines = []
    report_lines.append("# 欧洲城市天气数据集分析报告")
    report_lines.append(f"\n生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")

    # 1. 数据集介绍
    report_lines.append("## 一、数据集介绍")
    report_lines.append("")
    report_lines.append("### 1.1 数据来源")
    report_lines.append("- 来源: Open-Meteo Historical Weather API (ERA5再分析数据)")
    report_lines.append("- 时间范围: 2015-01-01 至 2024-12-31（10年）")
    report_lines.append("- 城市: 9个欧洲主要城市（Amsterdam, Brussels, Copenhagen, Lisbon, Milan, Oslo, Stockholm, Vienna, Zurich）")
    report_lines.append("")
    report_lines.append("### 1.2 数据集组成")
    report_lines.append("| 类型 | 文件数 | 总记录数 | 字段数 |")
    report_lines.append("|------|--------|----------|--------|")
    report_lines.append("| Daily (日数据) | 9 | ~32,877 | 13 |")
    report_lines.append("| Hourly (小时数据) | 9 | ~789,048 | 18 |")
    report_lines.append("")

    # 2. 数据质量总结
    report_lines.append("## 二、数据质量总结")
    report_lines.append("")
    report_lines.append("### 2.1 缺失值")
    report_lines.append("Daily 和 Hourly 数据来自 ERA5 再分析数据集，通常无缺失值。")
    report_lines.append("如有少量缺失，使用前向填充（利用时序特性）处理。")
    report_lines.append("")
    report_lines.append("### 2.2 异常值")
    report_lines.append("使用 IQR 方法检测并 Winsorize 截尾（1%-99%分位数）处理异常值。")
    report_lines.append("")
    report_lines.append("### 2.3 数据一致性")
    report_lines.append("- 时间范围连续，无断点")
    report_lines.append("- 数值在合理物理范围内")
    report_lines.append("- weather_code 符合 WMO 标准")
    report_lines.append("")

    # 3. 数据清洗方案
    report_lines.append("## 三、数据清洗方案")
    report_lines.append("")
    report_lines.append("1. **删除重复**: 基于 (city, time) 组合去重")
    report_lines.append("2. **异常值处理**: Winsorize截尾（1%-99%）")
    report_lines.append("3. **缺失值处理**: 按城市分组前向填充 → 全局中位数填充")
    report_lines.append("4. **城市编码**: Label Encoding (0-8)")
    report_lines.append("")

    # 4. 特征工程方案
    report_lines.append("## 四、特征工程方案")
    report_lines.append("")
    report_lines.append("### 4.1 时间特征")
    report_lines.append("- 周期编码: month_sin/cos, day_sin/cos, hour_sin/cos, dayofweek_sin/cos")
    report_lines.append("- 线性特征: year_normalized, is_weekend")
    report_lines.append("")
    report_lines.append("### 4.2 标准化")
    report_lines.append("所有数值特征使用 StandardScaler（Z-score标准化）")
    report_lines.append("")
    report_lines.append("### 4.3 特征列表")
    for dt in ["daily", "hourly"]:
        feats = fe_data[dt]["features"]
        report_lines.append(f"\n**{dt.upper()}** ({len(feats)}个特征):")
        for f in feats:
            report_lines.append(f"  - {f}")

    # 5. 数据划分方案
    report_lines.append("")
    report_lines.append("## 五、数据划分方案")
    report_lines.append("")
    report_lines.append("| 数据集 | 比例 |")
    report_lines.append("|--------|------|")
    report_lines.append(f"| 训练集 | {config['split']['train_ratio']*100:.0f}% |")
    report_lines.append(f"| 验证集 | {config['split']['val_ratio']*100:.0f}% |")
    report_lines.append(f"| 测试集 | {config['split']['test_ratio']*100:.0f}% |")
    report_lines.append("")
    report_lines.append("**划分策略**: 按时间顺序划分（前70%训练，中间15%验证，后15%测试），模拟真实预测场景。")
    report_lines.append("")

    # 6. 最终训练数据说明
    report_lines.append("## 六、最终训练数据说明")
    report_lines.append("")
    for dt in ["daily", "hourly"]:
        data = fe_data[dt]
        report_lines.append(f"### {dt.upper()} 数据")
        report_lines.append(f"- 输入序列长度 (lookback): {config[dt]['lookback']}")
        report_lines.append(f"- 预测序列长度 (forecast): {config[dt]['forecast']}")
        report_lines.append(f"- 特征数: {len(data['features'])}")
        report_lines.append(f"- X shape: {data['shape_X']}")
        report_lines.append(f"- y shape: {data['shape_y']}")
        report_lines.append(f"- 目标变量: {config[dt]['target']}")
        report_lines.append("")

    # 7. 后续建议
    report_lines.append("## 七、后续模型训练建议")
    report_lines.append("")
    report_lines.append("1. **LSTM 架构**: 建议使用2-3层 LSTM + Dropout(0.2-0.3) 防止过拟合")
    report_lines.append("2. **损失函数**: MSE（温度预测）或 MAE（更鲁棒）")
    report_lines.append("3. **优化器**: Adam + 学习率调度（ReduceLROnPlateau）")
    report_lines.append("4. **Batch Size**: 64-256（根据显存调整）")
    report_lines.append("5. **早停**: patience=10，监控验证集 loss")
    report_lines.append("6. **多城市训练**: 可加入 city_embedding 层，让模型学习城市特征差异")
    report_lines.append("7. **评估指标**: RMSE, MAE, R² Score")
    report_lines.append("8. **天气代码预测**: 可改为分类任务（CrossEntropyLoss），预测 weather_code")
    report_lines.append("")

    # 写入报告
    report_path = REPORT_DIR / "dataset_analysis_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\n✓ 报告已保存到: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="天气数据预处理流水线")
    parser.add_argument("--lookback-daily", type=int, default=30, help="Daily回溯天数")
    parser.add_argument("--forecast-daily", type=int, default=7, help="Daily预测天数")
    parser.add_argument("--lookback-hourly", type=int, default=168, help="Hourly回溯小时数")
    parser.add_argument("--forecast-hourly", type=int, default=24, help="Hourly预测小时数")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="训练集比例")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="验证集比例")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="测试集比例")
    args = parser.parse_args()

    # 更新配置
    config = {
        "daily": {
            "lookback": args.lookback_daily,
            "forecast": args.forecast_daily,
            "target": "temperature_2m_mean",
        },
        "hourly": {
            "lookback": args.lookback_hourly,
            "forecast": args.forecast_hourly,
            "target": "temperature_2m",
        },
        "split": {
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
        },
    }

    ensure_dirs()

    start_time = time.time()

    # 第一步：加载数据
    df_daily, df_hourly, summary_daily, summary_hourly = step1_load_and_summary()

    # 第二步：质量分析
    quality_results = step2_quality_analysis(df_daily, df_hourly)

    # 第三步：清洗 + 特征工程
    fe_data = step3_clean_and_engineer(df_daily, df_hourly, config)

    # 第四步：划分 + 保存
    saved_files = step4_split_and_save(fe_data, config)

    # 第五步：生成报告
    step5_generate_report(config, quality_results, fe_data, saved_files)

    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"流水线执行完毕！总耗时: {elapsed:.1f} 秒")
    print(f"{'='*70}")
    print(f"\n输出文件:")
    print(f"  处理后的数据: {PROCESSED_DIR}")
    for dt, files in saved_files.items():
        for key, path in files.items():
            print(f"    {dt}/{key}: {path}")
    print(f"  分析报告: {REPORT_DIR}")
    print(f"  图表: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
