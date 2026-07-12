"""
数据质量分析与清洗模块
=======================
功能：
  1. 缺失值检测与处理
  2. 重复数据检测与删除
  3. 异常值检测（Z-score / IQR）
  4. 数据一致性检查
  5. 生成数据质量报告
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
from scipy import stats
import warnings

warnings.filterwarnings("ignore")


def analyze_missing_values(df: pd.DataFrame, name: str = "") -> pd.DataFrame:
    """
    缺失值分析

    返回 DataFrame: 字段 | 缺失数量 | 缺失比例 | 处理策略
    """
    total = len(df)
    results = []

    for col in df.columns:
        missing = df[col].isna().sum()
        ratio = missing / total * 100

        # 自动推荐处理策略
        if ratio == 0:
            strategy = "无需处理"
        elif ratio < 5:
            strategy = "中位数填充（数值）/ 众数填充（类别）"
        elif ratio < 20:
            strategy = "KNN插补 或 前向填充（时序数据）"
        elif ratio < 50:
            strategy = "评估后决定：若为关键特征则插补，否则考虑删除该字段"
        else:
            strategy = "建议删除该字段（缺失过多）"

        results.append({
            "字段": col,
            "缺失数量": missing,
            "缺失比例(%)": round(ratio, 2),
            "处理策略": strategy,
        })

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values("缺失比例(%)", ascending=False)

    print(f"\n{'='*60}")
    print(f"缺失值分析 - {name}")
    print(f"{'='*60}")
    print(f"总记录数: {total:,}")
    print(f"存在缺失的字段数: {(result_df['缺失数量'] > 0).sum()}")
    print(result_df.to_string(index=False))

    return result_df


def analyze_duplicates(df: pd.DataFrame, name: str = "") -> Dict:
    """
    重复数据分析

    检查：
    - 完全重复行
    - 重复ID（city + time 组合）
    - 时间重复记录
    """
    print(f"\n{'='*60}")
    print(f"重复数据分析 - {name}")
    print(f"{'='*60}")

    # 1. 完全重复行
    full_dupes = df.duplicated().sum()
    print(f"\n1. 完全重复行: {full_dupes} ({full_dupes/len(df)*100:.2f}%)")

    # 2. city + time 组合重复
    if "city" in df.columns and "time" in df.columns:
        combo_dupes = df.duplicated(subset=["city", "time"]).sum()
        print(f"2. (city, time) 重复: {combo_dupes} ({combo_dupes/len(df)*100:.2f}%)")

    # 3. 时间重复（同一时间多条记录）
    if "city" in df.columns and "time" in df.columns:
        time_counts = df.groupby("time").size()
        time_dupes = (time_counts > 1).sum()
        print(f"3. 同一时间点多城市记录: {time_dupes} 个时间点")

    # 4. 检查时间序列完整性（每个城市的预期记录数）
    if "city" in df.columns and "time" in df.columns:
        print(f"\n4. 各城市记录数检查:")
        city_counts = df.groupby("city").size()
        for city, count in city_counts.items():
            print(f"   {city}: {count:,} 条")

    return {
        "full_duplicates": full_dupes,
        "combo_duplicates": combo_dupes if "city" in df.columns else 0,
    }


def detect_outliers_zscore(
    df: pd.DataFrame,
    columns: List[str],
    threshold: float = 3.0,
) -> pd.DataFrame:
    """
    使用 Z-score 方法检测异常值

    参数:
        threshold: Z-score 阈值，默认 3（即 3σ 之外视为异常）
    """
    total = len(df)
    results = []

    for col in columns:
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        series = df[col].dropna()
        z_scores = np.abs(stats.zscore(series))
        outliers = (z_scores > threshold).sum()
        ratio = outliers / total * 100

        results.append({
            "字段": col,
            "方法": f"Z-score (threshold={threshold})",
            "异常数量": outliers,
            "异常比例(%)": round(ratio, 2),
            "处理方案": "Winsorize截尾" if ratio < 10 else "评估后决定",
        })

    return pd.DataFrame(results)


def detect_outliers_iqr(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    使用 IQR（四分位距）方法检测异常值
    """
    total = len(df)
    results = []

    for col in columns:
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        series = df[col].dropna()
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = ((series < lower) | (series > upper)).sum()
        ratio = outliers / total * 100

        results.append({
            "字段": col,
            "方法": "IQR (1.5×IQR)",
            "异常数量": outliers,
            "异常比例(%)": round(ratio, 2),
            "Q1": round(Q1, 2),
            "Q3": round(Q3, 2),
            "IQR": round(IQR, 2),
            "下界": round(lower, 2),
            "上界": round(upper, 2),
        })

    return pd.DataFrame(results)


def check_data_consistency(df: pd.DataFrame, name: str = "") -> Dict:
    """
    数据一致性检查

    检查：
    - 数值范围合理性
    - 时间格式正确性
    - 字段编码一致性
    - 逻辑一致性（如 max >= min）
    """
    print(f"\n{'='*60}")
    print(f"数据一致性检查 - {name}")
    print(f"{'='*60}")

    issues = []

    # 1. 数值范围检查
    range_checks = {
        "temperature_2m": (-50, 55),
        "temperature_2m_max": (-50, 55),
        "temperature_2m_min": (-50, 55),
        "temperature_2m_mean": (-50, 55),
        "apparent_temperature": (-60, 60),
        "relative_humidity_2m": (0, 100),
        "pressure_msl": (870, 1085),
        "precipitation": (0, 500),
        "precipitation_sum": (0, 500),
        "rain": (0, 500),
        "rain_sum": (0, 500),
        "snowfall": (0, 200),
        "cloud_cover": (0, 100),
        "wind_speed_10m": (0, 200),
        "wind_speed_10m_max": (0, 200),
        "wind_gusts_10m": (0, 300),
        "wind_direction_10m": (0, 360),
        "shortwave_radiation": (0, 1500),
        "shortwave_radiation_sum": (0, 40),
    }

    for col, (lo, hi) in range_checks.items():
        if col not in df.columns:
            continue
        out_of_range = ((df[col] < lo) | (df[col] > hi)).sum()
        if out_of_range > 0:
            issue = f"  ⚠ {col}: {out_of_range} 条记录超出合理范围 [{lo}, {hi}]"
            print(issue)
            issues.append(issue)

    # 2. 时间连续性检查
    if "time" in df.columns:
        try:
            if df["time"].dtype == "object":
                df["time"] = pd.to_datetime(df["time"])
            time_min = df["time"].min()
            time_max = df["time"].max()
            print(f"  时间范围: {time_min} ~ {time_max}")
        except Exception as e:
            issue = f"  ⚠ 时间列格式异常: {e}"
            print(issue)
            issues.append(issue)

    # 3. 逻辑一致性：max >= min
    if "temperature_2m_max" in df.columns and "temperature_2m_min" in df.columns:
        bad = (df["temperature_2m_max"] < df["temperature_2m_min"]).sum()
        if bad > 0:
            issue = f"  ⚠ {bad} 条记录 t_max < t_min（逻辑错误）"
            print(issue)
            issues.append(issue)
        else:
            print("  ✓ temperature_2m_max >= temperature_2m_min: 全部通过")

    # 4. 城市名一致性
    if "city" in df.columns:
        cities = sorted(df["city"].unique())
        print(f"  城市列表 ({len(cities)}): {', '.join(cities)}")

    # 5. weather_code 范围检查
    if "weather_code" in df.columns:
        valid_codes = {0,1,2,3,45,48,51,53,55,61,63,65,71,73,75,80,81,82,85,86,95,96,99}
        actual_codes = set(df["weather_code"].dropna().unique())
        unknown = actual_codes - valid_codes
        if unknown:
            issue = f"  ⚠ weather_code 存在未知编码: {unknown}"
            print(issue)
            issues.append(issue)
        else:
            print("  ✓ weather_code: 所有编码在WMO标准范围内")

    if not issues:
        print("\n  总结: 数据一致性良好，未发现明显问题")

    return {"issues": issues, "status": "pass" if not issues else "warning"}


def clean_data(
    df: pd.DataFrame,
    numeric_cols: List[str],
    drop_duplicates: bool = True,
    outlier_method: str = "iqr",
) -> Tuple[pd.DataFrame, Dict]:
    """
    数据清洗主函数

    执行：
    1. 删除重复行
    2. 异常值处理（Winsorize截尾）
    3. 缺失值处理（前向填充 + 中位数填充）

    返回:
        (cleaned_df, cleaning_report)
    """
    report = {"original_rows": len(df)}
    orig_cols = df.columns.tolist()

    # 1. 删除完全重复行
    if drop_duplicates:
        before = len(df)
        df = df.drop_duplicates()
        report["duplicates_removed"] = before - len(df)
        if report["duplicates_removed"] > 0:
            print(f"  删除重复行: {report['duplicates_removed']} 条")

    # 2. 删除 (city, time) 组合重复（保留第一条）
    if "city" in df.columns and "time" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=["city", "time"], keep="first")
        report["combo_duplicates_removed"] = before - len(df)

    # 3. 异常值处理 - Winsorize 截尾
    outlier_report = {}
    for col in numeric_cols:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)

        if outlier_method == "iqr":
            # IQR方法: Q1 - 1.5*IQR ~ Q3 + 1.5*IQR
            iqr_val = Q3 - Q1
            lower_bound = Q1 - 1.5 * iqr_val
            upper_bound = Q3 + 1.5 * iqr_val
        elif outlier_method == "percentile":
            # 百分位法: 1%~99%
            lower_bound = series.quantile(0.01)
            upper_bound = series.quantile(0.99)
        else:
            raise ValueError(f"不支持的异常值处理方法: {outlier_method}，可选 'iqr' 或 'percentile'")

        before_outliers = ((series < lower_bound) | (series > upper_bound)).sum()
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        outlier_report[col] = {
            "method": outlier_method,
            "lower_bound": round(lower_bound, 4),
            "upper_bound": round(upper_bound, 4),
            "clipped": before_outliers,
        }
    report["outlier_winsorize"] = outlier_report

    # 4. 缺失值处理
    missing_report = {}
    # 先按城市分组做前向填充（时序特性）
    if "city" in df.columns:
        for col in numeric_cols:
            if col not in df.columns:
                continue
            before = df[col].isna().sum()
            df[col] = df.groupby("city")[col].transform(
                lambda x: x.ffill().bfill()
            )
            after = df[col].isna().sum()
            missing_report[col] = {"before": before, "after_ffill": after}

        # 如果仍有缺失，用全局中位数填充
        for col in numeric_cols:
            if col not in df.columns:
                continue
            if df[col].isna().sum() > 0:
                df[col] = df[col].fillna(df[col].median())

    report["missing_values"] = missing_report
    report["final_rows"] = len(df)

    print(f"\n清洗完成: {report['original_rows']:,} → {report['final_rows']:,} 条")

    return df, report


def generate_quality_report(
    df_daily: pd.DataFrame,
    df_hourly: pd.DataFrame,
    numeric_daily: List[str],
    numeric_hourly: List[str],
) -> str:
    """
    生成完整的数据质量分析报告文本
    """
    lines = []
    lines.append("=" * 70)
    lines.append("天气数据集质量分析报告")
    lines.append("=" * 70)

    for name, df, num_cols in [
        ("Daily (日数据)", df_daily, numeric_daily),
        ("Hourly (小时数据)", df_hourly, numeric_hourly),
    ]:
        lines.append(f"\n{'─'*60}")
        lines.append(f"### {name}")
        lines.append(f"{'─'*60}")
        lines.append(f"维度: {df.shape[0]:,} 行 × {df.shape[1]} 列")
        lines.append(f"城市数: {df['city'].nunique()}")
        lines.append(f"时间范围: {df['time'].min()} ~ {df['time'].max()}")

        # 缺失值
        lines.append(f"\n#### 缺失值统计")
        for col in df.columns:
            missing = df[col].isna().sum()
            if missing > 0:
                lines.append(f"  {col}: {missing} ({missing/len(df)*100:.2f}%)")

        # 数值统计
        lines.append(f"\n#### 数值字段统计")
        for col in num_cols:
            if col in df.columns:
                s = df[col].dropna()
                lines.append(
                    f"  {col}: mean={s.mean():.2f}, std={s.std():.2f}, "
                    f"min={s.min():.2f}, max={s.max():.2f}"
                )

    return "\n".join(lines)


if __name__ == "__main__":
    # 自测
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from load_data import load_and_merge_all, NUMERIC_FEATURES_DAILY, NUMERIC_FEATURES_HOURLY

    df_daily, df_hourly, _, _ = load_and_merge_all()

    # 缺失值分析
    _ = analyze_missing_values(df_daily, "Daily")
    _ = analyze_missing_values(df_hourly, "Hourly")

    # 重复值分析
    analyze_duplicates(df_daily, "Daily")
    analyze_duplicates(df_hourly, "Hourly")

    # 异常值检测
    print("\n异常值检测 (Z-score):")
    print(detect_outliers_zscore(df_hourly, NUMERIC_FEATURES_HOURLY).to_string(index=False))
    print("\n异常值检测 (IQR):")
    print(detect_outliers_iqr(df_hourly, NUMERIC_FEATURES_HOURLY).to_string(index=False))

    # 一致性检查
    check_data_consistency(df_daily, "Daily")
    check_data_consistency(df_hourly, "Hourly")

    # 数据清洗
    df_hourly_clean, report = clean_data(df_hourly, NUMERIC_FEATURES_HOURLY)
