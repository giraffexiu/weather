"""
探索性数据分析 (EDA) 脚本
==========================
生成完整的 EDA 可视化图表和统计分析。

功能：
  1. 数据分布分析（直方图 + KDE + 箱线图）
  2. 特征关系分析（相关性热力图）
  3. 标签分布分析
  4. 时间序列趋势分析
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 非交互后端，用于服务器环境
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from load_data import (
    load_and_merge_all,
    NUMERIC_FEATURES_DAILY,
    NUMERIC_FEATURES_HOURLY,
    WEATHER_CODE_MAP,
)

# 设置中文字体（macOS）
plt.rcParams["font.family"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_style("whitegrid")

FIGURE_DIR = Path(__file__).resolve().parent.parent / "outputs" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def plot_distributions(df, columns, title_prefix, filename, max_cols=4):
    """
    绘制数值特征的分布图（直方图 + KDE + 箱线图）
    """
    available = [c for c in columns if c in df.columns]
    n = len(available)
    if n == 0:
        return

    rows = (n + max_cols - 1) // max_cols
    fig, axes = plt.subplots(rows * 2, min(n, max_cols), figsize=(5*min(n,max_cols), 4*rows*2))

    if n == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for i, col in enumerate(available):
        row = i // max_cols
        c = i % max_cols

        if rows * 2 > 1 and min(n, max_cols) > 1:
            ax_hist = axes[row*2, c]
            ax_box = axes[row*2+1, c]
        elif min(n, max_cols) == 1:
            ax_hist = axes[0, 0]
            ax_box = axes[1, 0]
        else:
            ax_hist = axes[0, c] if rows == 1 else axes[row*2, c]
            ax_box = axes[1, c] if rows == 1 else axes[row*2+1, c]

        data = df[col].dropna()

        # 直方图 + KDE
        ax_hist.hist(data, bins=50, density=True, alpha=0.6, color="steelblue", edgecolor="white")
        try:
            from scipy import stats
            kde = stats.gaussian_kde(data)
            x_range = np.linspace(data.min(), data.max(), 200)
            ax_hist.plot(x_range, kde(x_range), "r-", linewidth=2, label="KDE")
        except Exception:
            pass
        ax_hist.set_title(f"{col} 分布", fontsize=10)
        ax_hist.set_ylabel("密度")

        # 箱线图
        ax_box.boxplot(data, vert=False, patch_artist=True,
                       boxprops=dict(facecolor="lightblue"))
        ax_box.set_title(f"{col} 箱线图", fontsize=10)

    # 隐藏多余的子图
    for i in range(n, rows * max_cols):
        row = i // max_cols
        c = i % max_cols
        if rows * 2 > 1 and min(n, max_cols) > 1:
            axes[row*2, c].set_visible(False)
            axes[row*2+1, c].set_visible(False)

    plt.suptitle(f"{title_prefix} - 特征分布分析", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ 分布图已保存: {filename}")


def plot_correlation_heatmap(df, columns, title, filename):
    """
    绘制相关性热力图
    """
    available = [c for c in columns if c in df.columns]
    if len(available) < 2:
        return

    corr = df[available].corr()

    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    plt.title(f"{title} - Pearson相关系数热力图", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ 热力图已保存: {filename}")


def plot_target_distribution(df, target_col, title, filename):
    """
    绘制标签分布
    """
    if target_col not in df.columns:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    data = df[target_col].dropna()

    # 直方图
    axes[0].hist(data, bins=50, color="steelblue", alpha=0.7, edgecolor="white")
    axes[0].axvline(data.mean(), color="red", linestyle="--", label=f"均值={data.mean():.2f}")
    axes[0].axvline(data.median(), color="green", linestyle="--", label=f"中位数={data.median():.2f}")
    axes[0].set_title(f"{target_col} 分布直方图")
    axes[0].legend()
    axes[0].set_ylabel("频数")

    # 按月箱线图
    if "month" not in df.columns:
        df_temp = df.copy()
        df_temp["month"] = pd.to_datetime(df_temp["time"]).dt.month
    else:
        df_temp = df

    month_data = [df_temp[df_temp["month"] == m][target_col].dropna().values for m in range(1, 13)]
    axes[1].boxplot(month_data, tick_labels=range(1, 13), patch_artist=True,
                    boxprops=dict(facecolor="lightblue"))
    axes[1].set_title(f"{target_col} 按月分布")
    axes[1].set_xlabel("月份")
    axes[1].set_ylabel(target_col)

    plt.suptitle(f"{title} - 标签分布分析", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ 标签分布图已保存: {filename}")


def plot_city_temperature_trends(df_hourly, filename):
    """
    绘制所有城市的温度趋势对比（采样可视化）
    """
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    axes = axes.flatten()

    cities = sorted(df_hourly["city"].unique())

    for i, city in enumerate(cities[:9]):
        city_data = df_hourly[df_hourly["city"] == city].copy()
        # 每日平均温度，减少数据量
        city_data["date"] = city_data["time"].dt.date
        daily_mean = city_data.groupby("date")["temperature_2m"].mean().reset_index()
        daily_mean["date"] = pd.to_datetime(daily_mean["date"])

        # 只画每年的数据点（降采样）
        sample = daily_mean.iloc[::30]

        axes[i].plot(sample["date"], sample["temperature_2m"],
                     linewidth=0.5, color="steelblue", alpha=0.8)
        axes[i].set_title(city, fontsize=11)
        axes[i].set_ylabel("温度 (°C)")
        axes[i].xaxis.set_major_locator(mdates.YearLocator(2))
        axes[i].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        axes[i].tick_params(axis="x", rotation=45)

    plt.suptitle("各城市温度趋势 (2015-2024)", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ 城市温度趋势图已保存: {filename}")


def plot_weather_code_distribution(df_hourly, filename):
    """
    绘制天气代码分布
    """
    if "weather_code" not in df_hourly.columns:
        return

    code_counts = df_hourly["weather_code"].value_counts().sort_index()

    # 映射为中文标签
    labels = [WEATHER_CODE_MAP.get(c, f"未知({c})") for c in code_counts.index]

    plt.figure(figsize=(14, 6))
    colors = plt.cm.Set3(np.linspace(0, 1, len(code_counts)))
    bars = plt.bar(range(len(code_counts)), code_counts.values, color=colors)

    plt.xticks(range(len(code_counts)), labels, rotation=45, ha="right", fontsize=9)
    plt.ylabel("记录数")
    plt.title("天气代码 (weather_code) 分布", fontsize=14)

    # 添加数值标签
    for bar, v in zip(bars, code_counts.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f"{v:,}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ 天气代码分布图已保存: {filename}")


def plot_seasonal_patterns(df, target_col, data_type, filename):
    """
    绘制季节性模式（按月份和小时的平均值）
    """
    df = df.copy()
    df["month"] = pd.to_datetime(df["time"]).dt.month

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 按月平均
    monthly = df.groupby("month")[target_col].agg(["mean", "std"]).reset_index()
    axes[0].fill_between(monthly["month"],
                         monthly["mean"] - monthly["std"],
                         monthly["mean"] + monthly["std"],
                         alpha=0.3, color="steelblue")
    axes[0].plot(monthly["month"], monthly["mean"], "o-", color="steelblue", linewidth=2)
    axes[0].set_title(f"{target_col} 月平均±标准差")
    axes[0].set_xlabel("月份")
    axes[0].set_ylabel(target_col)
    axes[0].set_xticks(range(1, 13))

    # 按小时平均（仅hourly数据）
    if data_type == "hourly" and "hour" in df.columns:
        hourly_avg = df.groupby("hour")[target_col].agg(["mean", "std"]).reset_index()
        axes[1].fill_between(hourly_avg["hour"],
                            hourly_avg["mean"] - hourly_avg["std"],
                            hourly_avg["mean"] + hourly_avg["std"],
                            alpha=0.3, color="coral")
        axes[1].plot(hourly_avg["hour"], hourly_avg["mean"], "o-", color="coral", linewidth=2)
        axes[1].set_title(f"{target_col} 小时平均±标准差")
        axes[1].set_xlabel("小时")
        axes[1].set_ylabel(target_col)
        axes[1].set_xticks(range(0, 24, 3))
    else:
        axes[1].text(0.5, 0.5, "（仅hourly数据显示）", ha="center", va="center",
                    transform=axes[1].transAxes, fontsize=12)
        axes[1].set_title("hourly 模式")

    plt.suptitle(f"{data_type.upper()} - 季节性模式分析", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ 季节性模式图已保存: {filename}")


def main():
    print("=" * 60)
    print("探索性数据分析 (EDA)")
    print("=" * 60)

    # 加载数据
    df_daily, df_hourly, _, _ = load_and_merge_all()

    # ========== Daily EDA ==========
    print("\n### Daily 数据 EDA ###")

    print("1. 特征分布分析...")
    plot_distributions(
        df_daily, NUMERIC_FEATURES_DAILY,
        "Daily", "01_daily_distributions.png",
    )

    print("2. 相关性热力图...")
    plot_correlation_heatmap(
        df_daily, NUMERIC_FEATURES_DAILY,
        "Daily", "02_daily_correlation_heatmap.png",
    )

    print("3. 标签分布分析...")
    plot_target_distribution(
        df_daily, "temperature_2m_mean",
        "Daily", "03_daily_target_distribution.png",
    )

    print("4. 季节性模式...")
    plot_seasonal_patterns(
        df_daily, "temperature_2m_mean", "daily",
        "04_daily_seasonal_patterns.png",
    )

    # ========== Hourly EDA ==========
    print("\n### Hourly 数据 EDA ###")

    print("5. 特征分布分析...")
    plot_distributions(
        df_hourly, NUMERIC_FEATURES_HOURLY,
        "Hourly", "05_hourly_distributions.png",
    )

    print("6. 相关性热力图...")
    plot_correlation_heatmap(
        df_hourly, NUMERIC_FEATURES_HOURLY,
        "Hourly", "06_hourly_correlation_heatmap.png",
    )

    print("7. 标签分布分析...")
    plot_target_distribution(
        df_hourly, "temperature_2m",
        "Hourly", "07_hourly_target_distribution.png",
    )

    print("8. 城市温度趋势...")
    plot_city_temperature_trends(
        df_hourly, "08_city_temperature_trends.png",
    )

    print("9. 天气代码分布...")
    plot_weather_code_distribution(
        df_hourly, "09_weather_code_distribution.png",
    )

    print("10. 季节性模式...")
    plot_seasonal_patterns(
        df_hourly, "temperature_2m", "hourly",
        "10_hourly_seasonal_patterns.png",
    )

    print(f"\n✓ EDA完成！所有图表已保存到: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
