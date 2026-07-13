"""
Hourly Weather Data 特征工程主程序
执行完整的特征工程流程（包含小时周期编码、weather_code 编码等）
"""
import sys
from pathlib import Path

# 添加当前目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent))

from feature_engineer import FeatureEngineeringPipeline
import config


def main():
    """主函数：执行完整的特征工程流程"""

    print("\n" + "=" * 60)
    print("Hourly Weather Data - 特征工程流程")
    print("=" * 60)

    # 检查输入文件是否存在
    if not config.INPUT_DATA_PATH.exists():
        print(f"\n错误: 输入文件不存在: {config.INPUT_DATA_PATH}")
        print("请确保数据清洗步骤已完成，并且文件路径正确")
        return

    # 初始化流水线
    pipeline = FeatureEngineeringPipeline(
        time_column=config.TIME_COLUMN,
        categorical_columns=config.CATEGORICAL_FEATURES,
        numerical_columns=config.NUMERICAL_FEATURES,
        use_cyclical_encoding=config.USE_CYCLICAL_ENCODING,
        create_derived_features=config.CREATE_DERIVED_FEATURES,
        scaling_method=config.SCALING_METHOD,
        train_end_date=config.TRAIN_END_DATE,
        test_start_date=config.TEST_START_DATE
    )

    try:
        # 执行完整的特征工程流程
        train_processed, test_processed = pipeline.fit_transform_all(
            data_path=config.INPUT_DATA_PATH
        )

        # 保存处理后的数据
        pipeline.save_processed_data(
            train_df=train_processed,
            test_df=test_processed,
            output_dir=config.OUTPUT_DIR
        )

        # 打印流水线摘要
        pipeline.print_summary()

        # 打印最终统计
        print("\n" + "=" * 60)
        print("特征工程完成！")
        print("=" * 60)
        print(f"\n训练集形状: {train_processed.shape}")
        print(f"测试集形状: {test_processed.shape}")
        print(f"\n输出目录: {config.OUTPUT_DIR}")
        print(f"\n生成的文件:")
        print(f"  1. {config.TRAIN_OUTPUT_PATH.name}")
        print(f"  2. {config.TEST_OUTPUT_PATH.name}")
        print(f"  3. preprocessors/ (标准化器和编码器)")
        print(f"  4. feature_list.txt (特征列表)")

        print("\n" + "=" * 60)
        print("后续步骤:")
        print("=" * 60)
        print("1. 使用 train_features.csv 和 test_features.csv 进行模型训练")
        print("2. 使用 preprocessors/ 中的对象进行新数据的预测")
        print("3. 参考 feature_list.txt 了解所有特征列名")
        print("4. city_id/country_id/weather_code_id 可用于 nn.Embedding")

        return True

    except Exception as e:
        print(f"\n错误: 特征工程过程中发生异常")
        print(f"异常信息: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
