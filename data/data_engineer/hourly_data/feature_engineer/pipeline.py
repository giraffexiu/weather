"""
特征工程流水线模块（小时数据版本）
功能：
1. 整合所有特征处理步骤
2. 管理训练/测试集的切分
3. 协调各个特征处理器的执行顺序
4. 保存和加载完整的特征工程流水线
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
import warnings

from .time_features import TimeFeatureExtractor
from .categorical_encoder import CategoricalEncoder
from .numerical_scaler import NumericalScaler
from .feature_creator import FeatureCreator


class FeatureEngineeringPipeline:
    """特征工程完整流水线（小时数据版）"""

    def __init__(
        self,
        time_column: str = 'time',
        categorical_columns: list = None,
        numerical_columns: list = None,
        use_cyclical_encoding: bool = True,
        create_derived_features: bool = True,
        scaling_method: str = 'standard',
        train_end_date: str = '2023-12-31',
        test_start_date: str = '2024-01-01'
    ):
        self.time_column = time_column
        self.categorical_columns = categorical_columns or []
        self.numerical_columns = numerical_columns or []
        self.use_cyclical_encoding = use_cyclical_encoding
        self.create_derived_features = create_derived_features
        self.scaling_method = scaling_method
        self.train_end_date = train_end_date
        self.test_start_date = test_start_date

        # 初始化各个处理器
        self.time_extractor = TimeFeatureExtractor(
            time_column=time_column,
            use_cyclical=use_cyclical_encoding
        )

        self.categorical_encoder = (
            CategoricalEncoder(categorical_columns)
            if self.categorical_columns else None
        )

        self.feature_creator = FeatureCreator() if self.create_derived_features else None

        self.numerical_scaler = None
        self.is_fitted = False

    def load_data(self, data_path: Path) -> pd.DataFrame:
        """加载数据"""
        print(f"\n{'='*60}")
        print(f"正在加载数据: {data_path}")
        print(f"{'='*60}")

        df = pd.read_csv(data_path)
        print(f"数据形状: {df.shape}")
        if self.time_column in df.columns:
            df[self.time_column] = pd.to_datetime(df[self.time_column])
            print(f"时间范围: {df[self.time_column].min()} 至 {df[self.time_column].max()}")
        return df

    def split_train_test(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """按时间切分训练集和测试集"""
        print(f"\n{'='*60}")
        print("时间切分训练集和测试集")
        print(f"{'='*60}")

        df = df.copy()
        df[self.time_column] = pd.to_datetime(df[self.time_column])

        train_df = df[df[self.time_column] <= self.train_end_date].copy()
        test_df = df[df[self.time_column] >= self.test_start_date].copy()

        print(f"训练集: {len(train_df):,} 条 "
              f"({train_df[self.time_column].min()} 至 {train_df[self.time_column].max()})")
        print(f"测试集: {len(test_df):,} 条 "
              f"({test_df[self.time_column].min()} 至 {test_df[self.time_column].max()})")

        if len(train_df) == 0 or len(test_df) == 0:
            warnings.warn("训练集或测试集为空，请检查切分日期！")

        return train_df, test_df

    def fit_transform_train(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """拟合并转换训练集"""
        print(f"\n{'='*60}")
        print("处理训练集")
        print(f"{'='*60}")

        df = train_df.copy()

        # 1. 时间特征提取（含小时周期编码）
        print("\n[1/5] 提取时间特征（含小时周期编码）...")
        df = self.time_extractor.fit_transform(df)
        time_features = self.time_extractor.get_feature_names()
        print(f"  生成 {len(time_features)} 个时间特征")

        # 2. 创建派生特征
        if self.feature_creator:
            print("\n[2/5] 创建派生特征...")
            df = self.feature_creator.fit_transform(df)
            derived_features = self.feature_creator.get_feature_names()
            print(f"  生成 {len(derived_features)} 个派生特征")
        else:
            print("\n[2/5] 跳过派生特征创建")
            derived_features = []

        # 3. 类别特征编码（city, country, weather_code）
        if self.categorical_encoder:
            print("\n[3/5] 编码类别特征...")
            df = self.categorical_encoder.fit_transform(df)
            for col in self.categorical_columns:
                num_classes = self.categorical_encoder.get_num_classes(col)
                print(f"  {col}: {num_classes} 个类别")
        else:
            print("\n[3/5] 跳过类别特征编码")

        # 4. 数值特征标准化
        # 收集所有需要标准化的数值列
        all_numerical_cols = self.numerical_columns.copy()

        # 添加派生的数值特征（排除二值和分类特征）
        derived_numerical = [
            feat for feat in derived_features
            if feat in df.columns
            and not feat.startswith('is_')
            and '_level' not in feat
        ]
        all_numerical_cols.extend(derived_numerical)

        # 添加时间周期编码特征
        if self.use_cyclical_encoding:
            cyclical_features = [
                col for col in df.columns
                if col.endswith('_sin') or col.endswith('_cos')
            ]
            all_numerical_cols.extend(cyclical_features)

        # 去重
        all_numerical_cols = list(set(all_numerical_cols))

        print(f"\n[4/5] 标准化数值特征 (共 {len(all_numerical_cols)} 个)...")
        self.numerical_scaler = NumericalScaler(
            numerical_columns=all_numerical_cols,
            method=self.scaling_method
        )
        df = self.numerical_scaler.fit_transform(df)
        print(f"  使用 {self.scaling_method.upper()} 标准化")

        # 5. 数据质量检查
        print("\n[5/5] 数据质量检查...")
        self._quality_check(df, "训练集")

        self.is_fitted = True
        return df

    def transform_test(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """转换测试集（使用训练集的参数）"""
        if not self.is_fitted:
            raise ValueError(
                "流水线未拟合，请先对训练集调用 fit_transform_train()"
            )

        print(f"\n{'='*60}")
        print("处理测试集")
        print(f"{'='*60}")

        df = test_df.copy()

        # 1. 时间特征
        print("\n[1/4] 提取时间特征...")
        df = self.time_extractor.transform(df)

        # 2. 派生特征
        if self.feature_creator:
            print("\n[2/4] 创建派生特征...")
            df = self.feature_creator.transform(df)

        # 3. 类别特征编码
        if self.categorical_encoder:
            print("\n[3/4] 编码类别特征...")
            df = self.categorical_encoder.transform(df)

        # 4. 数值特征标准化
        print("\n[4/4] 标准化数值特征...")
        df = self.numerical_scaler.transform(df)

        print("\n数据质量检查...")
        self._quality_check(df, "测试集")

        return df

    def fit_transform_all(
        self,
        data_path: Path
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """完整流程：加载、切分、处理训练集和测试集"""
        df = self.load_data(data_path)
        train_df, test_df = self.split_train_test(df)
        train_processed = self.fit_transform_train(train_df)
        test_processed = self.transform_test(test_df)
        return train_processed, test_processed

    def save_processed_data(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        output_dir: Path
    ) -> None:
        """保存处理后的数据和预处理对象"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print("保存处理后的数据")
        print(f"{'='*60}")

        # 保存数据
        train_path = output_dir / "train_features.csv"
        test_path = output_dir / "test_features.csv"
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        print(f"训练集已保存: {train_path} "
              f"({len(train_df):,} 行 × {len(train_df.columns)} 列)")
        print(f"测试集已保存: {test_path} "
              f"({len(test_df):,} 行 × {len(test_df.columns)} 列)")

        # 保存预处理对象
        preprocessor_dir = output_dir / "preprocessors"
        preprocessor_dir.mkdir(exist_ok=True)

        if self.numerical_scaler:
            self.numerical_scaler.save(preprocessor_dir / "scaler.pkl")

        if self.categorical_encoder:
            self.categorical_encoder.save_mappings(preprocessor_dir)

        print(f"\n预处理对象已保存至: {preprocessor_dir}")

        # 生成特征列表文件
        feature_list_path = output_dir / "feature_list.txt"
        with open(feature_list_path, 'w', encoding='utf-8') as f:
            f.write("小时天气数据 - 特征工程后的所有特征列名\n")
            f.write("=" * 60 + "\n\n")
            for i, col in enumerate(train_df.columns, 1):
                f.write(f"{i:3d}. {col}\n")
        print(f"特征列表已保存: {feature_list_path}")

    def _quality_check(self, df: pd.DataFrame, dataset_name: str) -> None:
        """数据质量检查"""
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print(f"\n  警告: {dataset_name}存在缺失值:")
            print(missing[missing > 0])
        else:
            print("  ✓ 无缺失值")

        numerical_cols = df.select_dtypes(include=[np.number]).columns
        inf_check = np.isinf(df[numerical_cols]).sum()
        if inf_check.sum() > 0:
            print(f"  警告: {dataset_name}存在无穷值:")
            print(inf_check[inf_check > 0])
        else:
            print("  ✓ 无无穷值")

        constant_cols = [col for col in df.columns if df[col].nunique() == 1]
        if constant_cols:
            print(f"  警告: {dataset_name}存在常量列: {constant_cols}")
        else:
            print("  ✓ 无常量列")

    def get_summary(self) -> dict:
        """获取流水线处理摘要"""
        if not self.is_fitted:
            return {"status": "未拟合"}

        summary = {
            "时间特征": (
                self.time_extractor.get_feature_names()
                if self.time_extractor else []
            ),
            "派生特征": (
                self.feature_creator.get_feature_names()
                if self.feature_creator else []
            ),
            "类别特征": (
                self.categorical_encoder.get_info()
                if self.categorical_encoder else {}
            ),
            "标准化方法": self.scaling_method
        }
        return summary

    def print_summary(self) -> None:
        """打印流水线处理摘要"""
        summary = self.get_summary()

        print(f"\n{'='*60}")
        print("特征工程流水线摘要")
        print(f"{'='*60}")
        if self.time_extractor:
            print(f"时间特征: {len(summary['时间特征'])} 个（含小时周期编码）")
        if self.feature_creator:
            print(f"派生特征: {len(summary['派生特征'])} 个")
            if summary['派生特征']:
                print(f"  {summary['派生特征']}")
        if self.categorical_encoder:
            print(f"类别特征: {len(self.categorical_columns)} 个")
            for col, info in summary['类别特征'].items():
                print(f"  {col}: {info['num_classes']} 个类别")
        print(f"标准化方法: {summary['标准化方法'].upper()}")
        print(f"{'='*60}")
