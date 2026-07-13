"""
小时级集成预测脚本
基于Wide & Deep模型进行预测（含逆标准化 + 物理约束）
"""
import sys
import argparse
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    TEST_DATA_PATH,
    PREDICTIONS_DIR,
    DEVICE,
    setup_paths,
    validate_config,
    PROBABILITY_CONVERSION_CONFIG,
    HOUR_TARGET_COLUMNS,
)


def predict(input_path: str = None, output_path: str = None, batch_size: int = 512):
    print("\n" + "=" * 70)
    print("Hourly Ensemble 预测")
    print("=" * 70)

    setup_paths()
    validate_config()
    print(f"设备: {DEVICE}")

    input_path = input_path or TEST_DATA_PATH
    print(f"\n输入: {input_path}")

    from dataset_loader import get_dataloader
    loader = get_dataloader(
        split='test', data_path=input_path, batch_size=batch_size,
        shuffle=False, num_workers=0,
    )
    print(f"批次数: {len(loader)}")

    from model_wrapper import HourModelWrapper
    from probability_converter import ProbabilityConverter

    converter = ProbabilityConverter(PROBABILITY_CONVERSION_CONFIG)
    model = HourModelWrapper(probability_converter=converter)

    print("\n执行预测...")
    results = model.predict(loader)
    df = pd.read_csv(input_path)

    # 填入回归预测（已是真实物理单位）
    shift = len(df) - len(results['regression'][HOUR_TARGET_COLUMNS[0]])
    for col in HOUR_TARGET_COLUMNS:
        df[f'pred_{col}'] = float('nan')
        if shift < len(df):
            df.loc[shift:, f'pred_{col}'] = results['regression'][col]

    # 填入分类概率
    for task, pred_dict in results.get('classification', {}).items():
        df[f'prob_{task}'] = float('nan')
        if shift < len(df):
            df.loc[shift:, f'prob_{task}'] = pred_dict['probability']

    output_path = output_path or PREDICTIONS_DIR / "hourly_predictions.csv"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\n预测完成 — {output_path}")
    print(f"总行数: {len(df)}, 有效预测: {len(df) - shift}")

    print("\n【预测摘要（真实物理单位）】")
    print("-" * 55)
    for col in HOUR_TARGET_COLUMNS:
        vals = df[f'pred_{col}'].dropna()
        print(f"  {col:<26} 均值={vals.mean():9.4f}  范围=[{vals.min():.2f}, {vals.max():.2f}]")

    return df


def main():
    parser = argparse.ArgumentParser(description="Hourly 集成预测")
    parser.add_argument('--input', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=512)
    args = parser.parse_args()

    try:
        predict(args.input, args.output, args.batch_size)
    except KeyboardInterrupt:
        print("\n预测被中断")
    except Exception as e:
        print(f"\n预测出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
