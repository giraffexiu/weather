"""
集成预测脚本
使用软投票集成进行天气预测
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
    PROBABILITY_CONVERSION_CONFIG
)
from model_wrapper import Model1Wrapper, Model3Wrapper
from probability_converter import ProbabilityConverter
from soft_voting_ensemble import SoftVotingEnsemble


def predict(
    input_path: str = None,
    output_path: str = None,
    batch_size: int = 128
):
    """
    执行集成预测
    
    Args:
        input_path: 输入数据路径（默认使用测试集）
        output_path: 输出路径（默认保存到 predictions 目录）
        batch_size: Model 3 的批次大小
    """
    print("\n" + "="*70)
    print("Daily Ensemble 预测")
    print("="*70)
    
    # 设置路径和验证配置
    setup_paths()
    validate_config()
    
    print(f"\n使用设备: {DEVICE}")
    
    # 加载输入数据
    if input_path is None:
        input_path = TEST_DATA_PATH
        print(f"\n使用默认测试集: {input_path}")
    else:
        print(f"\n加载输入数据: {input_path}")
    
    test_df = pd.read_csv(input_path)
    print(f"  数据大小: {len(test_df):,} 样本")
    print(f"  特征数量: {len(test_df.columns)} 列")
    
    # 初始化模型
    print("\n" + "="*70)
    print("初始化模型")
    print("="*70)
    
    model1 = Model1Wrapper()
    
    converter = ProbabilityConverter(PROBABILITY_CONVERSION_CONFIG)
    model3 = Model3Wrapper(probability_converter=converter)
    
    # 初始化集成器
    ensemble = SoftVotingEnsemble(
        model1_wrapper=model1,
        model3_wrapper=model3,
        weight_method='performance_based',
        verbose=True
    )
    
    # 准备输入
    print("\n" + "="*70)
    print("准备输入数据")
    print("="*70)
    
    print("\n准备 Model 1 输入...")
    X_model1 = model1.prepare_features(test_df)
    print(f"  特征矩阵: {X_model1.shape}")
    
    print("\n准备 Model 3 输入...")
    from dataset_loader import get_dataloaders
    loaders = get_dataloaders(batch_size=batch_size)
    test_loader = loaders['test']
    print(f"  批次数: {len(test_loader)}")
    
    # 执行预测
    print("\n" + "="*70)
    print("执行集成预测")
    print("="*70)
    
    result_df = ensemble.predict_with_dataframe(test_df, test_loader)
    
    print(f"\n预测完成！共 {len(result_df)} 个样本")
    
    # 显示预测结果摘要
    print("\n" + "="*70)
    print("预测结果摘要")
    print("="*70)
    
    print("\n【回归任务】")
    print("-"*70)
    regression_cols = [col for col in result_df.columns if 'ensemble_' in col and '_prob' not in col and '_pred' not in col]
    for col in regression_cols:
        task_name = col.replace('ensemble_', '')
        values = result_df[col]
        print(f"  {task_name:20s} | 均值: {values.mean():8.3f} | "
              f"标准差: {values.std():8.3f} | "
              f"范围: [{values.min():.3f}, {values.max():.3f}]")
    
    print("\n【分类任务】")
    print("-"*70)
    classification_tasks = [col.replace('ensemble_', '').replace('_pred', '') 
                           for col in result_df.columns if 'ensemble_' in col and '_pred' in col]
    for task in classification_tasks:
        prob_col = f'ensemble_{task}_prob'
        pred_col = f'ensemble_{task}_pred'
        
        probs = result_df[prob_col]
        preds = result_df[pred_col]
        
        positive_rate = (preds == 1).sum() / len(preds) * 100
        avg_prob = probs.mean()
        
        print(f"  {task:20s} | 正例率: {positive_rate:5.1f}% | "
              f"平均概率: {avg_prob:.3f} | "
              f"预测为1: {(preds == 1).sum():,} 个")
    
    # 保存结果
    if output_path is None:
        output_path = PREDICTIONS_DIR / "ensemble_predictions.csv"
    
    result_df.to_csv(output_path, index=False)
    print(f"\n预测结果已保存到: {output_path}")
    
    # 显示前几行
    print("\n" + "="*70)
    print("预测结果示例（前10行）")
    print("="*70)
    
    display_cols = ['time', 'city', 'ensemble_temp_mean', 'ensemble_precipitation', 
                   'ensemble_rain_prob', 'ensemble_rain_pred']
    display_cols = [col for col in display_cols if col in result_df.columns]
    
    print(result_df[display_cols].head(10).to_string(index=False))
    
    print("\n" + "="*70)
    print("预测完成！")
    print("="*70)
    
    return result_df


def main():
    """命令行接口"""
    parser = argparse.ArgumentParser(
        description='使用软投票集成进行天气预测',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python predict_ensemble.py                              # 使用默认测试集
  python predict_ensemble.py --input data.csv             # 使用自定义输入
  python predict_ensemble.py --output results.csv         # 指定输出路径
  python predict_ensemble.py --batch-size 256             # 调整批次大小
        """
    )
    
    parser.add_argument(
        '--input', type=str, default=None,
        help='输入数据CSV文件路径（默认使用测试集）'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='输出CSV文件路径（默认保存到 predictions 目录）'
    )
    parser.add_argument(
        '--batch-size', type=int, default=128,
        help='Model 3 的批次大小（默认128）'
    )
    
    args = parser.parse_args()
    
    try:
        result_df = predict(
            input_path=args.input,
            output_path=args.output,
            batch_size=args.batch_size
        )
    except KeyboardInterrupt:
        print("\n\n预测被中断")
    except Exception as e:
        print(f"\n\n预测出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
