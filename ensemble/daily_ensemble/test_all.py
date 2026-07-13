"""
快速测试脚本：验证所有模块是否正常工作
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def test_imports():
    """测试所有导入"""
    print("="*70)
    print("测试模块导入")
    print("="*70)
    
    try:
        print("\n1. 导入配置模块...")
        from config import validate_config, setup_paths, DEVICE
        setup_paths()
        validate_config()
        print(f"   ✓ 配置验证通过，设备: {DEVICE}")
    except Exception as e:
        print(f"   ✗ 配置模块失败: {e}")
        return False
    
    try:
        print("\n2. 导入概率转换器...")
        from probability_converter import ProbabilityConverter
        from config import PROBABILITY_CONVERSION_CONFIG
        converter = ProbabilityConverter(PROBABILITY_CONVERSION_CONFIG)
        print("   ✓ 概率转换器初始化成功")
    except Exception as e:
        print(f"   ✗ 概率转换器失败: {e}")
        return False
    
    try:
        print("\n3. 导入模型包装器...")
        from model_wrapper import Model1Wrapper, Model3Wrapper
        print("   ✓ 模型包装器导入成功")
    except Exception as e:
        print(f"   ✗ 模型包装器失败: {e}")
        return False
    
    try:
        print("\n4. 导入集成器...")
        from soft_voting_ensemble import SoftVotingEnsemble
        print("   ✓ 集成器导入成功")
    except Exception as e:
        print(f"   ✗ 集成器失败: {e}")
        return False
    
    return True


def test_model_loading():
    """测试模型加载"""
    print("\n" + "="*70)
    print("测试模型加载")
    print("="*70)
    
    try:
        print("\n1. 加载 Model 1...")
        from model_wrapper import Model1Wrapper
        model1 = Model1Wrapper()
        print("   ✓ Model 1 加载成功")
        print(f"   - 分类模型: {len(model1.classification_models)} 个")
        print(f"   - 回归模型: {len(model1.regression_models)} 个")
        print(f"   - 特征数量: {len(model1.feature_names)} 个")
    except Exception as e:
        print(f"   ✗ Model 1 加载失败: {e}")
        return False
    
    try:
        print("\n2. 加载 Model 3...")
        from model_wrapper import Model3Wrapper
        from probability_converter import ProbabilityConverter
        from config import PROBABILITY_CONVERSION_CONFIG
        
        converter = ProbabilityConverter(PROBABILITY_CONVERSION_CONFIG)
        model3 = Model3Wrapper(probability_converter=converter)
        print("   ✓ Model 3 加载成功")
    except Exception as e:
        print(f"   ✗ Model 3 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_probability_converter():
    """测试概率转换"""
    print("\n" + "="*70)
    print("测试概率转换")
    print("="*70)
    
    try:
        import numpy as np
        from probability_converter import ProbabilityConverter
        from config import PROBABILITY_CONVERSION_CONFIG
        
        converter = ProbabilityConverter(PROBABILITY_CONVERSION_CONFIG)
        
        # 测试降雨概率转换
        rain_sum = np.array([0.0, 0.05, 0.1, 1.0, 5.0])
        rain_prob = converter.convert_rain_probability(rain_sum)
        
        print("\n降雨量 -> 概率:")
        for r, p in zip(rain_sum, rain_prob):
            print(f"   Rain: {r:5.2f} mm -> Prob: {p:.3f}")
        
        # 测试恶劣天气概率
        temp_range = np.array([5.0, 15.0, 25.0])
        wind_speed = np.array([5.0, 10.0, 20.0])
        precipitation = np.array([0.0, 5.0, 15.0])
        
        severe_prob = converter.convert_severe_probability(
            temp_range, wind_speed, precipitation
        )
        
        print("\n恶劣天气综合评分:")
        for t, w, p, s in zip(temp_range, wind_speed, precipitation, severe_prob):
            print(f"   T_range: {t:5.1f}°C, Wind: {w:5.1f}m/s, "
                  f"Precip: {p:5.1f}mm -> Prob: {s:.3f}")
        
        print("\n   ✓ 概率转换测试通过")
        return True
        
    except Exception as e:
        print(f"   ✗ 概率转换失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ensemble_weights():
    """测试集成权重计算"""
    print("\n" + "="*70)
    print("测试集成权重计算")
    print("="*70)
    
    try:
        from model_wrapper import Model1Wrapper, Model3Wrapper
        from probability_converter import ProbabilityConverter
        from soft_voting_ensemble import SoftVotingEnsemble
        from config import PROBABILITY_CONVERSION_CONFIG
        
        model1 = Model1Wrapper()
        converter = ProbabilityConverter(PROBABILITY_CONVERSION_CONFIG)
        model3 = Model3Wrapper(probability_converter=converter)
        
        print("\n初始化集成器...")
        ensemble = SoftVotingEnsemble(
            model1_wrapper=model1,
            model3_wrapper=model3,
            weight_method='performance_based',
            verbose=False  # 不打印详细信息
        )
        
        print("   ✓ 集成器初始化成功")
        print("\n权重示例:")
        print("   回归任务:")
        for task in list(ensemble.weights['regression'].keys())[:3]:
            w = ensemble.weights['regression'][task]
            print(f"     {task:20s} | Model1: {w['model1']:.3f} | Model3: {w['model3']:.3f}")
        
        print("   分类任务:")
        for task in ensemble.weights['classification'].keys():
            w = ensemble.weights['classification'][task]
            print(f"     {task:20s} | Model1: {w['model1']:.3f} | Model3: {w['model3']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"   ✗ 集成器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "="*70)
    print("Daily Ensemble 模块测试")
    print("="*70)
    
    tests = [
        ("模块导入", test_imports),
        ("模型加载", test_model_loading),
        ("概率转换", test_probability_converter),
        ("集成权重", test_ensemble_weights)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n测试 '{test_name}' 发生异常: {e}")
            results.append((test_name, False))
    
    # 总结
    print("\n" + "="*70)
    print("测试总结")
    print("="*70)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {test_name:20s} : {status}")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    print("\n" + "="*70)
    if passed == total:
        print(f"✅ 所有测试通过 ({passed}/{total})")
        print("="*70)
        print("\n系统已就绪，可以执行预测或评估:")
        print("  - 预测: python predict_ensemble.py")
        print("  - 评估: python evaluate_ensemble.py")
        print("="*70)
    else:
        print(f"⚠️  部分测试失败 ({passed}/{total} 通过)")
        print("="*70)
        print("\n请检查失败的模块并修复问题")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n测试被中断")
    except Exception as e:
        print(f"\n\n测试出错: {e}")
        import traceback
        traceback.print_exc()
