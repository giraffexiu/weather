"""
快速测试脚本：验证 Hourly Ensemble 模块
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def test_imports():
    print("=" * 70)
    print("测试模块导入")
    print("=" * 70)
    try:
        from config import validate_config, setup_paths, DEVICE
        setup_paths()
        validate_config()
        print(f"  配置验证通过，设备: {DEVICE}")
    except Exception as e:
        print(f"  配置模块失败: {e}")
        return False

    try:
        from probability_converter import ProbabilityConverter
        from config import PROBABILITY_CONVERSION_CONFIG
        converter = ProbabilityConverter(PROBABILITY_CONVERSION_CONFIG)
        print("  概率转换器初始化成功")
    except Exception as e:
        print(f"  概率转换器失败: {e}")
        return False

    try:
        from model_wrapper import HourModelWrapper
        print("  模型包装器导入成功")
    except Exception as e:
        print(f"  模型包装器失败: {e}")
        return False

    return True


def test_model_loading():
    print("\n" + "=" * 70)
    print("测试模型加载")
    print("=" * 70)
    try:
        from model_wrapper import HourModelWrapper
        from probability_converter import ProbabilityConverter
        from config import PROBABILITY_CONVERSION_CONFIG

        converter = ProbabilityConverter(PROBABILITY_CONVERSION_CONFIG)
        model = HourModelWrapper(probability_converter=converter)
        model._ensure_model()
        print("  模型加载成功")
    except Exception as e:
        print(f"  模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    return True


def test_probability_converter():
    print("\n" + "=" * 70)
    print("测试概率转换")
    print("=" * 70)
    try:
        import numpy as np
        from probability_converter import ProbabilityConverter
        from config import PROBABILITY_CONVERSION_CONFIG

        converter = ProbabilityConverter(PROBABILITY_CONVERSION_CONFIG)
        rain_prob = converter.convert_rain_probability(np.array([0.0, 0.1, 1.0, 5.0]))
        print(f"  降水概率: {dict(zip([0.0, 0.1, 1.0, 5.0], rain_prob))}")
        print("  概率转换测试通过")
        return True
    except Exception as e:
        print(f"  概率转换失败: {e}")
        return False


def main():
    print("\n" + "=" * 70)
    print("Hourly Ensemble 模块测试")
    print("=" * 70)

    tests = [
        ("模块导入", test_imports),
        ("模型加载", test_model_loading),
        ("概率转换", test_probability_converter),
    ]

    results = []
    for name, func in tests:
        try:
            ok = func()
            results.append((name, ok))
        except Exception as e:
            print(f"\n测试 '{name}' 异常: {e}")
            results.append((name, False))

    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)
    for name, ok in results:
        print(f"  {name:20s} : {'PASS' if ok else 'FAIL'}")
    passed = sum(1 for _, r in results if r)
    print(f"\n  {passed}/{len(results)} 通过")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n测试被中断")
    except Exception as e:
        print(f"\n测试出错: {e}")
        import traceback
        traceback.print_exc()
