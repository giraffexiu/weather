# 模型一：传统统计与线性基准

## 算法选择
- **分类任务**: Logistic Regression（逻辑回归）
- **回归任务**: Linear Regression（线性回归）

## 模型特点
- ✅ 提供线性基准（Baseline）
- ✅ L2 正则化（Ridge）防止过拟合
- ✅ 训练快速，可解释性强
- ✅ 适合作为其他复杂模型的对照组

## 应用场景
1. **分类**: 预测明天是否有雨/雪（二分类或多分类）
2. **回归**: 预测明天的具体温度值

## 文件结构
```
model_1_linear/
├── README.md                 # 本文件
├── config.py                 # 模型配置参数
├── train.py                  # 训练脚本
├── evaluate.py               # 评估脚本
├── predict.py                # 预测接口
├── model_utils.py            # 工具函数
├── requirements.txt          # 依赖包
├── saved_models/             # 保存的模型文件
│   ├── logistic_model.pkl
│   └── linear_model.pkl
└── results/                  # 训练结果
    ├── metrics.json
    └── feature_coefficients.csv
```

## 技术要点
- 使用 sklearn.linear_model.LogisticRegression
- 使用 sklearn.linear_model.Ridge (L2正则化)
- 交叉验证选择最佳正则化参数 alpha
- 输出特征系数，提供模型可解释性
