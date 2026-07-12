# 模型二：非线性强规则捕捉 - 随机森林

## 算法选择
- **Random Forest Classifier/Regressor**

## 模型特点
- ✅ 机器学习"常青树"，处理结构化数据的最佳选择
- ✅ 自动捕捉特征间的非线性交互
- ✅ OOB（袋外误差）提供无偏泛化评估
- ✅ feature_importances_ 提供强大的可解释性
- ✅ 对缺失值和异常值有较强的鲁棒性

## 应用场景
1. **分类**: 预测天气类型（晴天/雨天/雪天等）
2. **回归**: 预测温度、湿度等连续值
3. **特征重要性分析**: 识别影响天气的关键因素

## 文件结构
```
model_2_random_forest/
├── README.md                 # 本文件
├── config.py                 # 模型配置参数
├── train.py                  # 训练脚本
├── evaluate.py               # 评估脚本
├── predict.py                # 预测接口
├── feature_importance.py     # 特征重要性分析
├── model_utils.py            # 工具函数
├── requirements.txt          # 依赖包
├── saved_models/             # 保存的模型文件
│   └── random_forest_model.pkl
└── results/                  # 训练结果
    ├── metrics.json
    ├── feature_importance.csv
    ├── feature_importance.png
    └── oob_analysis.json
```

## 技术要点
- 使用 sklearn.ensemble.RandomForestClassifier/Regressor
- 启用 oob_score=True 获取袋外误差评估
- 调优参数：n_estimators, max_depth, min_samples_split
- 输出 feature_importances_ 进行可解释性分析
- 识别关键指标：湿度、气压降幅、风速等

## 特征重要性输出示例
```
特征名称                重要性得分
气压                   0.25
湿度                   0.18
温度                   0.15
气压降幅（3小时）       0.12
风速                   0.10
...
```

## OOB评估优势
- 无需单独划分验证集
- 利用每棵树未使用的样本进行验证
- 提供接近真实泛化误差的无偏估计
