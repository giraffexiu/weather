# 模型三：高维泛化专家 - Wide & Deep 神经网络

## 算法选择
- **Wide & Deep Neural Network (PyTorch)**

## 模型架构
```
输入层
  ├─> Wide 侧（记忆能力）
  │    ├─ 当前气压
  │    ├─ 当前温度
  │    └─ 交叉特征（风向×月份、温度×湿度等）
  │
  └─> Deep 侧（泛化能力）
       ├─ 标准化数值特征（气压、温度、湿度、风速等）
       ├─ 风向 Embedding（8维或16维）
       └─ 3层全连接DNN
            ├─ FC1: 128 units + BatchNorm + ReLU + Dropout(0.3)
            ├─ FC2: 64 units + BatchNorm + ReLU + Dropout(0.2)
            └─ FC3: 32 units + BatchNorm + ReLU + Dropout(0.1)
  
融合层：Wide输出 + Deep输出
  └─> 输出层（Sigmoid/Softmax 或 Linear）
```

## 模型特点
- ✅ **Wide 侧**: 捕捉强规则（如"7月+南风=暴雨"）
- ✅ **Deep 侧**: 提取高阶非线性特征
- ✅ Embedding 处理风向等类别特征
- ✅ BatchNorm 加速训练并提升稳定性
- ✅ Dropout 防止过拟合
- ✅ 适合处理高维复杂天气数据

## 文件结构
```
model_3_deep_learning/
├── README.md                 # 本文件
├── config.py                 # 模型配置参数
├── model.py                  # Wide & Deep 模型定义
├── train.py                  # 训练脚本
├── evaluate.py               # 评估脚本
├── predict.py                # 预测接口
├── dataset.py                # 数据集类
├── utils.py                  # 工具函数
├── requirements.txt          # 依赖包
├── saved_models/             # 保存的模型文件
│   ├── wide_deep_best.pth
│   └── wide_deep_final.pth
├── checkpoints/              # 训练检查点
│   └── checkpoint_epoch_*.pth
└── results/                  # 训练结果
    ├── metrics.json
    ├── training_curves.png
    └── confusion_matrix.png
```

## 技术要点

### Wide 侧设计
- 直接输入原始特征和交叉特征
- 无隐藏层，线性变换
- 捕捉明显的特征组合规则

### Deep 侧设计
- 标准化的数值特征
- Embedding 层处理类别特征（风向、月份等）
- 3层全连接网络，逐层降维（128→64→32）
- 每层使用 BatchNorm + ReLU + Dropout

### 训练策略
- 优化器：Adam (lr=0.001) + ReduceLROnPlateau
- 损失函数：BCELoss（分类）/ MSELoss（回归）
- 早停机制：验证集loss连续10个epoch不下降
- 学习率调度：验证loss平台期降低学习率
- 梯度裁剪：防止梯度爆炸

### 交叉特征示例
1. **风向×月份**: 捕捉季风模式
2. **温度×湿度**: 捕捉体感温度和降雨概率
3. **气压×气压变化率**: 捕捉天气突变信号
4. **风速×风向**: 捕捉风力特征

## 性能优化
- GPU 加速训练
- 混合精度训练（AMP）
- DataLoader 多进程加载
- 梯度累积（小batch size场景）

## 可视化输出
- 训练/验证损失曲线
- 训练/验证准确率曲线
- 混淆矩阵（分类任务）
- 预测值vs真实值散点图（回归任务）
