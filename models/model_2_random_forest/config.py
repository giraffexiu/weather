"""
模型二：随机森林配置文件
"""
import os

# ==================== 路径配置 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))

# 添加dataset_loader到Python路径，以便导入
DATASET_LOADER_DIR = os.path.join(PROJECT_ROOT, "data/data_engineer/daily_data/dataset_loader")
import sys
sys.path.insert(0, DATASET_LOADER_DIR)

# 可以从dataset_loader导入配置和数据加载工具
# from data_loader import get_train_dataloader, get_test_dataloader
# from feature_config import get_feature_groups, get_all_feature_columns

# 数据路径（CSV格式）
DATA_DIR = os.path.join(PROJECT_ROOT, "data/data_engineer/daily_data/processed_data")
TRAIN_DATA = os.path.join(DATA_DIR, "train_features.csv")
TEST_DATA = os.path.join(DATA_DIR, "test_features.csv")

# 模型保存路径
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# 创建必要目录
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==================== 模型配置 ====================
# 任务类型：'classification' 或 'regression'
TASK_TYPE = "classification"

# 目标变量列名
TARGET_COLUMN = "target"

# ==================== Random Forest 配置 ====================
RF_CONFIG = {
    "n_estimators": 200,  # 决策树数量
    "max_depth": 20,  # 树的最大深度
    "min_samples_split": 5,  # 内部节点分裂所需的最小样本数
    "min_samples_leaf": 2,  # 叶子节点的最小样本数
    "max_features": "sqrt",  # 每次分裂考虑的特征数（sqrt 或 log2）
    "bootstrap": True,  # 使用bootstrap采样
    "oob_score": True,  # 启用袋外误差评估（OOB）
    "random_state": 42,
    "n_jobs": -1,  # 使用所有CPU核心
    "verbose": 1,  # 显示训练进度
    "class_weight": "balanced"  # 处理类别不平衡（仅分类）
}

# 超参数搜索范围（用于网格搜索或随机搜索）
RF_PARAM_GRID = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [10, 20, 30, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None]
}

# 随机搜索配置（比网格搜索更快）
RANDOM_SEARCH_CONFIG = {
    "n_iter": 50,  # 随机搜索迭代次数
    "cv": 5,  # 交叉验证折数
    "scoring": "f1_weighted",  # 分类评分（或 'neg_mean_squared_error' 用于回归）
    "random_state": 42,
    "n_jobs": -1,
    "verbose": 2
}

# ==================== 特征重要性配置 ====================
FEATURE_IMPORTANCE_CONFIG = {
    "top_n": 20,  # 显示前N个重要特征
    "plot_style": "bar",  # 可视化风格：'bar' 或 'horizontal'
    "save_plot": True,  # 是否保存图表
    "plot_file": os.path.join(RESULTS_DIR, "feature_importance.png"),
    "csv_file": os.path.join(RESULTS_DIR, "feature_importance.csv")
}

# ==================== 训练配置 ====================
TRAIN_CONFIG = {
    "use_hyperparameter_tuning": True,  # 是否进行超参数调优
    "tuning_method": "random",  # 'grid' 或 'random'
    "cv_folds": 5,  # 交叉验证折数
    "random_state": 42,
    "enable_oob": True,  # 启用OOB评估
    "early_stopping": False  # 是否使用早停（需要验证集）
}

# ==================== OOB 评估配置 ====================
OOB_CONFIG = {
    "enable": True,  # 启用OOB评估
    "save_results": True,
    "results_file": os.path.join(RESULTS_DIR, "oob_analysis.json")
}

# ==================== 特征配置 ====================
FEATURE_CONFIG = {
    "exclude_columns": ["date", "city", "country"],  # 需要排除的列
    "key_features": [  # 关键特征（用于重点分析）
        "pressure",  # 气压
        "humidity",  # 湿度
        "temperature",  # 温度
        "wind_speed",  # 风速
        "pressure_diff_3h",  # 气压降幅（3小时）
    ]
}

# ==================== 日志配置 ====================
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_file": os.path.join(RESULTS_DIR, "training.log")
}

# ==================== 可解释性分析配置 ====================
INTERPRETABILITY_CONFIG = {
    "enable": True,
    "methods": ["feature_importance", "partial_dependence"],  # 使用的方法
    "pdp_features": ["pressure", "humidity", "temperature"],  # 部分依赖图的特征
    "save_plots": True
}
