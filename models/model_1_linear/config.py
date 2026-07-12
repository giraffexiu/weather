"""
模型一：线性模型配置文件
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
TASK_TYPE = "classification"  # 默认分类任务（是否下雨）

# 目标变量列名
TARGET_COLUMN = "target"  # 根据实际数据修改

# ==================== Logistic Regression 配置 ====================
LOGISTIC_CONFIG = {
    "penalty": "l2",  # L2正则化
    "C": 1.0,  # 正则化强度的倒数（较小的值表示更强的正则化）
    "solver": "lbfgs",  # 优化算法
    "max_iter": 1000,  # 最大迭代次数
    "random_state": 42,
    "class_weight": "balanced",  # 处理类别不平衡
    "n_jobs": -1  # 使用所有CPU核心
}

# 超参数搜索范围（用于交叉验证）
LOGISTIC_PARAM_GRID = {
    "C": [0.001, 0.01, 0.1, 1, 10, 100],
    "penalty": ["l2"],
    "solver": ["lbfgs", "saga"]
}

# ==================== Linear Regression 配置 ====================
LINEAR_CONFIG = {
    "alpha": 1.0,  # Ridge正则化参数
    "fit_intercept": True,
    "random_state": 42
}

# 超参数搜索范围
LINEAR_PARAM_GRID = {
    "alpha": [0.001, 0.01, 0.1, 1, 10, 100, 1000]
}

# ==================== 训练配置 ====================
TRAIN_CONFIG = {
    "test_size": 0.2,  # 如果需要进一步划分验证集
    "cv_folds": 5,  # 交叉验证折数
    "random_state": 42,
    "use_cross_validation": True,  # 是否使用交叉验证选择超参数
    "scoring": "f1_weighted"  # 分类任务评分标准（或 'r2' 用于回归）
}

# ==================== 特征配置 ====================
FEATURE_CONFIG = {
    "exclude_columns": ["date", "city", "country"],  # 需要排除的列
    "categorical_features": [],  # 分类特征（如果有）
    "numerical_features": []  # 数值特征（自动识别）
}

# ==================== 日志配置 ====================
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_file": os.path.join(RESULTS_DIR, "training.log")
}
