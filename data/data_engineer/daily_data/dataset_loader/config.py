"""
数据加载配置文件
"""
from pathlib import Path

# ==================== 数据路径配置 ====================
# 相对于 dataset_loader 目录的路径
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR.parent / "processed_data"

TRAIN_DATA_PATH = DATA_DIR / "train_features.csv"
TEST_DATA_PATH = DATA_DIR / "test_features.csv"

# ==================== 时序窗口配置 ====================
SEQ_LENGTH = 7              # 使用过去7天的数据作为输入序列
PRED_HORIZON = 1            # 预测未来1天的目标值（可调整为多天）

# 是否在序列末尾添加当天数据（用于预测）
# False: 使用 day[t-7:t-1] 预测 day[t]
# True:  使用 day[t-6:t] 预测 day[t+1]
INCLUDE_CURRENT_DAY = False

# ==================== 目标变量配置 ====================
TARGET_TYPE = 'regression'  # 'regression' 或 'classification'

# 回归任务：预测的目标列（可以是多个）
TARGET_COLUMNS = [
    'temperature_2m_mean',      # 预测平均温度
    # 'precipitation_sum',      # 可添加：预测降水量
    # 'wind_speed_10m_max',     # 可添加：预测最大风速
]

# 分类任务配置（如果 TARGET_TYPE='classification'）
CLASSIFICATION_TARGET = 'is_rainy'  # 分类目标列
NUM_CLASSES = 2                      # 类别数量

# ==================== DataLoader 配置 ====================
BATCH_SIZE = 64             # 批次大小
NUM_WORKERS = 4             # 数据加载线程数（根据CPU核心数调整）
PIN_MEMORY = True           # 是否将数据固定到内存（GPU训练时推荐）
SHUFFLE_TRAIN = True        # 训练集是否打乱
SHUFFLE_TEST = False        # 测试集不打乱

# ==================== 缓存配置 ====================
USE_CACHE = True            # 是否启用缓存机制
CACHE_DIR = BASE_DIR / "cache"  # 缓存目录
CACHE_DIR.mkdir(exist_ok=True)

# 缓存级别
# 'memory': 仅内存缓存（快但重启丢失）
# 'disk':   磁盘缓存（慢但持久）
# 'both':   两者都用
CACHE_LEVEL = 'both'

# ==================== 数据处理配置 ====================
# 是否按城市分组（保证同一城市的时序连续性）
GROUP_BY_CITY = True

# 最小序列长度（过滤掉数据不足的城市）
MIN_SEQUENCE_LENGTH = SEQ_LENGTH + PRED_HORIZON + 10

# ==================== 调试配置 ====================
DEBUG = False               # 调试模式（只加载少量数据）
DEBUG_SAMPLES = 1000        # 调试模式下的样本数量

# 随机种子（保证可复现性）
RANDOM_SEED = 42
