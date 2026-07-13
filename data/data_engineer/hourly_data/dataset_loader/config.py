"""
数据加载配置文件（小时数据）
"""
from pathlib import Path

# ==================== 数据路径配置 ====================
# 相对于 dataset_loader 目录的路径
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR.parent / "processed_data"

TRAIN_DATA_PATH = DATA_DIR / "train_features.csv"
TEST_DATA_PATH = DATA_DIR / "test_features.csv"

# ==================== 时序窗口配置 ====================
SEQ_LENGTH = 24             # 使用过去24小时的数据作为输入序列
PRED_HORIZON = 1            # 预测未来1小时的目标值（可调整为多步）

# 是否在序列末尾添加当前小时数据（用于预测）
# False: 使用 hour[t-24:t-1] 预测 hour[t]
# True:  使用 hour[t-23:t] 预测 hour[t+1]
INCLUDE_CURRENT_HOUR = False

# ==================== 目标变量配置 ====================
TARGET_TYPE = 'regression'  # 'regression' 或 'classification'

# 回归任务：多目标联合预测（共享表征提升各任务精度）
TARGET_COLUMNS = [
    'temperature_2m',           # 预测温度
    'precipitation',            # 预测降水量
    'wind_speed_10m',           # 预测风速
    'apparent_temperature',     # 预测体感温度
    'relative_humidity_2m',     # 预测相对湿度
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
# 关闭可大幅提速，数据已按城市+时间排序，简单窗口即可
GROUP_BY_CITY = False

# 最小序列长度（过滤掉数据不足的城市）
MIN_SEQUENCE_LENGTH = SEQ_LENGTH + PRED_HORIZON + 24

# ==================== 调试配置 ====================
DEBUG = False               # 调试模式（只加载少量数据）
DEBUG_SAMPLES = 10000       # 调试模式下的样本数量

# 随机种子（保证可复现性）
RANDOM_SEED = 42
