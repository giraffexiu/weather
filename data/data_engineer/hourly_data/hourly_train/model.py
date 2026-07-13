"""
Wide & Deep 模型 —— 高维泛化专家

Wide 侧：直接输入气压、温度以及风向×月份的交叉特征（捕捉强规则）
Deep 侧：Embedding + 标准化数值特征 → 3层DNN（Dropout + BatchNorm）
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

# 从 dataset_loader 获取特征配置，动态计算索引
_HERE = Path(__file__).parent
_PARENT = _HERE.parent
sys.path.insert(0, str(_PARENT))
from dataset_loader import feature_config as _fc

# —— 构建 numerical 张量中各特征的索引映射 ——
_full_numerical = _fc.NUMERICAL_FEATURES + _fc.ORDINAL_FEATURES + _fc.TIME_FEATURES
_NUM_IDX = {name: i for i, name in enumerate(_full_numerical)}

# Wide 侧需要的原始特征列名 → 在 numerical 中的索引
_WIDE_RAW_COLS = ['temperature_2m', 'pressure_msl', 'wind_speed_10m']
_WIDE_RAW_IDX = [_NUM_IDX[c] for c in _WIDE_RAW_COLS]

# 时间特征（整个 time 段）
_TIME_COLS = _fc.TIME_FEATURES
_TIME_START = _NUM_IDX[_TIME_COLS[0]]
_TIME_END   = _NUM_IDX[_TIME_COLS[-1]] + 1

# 等级特征
_LEVEL_COLS = _fc.ORDINAL_FEATURES
_LEVEL_START = _NUM_IDX[_LEVEL_COLS[0]]
_LEVEL_END   = _NUM_IDX[_LEVEL_COLS[-1]] + 1

# 特定时间列索引
_MONTH_IDX = _NUM_IDX['month']
_HOUR_IDX  = _NUM_IDX['hour']

# 动态计算维度
NUM_NUMERICAL = len(_fc.NUMERICAL_FEATURES) + len(_fc.ORDINAL_FEATURES) + len(_fc.TIME_FEATURES)
NUM_CYCLICAL  = len(_fc.CYCLICAL_FEATURES)
NUM_BINARY    = len(_fc.BINARY_FEATURES)
NUM_CAT       = len(_fc.CATEGORICAL_FEATURES)

# Wide 输入维度: 3(原始) + 8(时间) + 5(等级) + 1(season) + 1(day_period) + 3(cat)
WIDE_INPUT_DIM = len(_WIDE_RAW_COLS) + len(_TIME_COLS) + len(_LEVEL_COLS) + 1 + 1 + NUM_CAT


# ==================== 特征维度配置 ====================
FEATURE_DIMS = {
    'city_id':       49,       # 城市 Embedding
    'country_id':    29,       # 国家 Embedding
    'weather_code_id': 13,     # 天气编码 Embedding
    'month':         12,       # 月份 × 风向交叉
    'hour':          24,       # 小时 × 温度交叉
    'season':        4,        # 季节 × 湿度交叉
    'day_period':    5,        # 时段 × 云量交叉
    'wind_level':    6,        # 风力等级
    'temperature_level': 7,    # 温度等级
    'humidity_level': 5,       # 湿度等级（假设0-4）
    'precipitation_level': 5,  # 降水等级（假设0-4，从 precipitation_intensity 派生）
}

# Embedding 维度
EMBEDDING_DIMS = {
    'city_id':       8,
    'country_id':    6,
    'weather_code_id': 4,
    'month':         4,
    'hour':          6,
    'season':        2,
    'day_period':    3,
    'wind_level':    3,
    'temperature_level': 4,
    'humidity_level': 3,
}

# Wide 交叉特征组合
CROSS_FEATURE_PAIRS = [
    ('month',          'wind_level'),           # 7月+大风=暴雨
    ('season',         'humidity_level'),       # 夏季+潮湿
    ('hour',           'temperature_level'),    # 午后+高温
    ('day_period',     'weather_code_id'),      # 时段+天气类型
    ('month',          'country_id'),           # 月份+地域差异
]

# ==================== 构建 Wide 交叉特征 ====================
def build_wide_features(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    从 batch 中提取 Wide 侧的原始特征和交叉特征
    
    Args:
        batch: dataset 返回的字典，每个值 shape = (batch, seq_len, n_feat)
               取最后一个时间步
    
    Returns:
        wide_input: (batch, wide_dim) 用于 Wide 线性层
    """
    # 取最后一个时间步 (batch, n_feat)
    def last_step(key):
        t = batch[key]  # (B, S, F)
        return t[:, -1, :]  # (B, F)

    # 直接数值特征（用动态索引）
    num_raw = last_step('numerical')  # (B, num_numerical)
    pressure   = num_raw[:, _WIDE_RAW_IDX[1]:_WIDE_RAW_IDX[1]+1]   # pressure_msl
    temp       = num_raw[:, _WIDE_RAW_IDX[0]:_WIDE_RAW_IDX[0]+1]   # temperature_2m
    wind_speed = num_raw[:, _WIDE_RAW_IDX[2]:_WIDE_RAW_IDX[2]+1]   # wind_speed_10m
    
    wide_linear_features = torch.cat([pressure, temp, wind_speed], dim=1)

    # 时间 + 等级特征（用动态索引区间）
    time_feats  = num_raw[:, _TIME_START:_TIME_END]   # year..week_of_year
    level_feats = num_raw[:, _LEVEL_START:_LEVEL_END] # temperature_level..cloud_level

    # season / day_period (已编码为数值)
    season_val = last_step('season')       # (B, 1)
    dayperiod_val = last_step('day_period')  # (B, 1)

    # categorical ID
    cat_feats = last_step('categorical')   # (B, 3): city_id, country_id, weather_code_id
    country_id = cat_feats[:, 1:2]
    weather_id = cat_feats[:, 2:3]

    # 组装所有 Wide 输入：[线性特征 | 时间 | 等级 | 季节 | 时段 | 类别]
    wide_all = torch.cat([
        wide_linear_features,   # 3
        time_feats,             # 8
        level_feats,            # 5
        season_val,             # 1
        dayperiod_val,          # 1
        cat_feats.float(),      # 3
    ], dim=1)

    return wide_all


# ==================== Wide & Deep 模型 ====================
class WideAndDeep(nn.Module):
    """
    Wide & Deep 架构用于小时天气预测
    
    Wide 侧: 线性 + 交叉特征变换 → 直接学习强关联规则
    Deep 侧: Embedding + DNN → 挖掘高阶非线性模式
    """

    def __init__(
        self,
        num_numerical: int = None,
        num_cyclical: int = None,
        num_binary: int = None,
        num_targets: int = 5,
        wide_input_dim: int = None,
        deep_hidden_dims: list = None,
        dropout: float = 0.3,
    ):
        super().__init__()

        # 默认使用动态维度
        num_numerical = num_numerical or NUM_NUMERICAL
        num_cyclical  = num_cyclical or NUM_CYCLICAL
        num_binary    = num_binary or NUM_BINARY
        wide_input_dim = wide_input_dim or WIDE_INPUT_DIM
        deep_hidden_dims = deep_hidden_dims or [512, 256, 128]

        self.num_targets = num_targets

        # ===== Embedding 层 =====
        self.embed_city = nn.Embedding(FEATURE_DIMS['city_id'], EMBEDDING_DIMS['city_id'])
        self.embed_country = nn.Embedding(FEATURE_DIMS['country_id'], EMBEDDING_DIMS['country_id'])
        self.embed_weather = nn.Embedding(FEATURE_DIMS['weather_code_id'], EMBEDDING_DIMS['weather_code_id'])

        self.embed_month = nn.Embedding(FEATURE_DIMS['month'] + 1, EMBEDDING_DIMS['month'], padding_idx=0)
        self.embed_hour = nn.Embedding(FEATURE_DIMS['hour'], EMBEDDING_DIMS['hour'])
        self.embed_season = nn.Embedding(FEATURE_DIMS['season'], EMBEDDING_DIMS['season'])
        self.embed_dayperiod = nn.Embedding(FEATURE_DIMS['day_period'], EMBEDDING_DIMS['day_period'])

        total_embed_dim = (
            EMBEDDING_DIMS['city_id'] +
            EMBEDDING_DIMS['country_id'] +
            EMBEDDING_DIMS['weather_code_id'] +
            EMBEDDING_DIMS['month'] +
            EMBEDDING_DIMS['hour'] +
            EMBEDDING_DIMS['season'] +
            EMBEDDING_DIMS['day_period']
        )

        # ===== Deep 侧 DNN =====
        deep_input_dim = num_numerical + num_cyclical + num_binary + total_embed_dim
        layers = []
        prev_dim = deep_input_dim
        for hidden_dim in deep_hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        self.deep_dnn = nn.Sequential(*layers)
        self.deep_output_dim = deep_hidden_dims[-1]

        # ===== Wide 侧 =====
        self.wide_linear = nn.Linear(wide_input_dim, 64)
        self.wide_bn = nn.BatchNorm1d(64)
        self.wide_dropout = nn.Dropout(dropout)

        # ===== 融合层 =====
        combined_dim = self.deep_output_dim + 64
        self.final_fc = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_targets),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            batch: {
                'categorical': (B, S, 3),     # city_id, country_id, weather_code_id
                'numerical':   (B, S, 35),     # 连续 + 等级 + 时间
                'cyclical':    (B, S, 8),      # sin/cos 编码
                'binary':      (B, S, 15),     # 二值标志
                'season':      (B, S, 1),      # 季节编码
                'day_period':  (B, S, 1),      # 时段编码
            }
        
        Returns:
            pred: (B, num_targets)
        """
        B = batch['categorical'].size(0)

        # ---- 取最后一个时间步 ----
        cat_last   = batch['categorical'][:, -1, :].long()   # (B, 3)
        num_last   = batch['numerical'][:, -1, :]             # (B, 35)
        cyc_last   = batch['cyclical'][:, -1, :]              # (B, 8)
        bin_last   = batch['binary'][:, -1, :]                # (B, 15)
        season_last = batch['season'][:, -1, :].long()        # (B, 1)
        dp_last    = batch['day_period'][:, -1, :].long()     # (B, 1)

        city_id    = cat_last[:, 0]     # (B,)
        country_id = cat_last[:, 1]     # (B,)
        weather_id = cat_last[:, 2]     # (B,)

        month_val = num_last[:, _MONTH_IDX].long().clamp(0, 12)
        hour_val  = num_last[:, _HOUR_IDX].long().clamp(0, 23)
        season_val = season_last.squeeze(1).clamp(0, 3)   # (B,)
        dayperiod_val = dp_last.squeeze(1).clamp(0, 4)     # (B,)

        # ===== Deep 侧 =====
        emb_city    = self.embed_city(city_id)
        emb_country = self.embed_country(country_id)
        emb_weather = self.embed_weather(weather_id)
        emb_month   = self.embed_month(month_val)
        emb_hour    = self.embed_hour(hour_val)
        emb_season  = self.embed_season(season_val)
        emb_dayperiod = self.embed_dayperiod(dayperiod_val)

        deep_feats = torch.cat([
            num_last,           # 35
            cyc_last,           # 8
            bin_last,           # 15
            emb_city,           # 8
            emb_country,        # 6
            emb_weather,        # 4
            emb_month,          # 4
            emb_hour,           # 6
            emb_season,         # 2
            emb_dayperiod,      # 3
        ], dim=1)

        deep_out = self.deep_dnn(deep_feats)  # (B, 128)

        # ===== Wide 侧 =====
        wide_feats = build_wide_features(batch)  # (B, 21)
        wide_out = F.relu(self.wide_bn(self.wide_linear(wide_feats)))
        wide_out = self.wide_dropout(wide_out)   # (B, 64)

        # ===== 融合 =====
        combined = torch.cat([deep_out, wide_out], dim=1)
        pred = self.final_fc(combined)  # (B, num_targets)

        return pred


# ==================== 模型工厂 ====================
def create_model(device: torch.device = None) -> WideAndDeep:
    """
    创建 Wide & Deep 模型
    
    Args:
        device: 计算设备
    
    Returns:
        模型实例
    """
    model = WideAndDeep(
        num_targets=5,
        deep_hidden_dims=[512, 256, 128],
        dropout=0.3,
    )
    if device is not None:
        model = model.to(device)
    return model


def count_parameters(model: nn.Module) -> int:
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
