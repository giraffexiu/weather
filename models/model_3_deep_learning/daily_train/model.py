"""
Wide & Deep 天气预测模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# 特征索引直接定义在这里，避免导入冲突
NUMERICAL_FEATURE_INDEX = {
    'latitude': 0, 'longitude': 1,
    'temperature_2m_max': 2, 'temperature_2m_min': 3,
    'temperature_2m_mean': 4, 'temperature_range': 5,
    'feels_like_temperature': 6, 'precipitation_sum': 7,
    'rain_sum': 8, 'snow_sum': 9,
    'wind_speed_10m_max': 10, 'shortwave_radiation_sum': 11,
    'month': 12, 'year': 13, 'day': 14,
    'day_of_year': 15, 'day_of_week': 16,
    'quarter': 17, 'week_of_year': 18,
}

BINARY_FEATURE_INDEX = {
    'is_freezing': 0, 'is_hot_day': 1, 'is_rainy': 2,
    'is_heavy_rain': 3, 'is_snowy': 4, 'is_windy': 5,
    'is_sunny': 6, 'is_severe_weather': 7, 'is_high_latitude': 8,
}


class FeaturePreprocessor(nn.Module):
    """特征预处理：将时序batch转换为Wide和Deep的输入"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Embedding层（减小维度）
        self.city_embed = nn.Embedding(49, config['city_embed_dim'])
        self.country_embed = nn.Embedding(29, config['country_embed_dim'])
        self.season_embed = nn.Embedding(4, config['season_embed_dim'])
        
        self.num_idx = NUMERICAL_FEATURE_INDEX
        self.bin_idx = BINARY_FEATURE_INDEX
    
    def forward(self, batch):
        """
        Args:
            batch: {
                'categorical': (B, 7, 2),
                'numerical': (B, 7, 22),
                'cyclical': (B, 7, 6),
                'binary': (B, 7, 9),
                'season': (B, 7, 1)
            }
        
        Returns:
            wide_input: (B, 25)
            deep_input: (B, deep_dim)
        """
        B = batch['categorical'].size(0)
        
        # === 1. 类别特征 ===
        # 按城市分组，7天内city/country不变
        city_id = batch['categorical'][:, 0, 0].long()
        country_id = batch['categorical'][:, 0, 1].long()
        season_id = batch['season'][:, -1, 0].long()
        
        city_emb = self.city_embed(city_id)
        country_emb = self.country_embed(country_id)
        season_emb = self.season_embed(season_id)
        cat_embeddings = torch.cat([city_emb, country_emb, season_emb], dim=1)  # (B, 15)
        
        # === 2. 数值特征聚合（增强时序特征）===
        numerical = batch['numerical']  # (B, 7, 22)
        
        num_mean = numerical.mean(dim=1)      # (B, 22) 7天平均趋势
        num_std = numerical.std(dim=1)        # (B, 22) 7天波动性
        num_last = numerical[:, -1, :]        # (B, 22) 最近一天（Day 7）
        num_first = numerical[:, 0, :]        # (B, 22) 7天前（Day 1）
        num_diff = num_last - num_first       # (B, 22) 7天变化
        
        num_features = torch.cat([num_mean, num_std, num_last, num_diff], dim=1)  # (B, 88)
        
        # === 3. 周期和二值特征 ===
        cyc_mean = batch['cyclical'].mean(dim=1)  # (B, 6)
        bin_mean = batch['binary'].mean(dim=1)    # (B, 9)
        
        # === 4. Wide侧特征（增强滞后特征）===
        wide_input = self._create_wide_features(numerical, season_id, batch['binary'])
        
        # === 5. Deep侧特征 ===
        deep_input = torch.cat([cat_embeddings, num_features, cyc_mean, bin_mean], dim=1)
        
        return wide_input, deep_input
    
    def _create_wide_features(self, numerical, season_id, binary):
        """构造Wide侧交叉特征（增强版）"""
        B = numerical.size(0)
        idx = self.num_idx
        bin_idx = self.bin_idx
        
        # 提取关键时序数据
        temp_seq = numerical[:, :, idx['temperature_2m_mean']]  # (B, 7)
        precip_seq = numerical[:, :, idx['precipitation_sum']]  # (B, 7)
        wind_seq = numerical[:, :, idx['wind_speed_10m_max']]   # (B, 7)
        
        # 滞后特征（多个时间点）
        temp_lag1 = temp_seq[:, -1]  # Day 7
        temp_lag2 = temp_seq[:, -2]  # Day 6
        temp_lag7 = temp_seq[:, 0]   # Day 1
        
        precip_lag1 = precip_seq[:, -1]
        precip_lag2 = precip_seq[:, -2]
        
        wind_lag1 = wind_seq[:, -1]
        
        # 滚动统计特征
        temp_7d_mean = temp_seq.mean(dim=1)
        temp_7d_std = temp_seq.std(dim=1)
        temp_3d_mean = temp_seq[:, -3:].mean(dim=1)  # 最近3天平均
        
        precip_7d_sum = precip_seq.sum(dim=1)
        precip_3d_sum = precip_seq[:, -3:].sum(dim=1)
        
        # 变化率特征
        temp_diff_1day = temp_lag1 - temp_lag2
        temp_diff_7day = temp_lag1 - temp_lag7
        
        # 地理和时间特征
        last_day = numerical[:, -1, :]
        lat = last_day[:, idx['latitude']]
        lon = last_day[:, idx['longitude']]
        month = last_day[:, idx['month']]
        
        # 交叉特征
        cross_month_precip = month * precip_lag1
        cross_lat_temp = lat * temp_lag1
        
        season_onehot = F.one_hot(season_id, 4).float()
        cross_season_wind = season_onehot * wind_lag1.unsqueeze(1)  # (B, 4)
        
        is_rainy = binary[:, -1, bin_idx['is_rainy']]
        cross_rain_precip = is_rainy * precip_lag1
        
        # 组合所有Wide特征
        wide_features = torch.cat([
            # 滞后特征 (7个) - 增加lon
            temp_lag1.unsqueeze(1),      # 1
            temp_lag2.unsqueeze(1),      # 2
            precip_lag1.unsqueeze(1),    # 3
            precip_lag2.unsqueeze(1),    # 4
            wind_lag1.unsqueeze(1),      # 5
            lat.unsqueeze(1),            # 6
            lon.unsqueeze(1),            # 7
            
            # 滚动统计 (5个)
            temp_7d_mean.unsqueeze(1),   # 8
            temp_7d_std.unsqueeze(1),    # 9
            temp_3d_mean.unsqueeze(1),   # 10
            precip_7d_sum.unsqueeze(1),  # 11
            precip_3d_sum.unsqueeze(1),  # 12
            
            # 变化率 (2个)
            temp_diff_1day.unsqueeze(1), # 13
            temp_diff_7day.unsqueeze(1), # 14
            
            # 交叉特征 (8个)
            cross_month_precip.unsqueeze(1),  # 15
            cross_lat_temp.unsqueeze(1),      # 16
            cross_season_wind,                # 17-20 (4维)
            cross_rain_precip.unsqueeze(1),   # 21
            month.unsqueeze(1),               # 22 - 添加month本身
        ], dim=1)  # 总计22维
        
        return wide_features


class WideNetwork(nn.Module):
    """Wide侧：线性记忆"""
    
    def __init__(self, wide_dim=14, num_targets=1):
        super().__init__()
        self.linear = nn.Linear(wide_dim, num_targets, bias=True)
    
    def forward(self, x):
        return self.linear(x)


class DeepNetwork(nn.Module):
    """Deep侧：非线性泛化"""
    
    def __init__(self, deep_dim=123, hidden_dims=[128, 64, 32], dropout=0.3, use_batch_norm=True, num_targets=1):
        super().__init__()
        
        layers = []
        in_dim = deep_dim
        
        for i, h_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(in_dim, h_dim))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            
            layers.append(nn.ReLU())
            
            # 最后一层dropout降低
            drop_rate = dropout if i < len(hidden_dims) - 1 else dropout * 0.67
            layers.append(nn.Dropout(drop_rate))
            
            in_dim = h_dim
        
        layers.append(nn.Linear(in_dim, num_targets))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class WideDeepModel(nn.Module):
    """Wide & Deep 多目标预测模型"""
    
    def __init__(self, config):
        super().__init__()
        
        self.num_targets = config.get('num_targets', 1)
        
        self.preprocessor = FeaturePreprocessor(config)
        
        self.wide = WideNetwork(
            wide_dim=config['wide_dim'],
            num_targets=self.num_targets
        )
        
        self.deep = DeepNetwork(
            deep_dim=config['deep_dim'],
            hidden_dims=config['hidden_dims'],
            dropout=config['dropout'],
            use_batch_norm=config['use_batch_norm'],
            num_targets=self.num_targets
        )
    
    def forward(self, batch):
        """
        Args:
            batch: DataLoader返回的字典
        
        Returns:
            predictions: (B, num_targets) 多目标预测值
        """
        wide_input, deep_input = self.preprocessor(batch)
        
        wide_out = self.wide(wide_input)   # (B, num_targets)
        deep_out = self.deep(deep_input)   # (B, num_targets)
        
        output = wide_out + deep_out        # (B, num_targets)
        
        return output
