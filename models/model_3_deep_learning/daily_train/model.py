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
        
        # Embedding层
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
            wide_input: (B, 15)
            deep_input: (B, 123)
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
        cat_embeddings = torch.cat([city_emb, country_emb, season_emb], dim=1)  # (B, 20)
        
        # === 2. 数值特征聚合 ===
        numerical = batch['numerical']  # (B, 7, 22)
        
        num_mean = numerical.mean(dim=1)      # (B, 22) 趋势
        num_std = numerical.std(dim=1)        # (B, 22) 波动
        num_last = numerical[:, -1, :]        # (B, 22) 当前
        num_first = numerical[:, 0, :]        # (B, 22) 初始
        num_diff = num_last - num_first       # (B, 22) 变化
        
        num_features = torch.cat([num_mean, num_std, num_last, num_diff], dim=1)  # (B, 88)
        
        # === 3. 周期和二值特征 ===
        cyc_mean = batch['cyclical'].mean(dim=1)  # (B, 6)
        bin_mean = batch['binary'].mean(dim=1)    # (B, 9)
        
        # === 4. Wide侧特征 ===
        wide_input = self._create_wide_features(numerical, season_id, batch['binary'])
        
        # === 5. Deep侧特征 ===
        deep_input = torch.cat([cat_embeddings, num_features, cyc_mean, bin_mean], dim=1)
        
        return wide_input, deep_input
    
    def _create_wide_features(self, numerical, season_id, binary):
        """构造Wide侧交叉特征"""
        B = numerical.size(0)
        idx = self.num_idx
        bin_idx = self.bin_idx
        
        last_day = numerical[:, -1, :]
        
        # 关键特征
        temp_mean = last_day[:, idx['temperature_2m_mean']]
        precip = last_day[:, idx['precipitation_sum']]
        wind = last_day[:, idx['wind_speed_10m_max']]
        lat = last_day[:, idx['latitude']]
        month = last_day[:, idx['month']]
        
        # 7天统计
        temp_7d_mean = numerical[:, :, idx['temperature_2m_mean']].mean(dim=1)
        temp_7d_std = numerical[:, :, idx['temperature_2m_mean']].std(dim=1)
        precip_7d_sum = numerical[:, :, idx['precipitation_sum']].sum(dim=1)
        
        # 交叉特征
        cross_month_precip = month * precip
        cross_lat_temp = lat * temp_mean
        
        season_onehot = F.one_hot(season_id, 4).float()
        cross_season_wind = season_onehot * wind.unsqueeze(1)  # (B, 4)
        
        is_rainy = binary[:, -1, bin_idx['is_rainy']]
        cross_rain_precip = is_rainy * precip
        
        wide_features = torch.cat([
            temp_mean.unsqueeze(1),       # 1
            precip.unsqueeze(1),          # 2
            wind.unsqueeze(1),            # 3
            lat.unsqueeze(1),             # 4
            temp_7d_mean.unsqueeze(1),    # 5
            temp_7d_std.unsqueeze(1),     # 6
            precip_7d_sum.unsqueeze(1),   # 7
            cross_month_precip.unsqueeze(1),  # 8
            cross_lat_temp.unsqueeze(1),      # 9
            cross_season_wind,            # 10-13 (4维)
            cross_rain_precip.unsqueeze(1),   # 14
        ], dim=1)  # 总计14维
        
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
