"""
概率转换器：将 Model 3 的回归输出转换为分类概率
"""
import numpy as np
from typing import Dict, Any
import warnings


class ProbabilityConverter:
    """
    将回归值转换为分类概率
    
    支持多种转换方法：
    1. threshold_based: 基于阈值的映射
    2. sigmoid: Sigmoid 函数转换
    3. composite: 综合多个指标
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化概率转换器
        
        Args:
            config: 转换配置字典
        """
        self.config = config
    
    def convert_rain_probability(self, rain_sum: np.ndarray) -> np.ndarray:
        """
        将降雨量转换为下雨概率
        
        Args:
            rain_sum: 降雨量数组 (N,)
            
        Returns:
            概率数组 (N,) [0, 1]
        """
        method = self.config['rain']['method']
        
        if method == 'threshold_based':
            threshold = self.config['rain']['threshold']
            scale = self.config['rain']['scale']
            
            # 基于阈值的分段映射
            # rain_sum <= 0: 概率接近0
            # 0 < rain_sum < threshold: 线性插值 [0, 0.5]
            # rain_sum >= threshold: 线性插值 [0.5, 1.0]
            
            probabilities = np.zeros_like(rain_sum, dtype=np.float32)
            
            # 无雨或微量雨
            mask_no_rain = rain_sum <= 0
            probabilities[mask_no_rain] = 0.0
            
            # 小雨 (0 < rain <= threshold)
            mask_light = (rain_sum > 0) & (rain_sum < threshold)
            probabilities[mask_light] = 0.5 * (rain_sum[mask_light] / threshold)
            
            # 中雨及以上 (rain >= threshold)
            mask_heavy = rain_sum >= threshold
            probabilities[mask_heavy] = 0.5 + 0.5 * np.clip(
                rain_sum[mask_heavy] / scale, 0, 1
            )
            
        elif method == 'sigmoid':
            scale = self.config['rain']['scale']
            # Sigmoid 转换: 1 / (1 + exp(-x/scale))
            probabilities = 1.0 / (1.0 + np.exp(-rain_sum / scale))
            
        else:
            raise ValueError(f"Unknown conversion method: {method}")
        
        # 确保概率在 [0, 1] 范围内
        probabilities = np.clip(probabilities, 0.0, 1.0)
        
        return probabilities
    
    def convert_snow_probability(self, snow_sum: np.ndarray) -> np.ndarray:
        """
        将降雪量转换为下雪概率
        
        Args:
            snow_sum: 降雪量数组 (N,)
            
        Returns:
            概率数组 (N,) [0, 1]
        """
        method = self.config['snow']['method']
        
        if method == 'threshold_based':
            threshold = self.config['snow']['threshold']
            scale = self.config['snow']['scale']
            
            probabilities = np.zeros_like(snow_sum, dtype=np.float32)
            
            # 无雪
            mask_no_snow = snow_sum <= 0
            probabilities[mask_no_snow] = 0.0
            
            # 小雪
            mask_light = (snow_sum > 0) & (snow_sum < threshold)
            probabilities[mask_light] = 0.5 * (snow_sum[mask_light] / threshold)
            
            # 中雪及以上
            mask_heavy = snow_sum >= threshold
            probabilities[mask_heavy] = 0.5 + 0.5 * np.clip(
                snow_sum[mask_heavy] / scale, 0, 1
            )
            
        elif method == 'sigmoid':
            scale = self.config['snow']['scale']
            probabilities = 1.0 / (1.0 + np.exp(-snow_sum / scale))
            
        else:
            raise ValueError(f"Unknown conversion method: {method}")
        
        probabilities = np.clip(probabilities, 0.0, 1.0)
        
        return probabilities
    
    def convert_severe_probability(
        self,
        temp_range: np.ndarray,
        wind_speed: np.ndarray,
        precipitation: np.ndarray
    ) -> np.ndarray:
        """
        基于多个指标综合判断恶劣天气概率
        
        Args:
            temp_range: 温度范围 (N,)
            wind_speed: 风速 (N,)
            precipitation: 降水量 (N,)
            
        Returns:
            概率数组 (N,) [0, 1]
        """
        method = self.config['severe']['method']
        
        if method != 'composite':
            raise ValueError(f"Severe weather only supports 'composite' method")
        
        thresholds = self.config['severe']['thresholds']
        weights = self.config['severe']['weights']
        
        # 计算每个指标的得分 [0, 1]
        temp_score = self._calculate_indicator_score(
            temp_range, thresholds['temp_range']
        )
        wind_score = self._calculate_indicator_score(
            wind_speed, thresholds['wind_speed']
        )
        precip_score = self._calculate_indicator_score(
            precipitation, thresholds['precipitation']
        )
        
        # 加权组合
        severity_score = (
            weights['temp_range'] * temp_score +
            weights['wind_speed'] * wind_score +
            weights['precipitation'] * precip_score
        )
        
        # 确保在 [0, 1] 范围
        probabilities = np.clip(severity_score, 0.0, 1.0)
        
        return probabilities
    
    def _calculate_indicator_score(
        self,
        values: np.ndarray,
        threshold: float
    ) -> np.ndarray:
        """
        计算单个指标的得分
        
        Args:
            values: 指标值数组
            threshold: 阈值
            
        Returns:
            得分数组 [0, 1]
        """
        # 线性映射：value / (2 * threshold)
        # threshold 对应得分 0.5，2*threshold 对应得分 1.0
        scores = values / (2.0 * threshold)
        scores = np.clip(scores, 0.0, 1.0)
        
        return scores
    
    def convert_all(
        self,
        model3_outputs: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        转换所有分类任务的概率
        
        Args:
            model3_outputs: Model 3 的输出 (N, 9)
                索引: [T_max, T_min, T_mean, T_range, T_feels, 
                       Precip, Rain, Snow, Wind]
            
        Returns:
            字典: {'rain': prob_array, 'snow': prob_array, 'severe': prob_array}
        """
        if model3_outputs.ndim != 2 or model3_outputs.shape[1] != 9:
            raise ValueError(
                f"Expected shape (N, 9), got {model3_outputs.shape}"
            )
        
        # 提取各个指标
        rain_sum = model3_outputs[:, 6]      # index 6
        snow_sum = model3_outputs[:, 7]      # index 7
        temp_range = model3_outputs[:, 3]    # index 3
        wind_speed = model3_outputs[:, 8]    # index 8
        precipitation = model3_outputs[:, 5] # index 5
        
        # 转换概率
        probabilities = {
            'rain': self.convert_rain_probability(rain_sum),
            'snow': self.convert_snow_probability(snow_sum),
            'severe': self.convert_severe_probability(
                temp_range, wind_speed, precipitation
            )
        }
        
        return probabilities


def test_probability_converter():
    """测试概率转换器"""
    from config import PROBABILITY_CONVERSION_CONFIG
    
    converter = ProbabilityConverter(PROBABILITY_CONVERSION_CONFIG)
    
    # 测试数据
    rain_sum = np.array([0.0, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0])
    snow_sum = np.array([0.0, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0])
    
    print("Rain sum:", rain_sum)
    print("Rain prob:", converter.convert_rain_probability(rain_sum))
    print()
    print("Snow sum:", snow_sum)
    print("Snow prob:", converter.convert_snow_probability(snow_sum))
    print()
    
    # 测试恶劣天气
    temp_range = np.array([5.0, 10.0, 15.0, 20.0, 25.0])
    wind_speed = np.array([5.0, 8.0, 10.0, 15.0, 20.0])
    precipitation = np.array([0.0, 2.0, 5.0, 10.0, 15.0])
    
    severe_prob = converter.convert_severe_probability(
        temp_range, wind_speed, precipitation
    )
    print("Severe probability:", severe_prob)


if __name__ == "__main__":
    test_probability_converter()
