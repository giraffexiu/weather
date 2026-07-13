"""
概率转换器：将小时级回归输出转换为分类概率
"""
import numpy as np
from typing import Dict, Any


class ProbabilityConverter:
    """将回归值转换为分类概率"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def convert_rain_probability(self, precipitation: np.ndarray) -> np.ndarray:
        """降水量 → 下雨概率"""
        cfg = self.config['rain']
        threshold, scale = cfg['threshold'], cfg['scale']

        if cfg.get('method') == 'sigmoid':
            probs = 1.0 / (1.0 + np.exp(-precipitation / scale))
        else:
            # threshold_based
            probs = np.zeros_like(precipitation, dtype=np.float32)
            mask_light = (precipitation > 0) & (precipitation < threshold)
            probs[mask_light] = 0.5 * (precipitation[mask_light] / threshold)
            mask_heavy = precipitation >= threshold
            probs[mask_heavy] = 0.5 + 0.5 * np.clip(precipitation[mask_heavy] / scale, 0, 1)

        return np.clip(probs, 0.0, 1.0)

    def convert_freezing_probability(self, temperature: np.ndarray) -> np.ndarray:
        """温度 → 结冰概率"""
        cfg = self.config['freezing']
        scale = cfg['scale']
        # 温度越低概率越高
        probs = 1.0 / (1.0 + np.exp(temperature / scale))
        return np.clip(probs, 0.0, 1.0)

    def convert_windy_probability(self, wind_speed: np.ndarray) -> np.ndarray:
        """风速 → 大风概率"""
        cfg = self.config['windy']
        threshold, scale = cfg['threshold'], cfg['scale']

        if cfg.get('method') == 'sigmoid':
            probs = 1.0 / (1.0 + np.exp(-(wind_speed - threshold) / scale))
        else:
            probs = np.zeros_like(wind_speed, dtype=np.float32)
            mask_light = (wind_speed > 0) & (wind_speed < threshold)
            probs[mask_light] = 0.3 * (wind_speed[mask_light] / threshold)
            mask_heavy = wind_speed >= threshold
            probs[mask_heavy] = 0.3 + 0.7 * np.clip((wind_speed[mask_heavy] - threshold) / scale, 0, 1)

        return np.clip(probs, 0.0, 1.0)

    def convert_all(self, model_outputs: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
        """
        转换所有分类概率

        Args:
            model_outputs: (N, 5) [temperature_2m, precipitation, wind_speed_10m,
                                   apparent_temperature, relative_humidity_2m]

        Returns:
            {'rain': {'probability': array}, 'freezing': {...}, 'windy': {...}}
        """
        assert model_outputs.ndim == 2 and model_outputs.shape[1] == 5, \
            f"Expected (N, 5), got {model_outputs.shape}"

        temp = model_outputs[:, 0]       # temperature_2m
        precip = model_outputs[:, 1]     # precipitation
        wind = model_outputs[:, 2]       # wind_speed_10m

        return {
            'rain':     {'probability': self.convert_rain_probability(precip)},
            'freezing': {'probability': self.convert_freezing_probability(temp)},
            'windy':    {'probability': self.convert_windy_probability(wind)},
        }


if __name__ == "__main__":
    from config import PROBABILITY_CONVERSION_CONFIG

    converter = ProbabilityConverter(PROBABILITY_CONVERSION_CONFIG)

    precips = np.array([0.0, 0.05, 0.1, 0.5, 1.0, 5.0])
    temps = np.array([5.0, 0.0, -5.0, -10.0, 10.0, 20.0])
    winds = np.array([0.0, 3.0, 5.0, 10.0, 15.0, 25.0])

    print("降水 → 下雨概率:", dict(zip(precips, converter.convert_rain_probability(precips))))
    print("温度 → 结冰概率:", dict(zip(temps, converter.convert_freezing_probability(temps))))
    print("风速 → 大风概率:", dict(zip(winds, converter.convert_windy_probability(winds))))
