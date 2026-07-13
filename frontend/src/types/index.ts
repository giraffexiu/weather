/** 天气预测 API 类型定义 */

// ==================== 请求类型 ====================

export interface PredictRequest {
  city: string
  target_time?: string
}

// ==================== 响应类型 ====================

export interface HourlyPrediction {
  time: string
  temperature: number
  apparent_temperature: number
  humidity: number
  precipitation: number
  wind_speed: number
  rain_probability: number
  weather: string
}

export interface DailyPrediction {
  date: string
  temperature_max: number
  temperature_min: number
  temperature_mean: number
  apparent_temperature: number
  precipitation: number
  rain_amount: number
  snow_amount: number
  wind_speed: number
  humidity: number
  weather: string
  rain_probability: number
  snow_probability: number
  severe_probability: number
}

export interface FeatureImportance {
  feature: string
  importance: number
}

export interface ModelExplanation {
  feature_importance: FeatureImportance[]
  model_confidence: number
  weather_summary: string
  contributing_factors: string[]
  model_weights: Record<string, number | Record<string, number>>
}

export interface PredictionResponse {
  city: string
  country: string
  latitude: number
  longitude: number
  target_time: string
  current: HourlyPrediction
  daily: DailyPrediction[]
  hourly: HourlyPrediction[]
  explanation: ModelExplanation
  confidence: number
}

export interface CityListResponse {
  cities: string[]
  count: number
}
