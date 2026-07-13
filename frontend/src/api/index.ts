/** API 请求封装 */
import axios from 'axios'
import type { PredictRequest, PredictionResponse, CityListResponse } from '../types'

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE || '',
  timeout: 30000,
  headers: { 'Content-Type': 'application/json' },
})

export async function fetchPrediction(data: PredictRequest): Promise<PredictionResponse> {
  const response = await api.post<PredictionResponse>('/api/predict', data)
  return response.data
}

export async function fetchCities(): Promise<CityListResponse> {
  const response = await api.get<CityListResponse>('/api/cities')
  return response.data
}

export async function checkHealth(): Promise<{ status: string }> {
  const response = await api.get('/api/health')
  return response.data
}
