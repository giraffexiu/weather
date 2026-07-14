<template>
  <div class="dashboard">
    <!-- 位置信息 -->
    <div class="location-bar">
      <span class="location-icon">&#127758;</span>
      <span class="location-name">{{ data.city }}, {{ data.country }}</span>
      <span class="location-coords">{{ data.latitude.toFixed(2) }}&deg;N, {{ data.longitude.toFixed(2) }}&deg;E</span>
      <el-tag size="small" type="info" effect="dark" class="time-tag">
        {{ data.target_time }}
      </el-tag>
    </div>

    <!-- 当前天气 + 核心卡片 -->
    <el-row :gutter="20" class="stats-row">
      <el-col :span="6">
        <el-card shadow="hover" class="stat-card stat-temp">
          <div class="stat-label">温度</div>
          <div class="stat-value">
            {{ data.current.temperature.toFixed(1) }}&deg;C
          </div>
          <div class="stat-sub">
            体感 {{ data.current.apparent_temperature.toFixed(1) }}&deg;C
          </div>
          <div class="stat-weather">{{ data.current.weather }}</div>
        </el-card>
      </el-col>

      <el-col :span="6">
        <el-card shadow="hover" class="stat-card stat-rain">
          <div class="stat-label">降雨概率</div>
          <div class="stat-value">
            {{ (data.current.rain_probability * 100).toFixed(0) }}%
          </div>
          <div class="stat-sub">
            降水量 {{ data.current.precipitation.toFixed(1) }} mm
          </div>
          <el-progress
            :percentage="data.current.rain_probability * 100"
            :color="rainColor"
            :show-text="false"
            :stroke-width="6"
          />
        </el-card>
      </el-col>

      <el-col :span="6">
        <el-card shadow="hover" class="stat-card stat-wind">
          <div class="stat-label">湿度 & 风速</div>
          <div class="stat-value-row">
            <div class="stat-half">
              <span class="half-label">湿度</span>
              <span class="half-value">{{ data.current.humidity.toFixed(0) }}%</span>
            </div>
            <div class="stat-half">
              <span class="half-label">风速</span>
              <span class="half-value">{{ data.current.wind_speed.toFixed(1) }} m/s</span>
            </div>
          </div>
        </el-card>
      </el-col>

      <el-col :span="6">
        <el-card shadow="hover" class="stat-card stat-confidence">
          <div class="stat-label">预测置信度</div>
          <div class="stat-value confidence-value">
            {{ (data.confidence * 100).toFixed(0) }}%
          </div>
          <el-progress
            :percentage="data.confidence * 100"
            :color="confidenceColor"
            :show-text="false"
            :stroke-width="8"
          />
          <div class="stat-sub confidence-sub">
            模型 {{ data.explanation.model_confidence > 0.85 ? '高' : '中' }}置信度
          </div>
        </el-card>
      </el-col>
    </el-row>

    <!-- 7天总览 -->
    <el-card shadow="never" class="section-card">
      <template #header>
        <span class="section-title">&#128198; 未来 7 天概览</span>
      </template>
      <div class="daily-strip">
        <div
          v-for="(day, idx) in data.daily"
          :key="idx"
          class="daily-strip-item"
          :class="{ today: idx === 0 }"
        >
          <div class="strip-date">{{ formatDate(day.date) }}</div>
          <div class="strip-icon">{{ weatherIcon(day.weather) }}</div>
          <div class="strip-temp">
            <span class="hi">{{ day.temperature_max.toFixed(0) }}&deg;</span>
            <span class="lo">{{ day.temperature_min.toFixed(0) }}&deg;</span>
          </div>
          <div class="strip-rain">
            &#9748; {{ (day.rain_probability * 100).toFixed(0) }}%
          </div>
        </div>
      </div>
    </el-card>

    <!-- 特征重要性 -->
    <el-card shadow="never" class="section-card">
      <template #header>
        <span class="section-title">&#128202; 特征重要性</span>
      </template>
      <div class="feature-bars">
        <div
          v-for="fi in data.explanation.feature_importance"
          :key="fi.feature"
          class="feature-bar-item"
        >
          <span class="feature-name">{{ fi.feature }}</span>
          <div class="feature-bar-track">
            <div
              class="feature-bar-fill"
              :style="{ width: (fi.importance * 100) + '%' }"
            />
          </div>
          <span class="feature-pct">{{ (fi.importance * 100).toFixed(0) }}%</span>
        </div>
      </div>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { PredictionResponse } from '../types'

interface Props {
  data: PredictionResponse
  loading: boolean
}

const props = defineProps<Props>()

const DAY_NAMES = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']

const rainColor = computed(() => {
  const p = props.data.current.rain_probability
  if (p > 0.6) return '#ef4444'
  if (p > 0.3) return '#f59e0b'
  return '#22c55e'
})

const confidenceColor = computed(() => {
  const c = props.data.confidence
  if (c > 0.85) return '#22c55e'
  if (c > 0.7) return '#f59e0b'
  return '#ef4444'
})

function formatDate(dateStr: string): string {
  const d = new Date(dateStr)
  return `${d.getMonth() + 1}/${d.getDate()} ${DAY_NAMES[d.getDay()]}`
}

function weatherIcon(weather: string): string {
  const map: Record<string, string> = {
    'Sunny': '\u2600\uFE0F',
    'Clear': '\uD83C\uDF19',
    'Cloudy': '\u2601\uFE0F',
    'Partly Cloudy': '\u26C5',
    'Overcast': '\uD83C\uDF25\uFE0F',
    'Rain': '\uD83C\uDF27\uFE0F',
    'Light Rain': '\uD83C\uDF26\uFE0F',
    'Moderate Rain': '\uD83C\uDF27\uFE0F',
    'Heavy Rain': '\u26C8\uFE0F',
    'Drizzle': '\uD83C\uDF26\uFE0F',
    'Thunderstorm': '\u26C8\uFE0F',
    'Snow': '\u2744\uFE0F',
    'Heavy Snow': '\u2744\uFE0F',
    'Sleet': '\uD83C\uDF28\uFE0F',
    'Fog': '\uD83C\uDF2B\uFE0F',
    'Mist': '\uD83C\uDF2B\uFE0F',
    'Freezing': '\uD83E\uDD76',
    'Scorching': '\uD83E\uDD75',
  }
  return map[weather] || '\u2601\uFE0F'
}
</script>

<style scoped>
.dashboard {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.location-bar {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 20px;
  background: rgba(30, 41, 59, 0.6);
  border-radius: 10px;
  border: 1px solid rgba(100, 140, 255, 0.12);
}

.location-icon { font-size: 20px; }

.location-name {
  font-size: 18px;
  font-weight: 700;
  color: #e2e8f0;
}

.location-coords {
  font-size: 13px;
  color: #64748b;
}

.time-tag { margin-left: auto; }

/* Stats Cards */
.stats-row { margin-bottom: 0; }

.stat-card {
  background: rgba(30, 41, 59, 0.7) !important;
  border: 1px solid rgba(100, 140, 255, 0.12) !important;
  border-radius: 12px !important;
  height: 180px;
  display: flex;
  flex-direction: column;
  justify-content: center;
}

.stat-card :deep(.el-card__body) {
  padding: 20px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.stat-label {
  font-size: 12px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 1px;
  color: #64748b;
}

.stat-value {
  font-size: 36px;
  font-weight: 700;
  color: #f1f5f9;
  line-height: 1;
}

.stat-temp .stat-value { color: #fb923c; }
.stat-rain .stat-value { color: #60a5fa; }
.stat-wind .stat-value { color: #a78bfa; }
.confidence-value { color: #22c55e; }

.stat-sub {
  font-size: 13px;
  color: #94a3b8;
}

.stat-weather {
  font-size: 14px;
  font-weight: 600;
  color: #f59e0b;
}

.stat-value-row {
  display: flex;
  gap: 20px;
  margin-top: 8px;
}

.stat-half {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.half-label {
  font-size: 12px;
  color: #64748b;
  text-transform: uppercase;
}

.half-value {
  font-size: 28px;
  font-weight: 700;
  color: #e2e8f0;
}

.confidence-sub {
  margin-top: 4px;
}

/* Section Card */
.section-card {
  background: rgba(30, 41, 59, 0.6) !important;
  border: 1px solid rgba(100, 140, 255, 0.12) !important;
  border-radius: 12px !important;
}

.section-title {
  font-size: 16px;
  font-weight: 700;
  color: #e2e8f0;
}

/* Daily Strip */
.daily-strip {
  display: flex;
  gap: 8px;
  overflow-x: auto;
  padding: 4px 0;
}

.daily-strip-item {
  flex: 1;
  min-width: 100px;
  text-align: center;
  padding: 12px 8px;
  border-radius: 10px;
  background: rgba(15, 23, 42, 0.4);
  transition: all 0.3s;
  cursor: default;
}

.daily-strip-item.today {
  background: rgba(59, 130, 246, 0.15);
  border: 1px solid rgba(59, 130, 246, 0.3);
}

.daily-strip-item:hover {
  background: rgba(59, 130, 246, 0.1);
}

.strip-date {
  font-size: 13px;
  color: #94a3b8;
  font-weight: 600;
}

.strip-icon { font-size: 28px; margin: 6px 0; }

.strip-temp {
  display: flex;
  justify-content: center;
  gap: 8px;
}

.hi { font-weight: 700; color: #fb923c; }
.lo { color: #94a3b8; }

.strip-rain {
  font-size: 12px;
  color: #60a5fa;
  margin-top: 4px;
}

/* Feature Importance Bars */
.feature-bars {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.feature-bar-item {
  display: flex;
  align-items: center;
  gap: 12px;
}

.feature-name {
  width: 180px;
  font-size: 13px;
  color: #cbd5e1;
  text-align: right;
  flex-shrink: 0;
}

.feature-bar-track {
  flex: 1;
  height: 10px;
  background: rgba(15, 23, 42, 0.6);
  border-radius: 5px;
  overflow: hidden;
}

.feature-bar-fill {
  height: 100%;
  background: linear-gradient(90deg, #3b82f6, #60a5fa);
  border-radius: 5px;
  transition: width 0.8s ease;
}

.feature-pct {
  width: 40px;
  font-size: 13px;
  font-weight: 700;
  color: #60a5fa;
}
</style>
