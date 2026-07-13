<template>
  <div class="daily-page">
    <el-card shadow="never" class="section-card">
      <template #header>
        <span class="section-title">&#128198; 未来 7 天详细预测 - {{ data.city }}</span>
      </template>

      <el-table
        :data="data.daily"
        stripe
        class="daily-table"
        :header-cell-style="tableHeaderStyle"
        :cell-style="tableCellStyle"
      >
        <el-table-column prop="date" label="日期" width="120">
          <template #default="{ row }">
            <span class="date-cell">{{ formatDate(row.date) }}</span>
          </template>
        </el-table-column>

        <el-table-column label="天气" width="90">
          <template #default="{ row }">
            <span class="weather-icon-cell">{{ weatherIcon(row.weather) }}</span>
          </template>
        </el-table-column>

        <el-table-column prop="weather" label="天气状况" width="110">
          <template #default="{ row }">
            <el-tag
              :type="weatherTagType(row.weather)"
              size="small"
              effect="dark"
            >{{ row.weather }}</el-tag>
          </template>
        </el-table-column>

        <el-table-column label="最高温" width="90" sortable prop="temperature_max">
          <template #default="{ row }">
            <span class="temp-hi">{{ row.temperature_max.toFixed(1) }}&deg;</span>
          </template>
        </el-table-column>

        <el-table-column label="最低温" width="90" sortable prop="temperature_min">
          <template #default="{ row }">
            <span class="temp-lo">{{ row.temperature_min.toFixed(1) }}&deg;</span>
          </template>
        </el-table-column>

        <el-table-column label="降水量(mm)" width="110" sortable prop="precipitation">
          <template #default="{ row }">
            <span :class="{ 'rain-value': row.precipitation > 1 }">
              {{ row.precipitation.toFixed(1) }}
            </span>
          </template>
        </el-table-column>

        <el-table-column label="降雨概率" width="100">
          <template #default="{ row }">
            <div class="prob-cell">
              <el-progress
                :percentage="+(row.rain_probability * 100).toFixed(0)"
                :color="rainProgressColor(row.rain_probability)"
                :show-text="true"
                :stroke-width="8"
              />
            </div>
          </template>
        </el-table-column>

        <el-table-column label="降雪概率" width="100">
          <template #default="{ row }">
            <div class="prob-cell">
              <el-progress
                :percentage="+(row.snow_probability * 100).toFixed(0)"
                color="#a78bfa"
                :show-text="true"
                :stroke-width="8"
              />
            </div>
          </template>
        </el-table-column>

        <el-table-column label="湿度(%)" width="90" sortable prop="humidity">
          <template #default="{ row }">
            {{ row.humidity.toFixed(0) }}
          </template>
        </el-table-column>

        <el-table-column label="风速(m/s)" width="100" sortable prop="wind_speed">
          <template #default="{ row }">
            {{ row.wind_speed.toFixed(1) }}
          </template>
        </el-table-column>

        <el-table-column label="严重天气概率" width="120">
          <template #default="{ row }">
            <el-tag
              :type="row.severe_probability > 0.1 ? 'danger' : 'info'"
              size="small"
              effect="dark"
            >
              {{ (row.severe_probability * 100).toFixed(0) }}%
            </el-tag>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <!-- 温度范围对比图 -->
    <el-card shadow="never" class="section-card">
      <template #header>
        <span class="section-title">&#127777; 7天温度范围对比</span>
      </template>
      <v-chart
        class="temp-range-chart"
        :option="tempRangeOption"
        autoresize
      />
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { BarChart } from 'echarts/charts'
import { GridComponent, TooltipComponent } from 'echarts/components'
import type { PredictionResponse } from '../types'

use([CanvasRenderer, BarChart, GridComponent, TooltipComponent])

interface Props {
  data: PredictionResponse
  loading: boolean
}

const props = defineProps<Props>()

const DAY_NAMES = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']

function tableHeaderStyle() {
  return { background: 'rgba(30, 41, 59, 0.8)', color: '#94a3b8', fontWeight: 600 }
}

function tableCellStyle() {
  return { background: 'transparent', color: '#cbd5e1' }
}

function formatDate(dateStr: string): string {
  const d = new Date(dateStr)
  return `${d.getMonth() + 1}/${d.getDate()} (${DAY_NAMES[d.getDay()]})`
}

function weatherIcon(w: string): string {
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
  return map[w] || '\u2601\uFE0F'
}

function weatherTagType(w: string): string {
  if (w === 'Sunny' || w === 'Clear') return 'success'
  if (w === 'Partly Cloudy' || w === 'Cloudy' || w === 'Overcast') return ''
  if (w === 'Light Rain' || w === 'Drizzle' || w === 'Mist') return 'warning'
  if (w === 'Rain' || w === 'Moderate Rain') return 'warning'
  if (w === 'Heavy Rain' || w === 'Thunderstorm') return 'danger'
  if (w === 'Snow' || w === 'Heavy Snow' || w === 'Sleet') return 'danger'
  if (w === 'Fog') return 'warning'
  if (w === 'Freezing' || w === 'Scorching') return 'danger'
  return 'info'
}

function rainProgressColor(p: number): string {
  if (p > 0.6) return '#ef4444'
  if (p > 0.3) return '#f59e0b'
  return '#22c55e'
}

const tempRangeOption = computed(() => {
  const dates = props.data.daily.map(d => {
    const dt = new Date(d.date)
    return `${dt.getMonth() + 1}/${dt.getDate()}`
  })
  const highs = props.data.daily.map(d => d.temperature_max)
  const lows = props.data.daily.map(d => d.temperature_min)

  return {
    tooltip: {
      trigger: 'axis',
      backgroundColor: 'rgba(15, 23, 42, 0.95)',
      borderColor: 'rgba(100, 140, 255, 0.3)',
      textStyle: { color: '#e2e8f0' },
    },
    grid: { top: 20, right: 20, bottom: 30, left: 50 },
    xAxis: {
      type: 'category',
      data: dates,
      axisLine: { lineStyle: { color: '#334155' } },
      axisLabel: { color: '#94a3b8' },
    },
    yAxis: {
      type: 'value',
      name: '°C',
      nameTextStyle: { color: '#64748b' },
      axisLabel: { color: '#94a3b8' },
      splitLine: { lineStyle: { color: '#1e293b' } },
    },
    series: [
      {
        name: '最高温度',
        type: 'bar',
        data: highs,
        itemStyle: { color: '#fb923c', borderRadius: [4, 4, 0, 0] },
        barGap: '20%',
      },
      {
        name: '最低温度',
        type: 'bar',
        data: lows,
        itemStyle: { color: '#60a5fa', borderRadius: [4, 4, 0, 0] },
      },
    ],
  }
})
</script>

<style scoped>
.daily-page {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

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

.daily-table {
  --el-table-tr-bg-color: transparent;
  --el-table-header-bg-color: rgba(30, 41, 59, 0.8);
}

.daily-table :deep(.el-table__body tr:hover > td) {
  background: rgba(59, 130, 246, 0.08) !important;
}

.date-cell {
  font-weight: 600;
  color: #e2e8f0;
}

.weather-icon-cell {
  font-size: 24px;
}

.temp-hi {
  color: #fb923c;
  font-weight: 700;
}

.temp-lo {
  color: #60a5fa;
  font-weight: 600;
}

.rain-value {
  color: #60a5fa;
  font-weight: 600;
}

.prob-cell {
  min-width: 80px;
}

.temp-range-chart {
  height: 300px;
}
</style>
