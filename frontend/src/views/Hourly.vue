<template>
  <div class="hourly-page">
    <el-card shadow="never" class="chart-card">
      <template #header>
        <div class="chart-header">
          <span class="chart-title">&#127777; 24小时温度变化</span>
          <span class="chart-subtitle">{{ data.city }}, {{ data.target_time }}</span>
        </div>
      </template>
      <v-chart
        class="chart"
        :option="tempOption"
        :loading="loading"
        autoresize
      />
    </el-card>

    <el-card shadow="never" class="chart-card">
      <template #header>
        <div class="chart-header">
          <span class="chart-title">&#9748; 24小时降雨概率</span>
        </div>
      </template>
      <v-chart
        class="chart"
        :option="rainOption"
        :loading="loading"
        autoresize
      />
    </el-card>

    <el-card shadow="never" class="chart-card">
      <template #header>
        <div class="chart-header">
          <span class="chart-title">&#128168; 风速与湿度</span>
        </div>
      </template>
      <v-chart
        class="chart"
        :option="windHumidityOption"
        :loading="loading"
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
import { LineChart, BarChart } from 'echarts/charts'
import {
  GridComponent, TooltipComponent, LegendComponent, TitleComponent,
} from 'echarts/components'
import type { PredictionResponse } from '../types'

use([
  CanvasRenderer, LineChart, BarChart,
  GridComponent, TooltipComponent, LegendComponent, TitleComponent,
])

interface Props {
  data: PredictionResponse
  loading: boolean
}

const props = defineProps<Props>()

function fmtHHmm(iso: string): string {
  return iso.slice(11, 16)
}

const hours = computed(() =>
  props.data.hourly.map(h => fmtHHmm(h.time))
)

const temperatures = computed(() =>
  props.data.hourly.map(h => h.temperature)
)

const apparentTemps = computed(() =>
  props.data.hourly.map(h => h.apparent_temperature)
)

const rainProbs = computed(() =>
  props.data.hourly.map(h => +(h.rain_probability * 100).toFixed(0))
)

const windSpeeds = computed(() =>
  props.data.hourly.map(h => h.wind_speed)
)

const humidities = computed(() =>
  props.data.hourly.map(h => h.humidity)
)

const tempOption = computed(() => ({
  tooltip: {
    trigger: 'axis',
    backgroundColor: 'rgba(15, 23, 42, 0.95)',
    borderColor: 'rgba(100, 140, 255, 0.3)',
    textStyle: { color: '#e2e8f0' },
  },
  legend: {
    data: ['温度', '体感温度'],
    textStyle: { color: '#94a3b8' },
    top: 0,
  },
  grid: { top: 40, right: 20, bottom: 30, left: 50 },
  xAxis: {
    type: 'category',
    data: hours.value,
    axisLine: { lineStyle: { color: '#334155' } },
    axisLabel: { color: '#94a3b8', fontSize: 11 },
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
      name: '温度',
      type: 'line',
      data: temperatures.value,
      smooth: true,
      lineStyle: { color: '#fb923c', width: 3 },
      itemStyle: { color: '#fb923c' },
      areaStyle: {
        color: {
          type: 'linear',
          x: 0, y: 0, x2: 0, y2: 1,
          colorStops: [
            { offset: 0, color: 'rgba(251, 146, 60, 0.3)' },
            { offset: 1, color: 'rgba(251, 146, 60, 0.02)' },
          ],
        },
      },
      markLine: {
        silent: true,
        data: [{ type: 'average', name: '平均' }],
        lineStyle: { color: '#f59e0b', type: 'dashed' },
        label: { color: '#f59e0b' },
      },
    },
    {
      name: '体感温度',
      type: 'line',
      data: apparentTemps.value,
      smooth: true,
      lineStyle: { color: '#60a5fa', width: 2, type: 'dashed' },
      itemStyle: { color: '#60a5fa' },
    },
  ],
}))

const rainOption = computed(() => ({
  tooltip: {
    trigger: 'axis',
    backgroundColor: 'rgba(15, 23, 42, 0.95)',
    borderColor: 'rgba(100, 140, 255, 0.3)',
    textStyle: { color: '#e2e8f0' },
    formatter: (p: any) => `${p[0].axisValue}<br/>${p[0].marker} 降雨概率: ${p[0].value}%`,
  },
  grid: { top: 20, right: 20, bottom: 30, left: 50 },
  xAxis: {
    type: 'category',
    data: hours.value,
    axisLine: { lineStyle: { color: '#334155' } },
    axisLabel: { color: '#94a3b8', fontSize: 11 },
  },
  yAxis: {
    type: 'value',
    name: '%',
    max: 100,
    nameTextStyle: { color: '#64748b' },
    axisLabel: { color: '#94a3b8' },
    splitLine: { lineStyle: { color: '#1e293b' } },
  },
  series: [{
    type: 'bar',
    data: rainProbs.value.map((v: number) => ({
      value: v,
      itemStyle: {
        color: v > 60 ? '#ef4444' : v > 30 ? '#f59e0b' : '#22c55e',
        borderRadius: [4, 4, 0, 0],
      },
    })),
    barWidth: '60%',
  }],
}))

const windHumidityOption = computed(() => ({
  tooltip: {
    trigger: 'axis',
    backgroundColor: 'rgba(15, 23, 42, 0.95)',
    borderColor: 'rgba(100, 140, 255, 0.3)',
    textStyle: { color: '#e2e8f0' },
  },
  legend: {
    data: ['风速 (m/s)', '湿度 (%)'],
    textStyle: { color: '#94a3b8' },
    top: 0,
  },
  grid: { top: 40, right: 20, bottom: 30, left: 50 },
  xAxis: {
    type: 'category',
    data: hours.value,
    axisLine: { lineStyle: { color: '#334155' } },
    axisLabel: { color: '#94a3b8', fontSize: 11 },
  },
  yAxis: [
    {
      type: 'value',
      name: 'm/s',
      nameTextStyle: { color: '#64748b' },
      axisLabel: { color: '#94a3b8' },
      splitLine: { lineStyle: { color: '#1e293b' } },
    },
    {
      type: 'value',
      name: '%',
      min: 0,
      max: 100,
      nameTextStyle: { color: '#64748b' },
      axisLabel: { color: '#94a3b8' },
      splitLine: { show: false },
    },
  ],
  series: [
    {
      name: '风速 (m/s)',
      type: 'line',
      data: windSpeeds.value,
      smooth: true,
      lineStyle: { color: '#a78bfa', width: 2 },
      itemStyle: { color: '#a78bfa' },
    },
    {
      name: '湿度 (%)',
      type: 'line',
      yAxisIndex: 1,
      data: humidities.value,
      smooth: true,
      lineStyle: { color: '#34d399', width: 2 },
      itemStyle: { color: '#34d399' },
      areaStyle: {
        color: {
          type: 'linear',
          x: 0, y: 0, x2: 0, y2: 1,
          colorStops: [
            { offset: 0, color: 'rgba(52, 211, 153, 0.15)' },
            { offset: 1, color: 'rgba(52, 211, 153, 0.02)' },
          ],
        },
      },
    },
  ],
}))
</script>

<style scoped>
.hourly-page {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.chart-card {
  background: rgba(30, 41, 59, 0.6) !important;
  border: 1px solid rgba(100, 140, 255, 0.12) !important;
  border-radius: 12px !important;
}

.chart-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.chart-title {
  font-size: 16px;
  font-weight: 700;
  color: #e2e8f0;
}

.chart-subtitle {
  font-size: 13px;
  color: #64748b;
}

.chart {
  height: 320px;
}
</style>
