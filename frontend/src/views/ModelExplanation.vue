<template>
  <div class="explanation-page">
    <!-- 摘要 -->
    <el-card shadow="never" class="section-card">
      <template #header>
        <span class="section-title">&#129514; 预测解释摘要</span>
      </template>
      <div class="summary-text">{{ data.explanation.weather_summary }}</div>
    </el-card>

    <!-- 置信度 + 因素 -->
    <el-row :gutter="20" class="top-row">
      <el-col :span="12">
        <el-card shadow="never" class="section-card">
          <template #header>
            <span class="section-title">&#127919; 模型置信度</span>
          </template>
          <div class="confidence-section">
            <div class="big-confidence">{{ (data.explanation.model_confidence * 100).toFixed(1) }}%</div>
            <el-progress
              :percentage="data.explanation.model_confidence * 100"
              :color="confidenceColor"
              :stroke-width="16"
              :show-text="false"
              class="big-progress"
            />
            <p class="confidence-note">
              置信度基于模型在测试集上的表现（R²、F1 等指标）计算。
              值越接近 100% 表示预测越可靠。
            </p>
          </div>
        </el-card>
      </el-col>

      <el-col :span="12">
        <el-card shadow="never" class="section-card">
          <template #header>
            <span class="section-title">&#128269; 主要影响因素</span>
          </template>
          <div class="factors-list">
            <div
              v-for="(factor, idx) in data.explanation.contributing_factors"
              :key="idx"
              class="factor-item"
            >
              <span class="factor-num">{{ idx + 1 }}</span>
              <span class="factor-text">{{ factor }}</span>
            </div>
          </div>
        </el-card>
      </el-col>
    </el-row>

    <!-- 特征重要性 -->
    <el-row :gutter="20">
      <el-col :span="12">
        <el-card shadow="never" class="section-card">
          <template #header>
            <span class="section-title">&#128202; 特征重要性</span>
          </template>
          <div class="feature-chart-wrapper">
            <v-chart
              class="feature-chart"
              :option="featureImportanceOption"
              autoresize
            />
          </div>
        </el-card>
      </el-col>

      <!-- 模型权重 -->
      <el-col :span="12">
        <el-card shadow="never" class="section-card">
          <template #header>
            <span class="section-title">&#9878; 集成模型权重</span>
          </template>
          <div class="weights-section">
            <h4 class="weight-subtitle">Daily Ensemble (软投票)</h4>
            <div class="weight-bars">
              <div class="weight-row">
                <span class="weight-name">Model 1 (Linear)</span>
                <div class="weight-track">
                  <div
                    class="weight-fill weight-fill-blue"
                    :style="{ width: (getWeight('daily_ensemble', 'model_1_linear') * 100) + '%' }"
                  />
                </div>
                <span class="weight-pct">{{ (getWeight('daily_ensemble', 'model_1_linear') * 100).toFixed(0) }}%</span>
              </div>
              <div class="weight-row">
                <span class="weight-name">Model 3 (Deep)</span>
                <div class="weight-track">
                  <div
                    class="weight-fill weight-fill-purple"
                    :style="{ width: (getWeight('daily_ensemble', 'model_3_deep') * 100) + '%' }"
                  />
                </div>
                <span class="weight-pct">{{ (getWeight('daily_ensemble', 'model_3_deep') * 100).toFixed(0) }}%</span>
              </div>
            </div>

            <h4 class="weight-subtitle" style="margin-top: 24px;">
              Hourly Ensemble
            </h4>
            <div class="weight-bars">
              <div class="weight-row">
                <span class="weight-name">Model 3 (Wide&amp;Deep)</span>
                <div class="weight-track">
                  <div
                    class="weight-fill weight-fill-purple"
                    style="width: 100%"
                  />
                </div>
                <span class="weight-pct">100%</span>
              </div>
            </div>

            <div class="ensemble-note">
              <h4>&#128161; 集成策略说明</h4>
              <p>
                Daily 预测使用<strong>软投票集成</strong>：线性模型（基于规则记忆）
                和深度学习模型（高维泛化）的预测结果按性能加权平均。
                权重基于各模型在验证集上的 R²/F1 指标动态计算。
              </p>
              <p style="margin-top: 8px;">
                Hourly 预测使用单一 Wide&amp;Deep 模型，该模型同时处理回归（温度/湿度）
                和分类（降雨概率）任务。
              </p>
            </div>
          </div>
        </el-card>
      </el-col>
    </el-row>
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

const confidenceColor = computed(() => {
  const c = props.data.explanation.model_confidence
  if (c > 0.85) return '#22c55e'
  if (c > 0.7) return '#f59e0b'
  return '#ef4444'
})

function getWeight(ensemble: string, model: string): number {
  const w = props.data.explanation.model_weights
  try {
    const ensembleWeights = w[ensemble] as Record<string, number>
    return ensembleWeights?.[model] ?? 0
  } catch {
    return 0
  }
}

const featureImportanceOption = computed(() => {
  const fi = props.data.explanation.feature_importance
  return {
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'shadow' },
      backgroundColor: 'rgba(15, 23, 42, 0.95)',
      borderColor: 'rgba(100, 140, 255, 0.3)',
      textStyle: { color: '#e2e8f0' },
    },
    grid: { top: 10, right: 30, bottom: 20, left: 160 },
    xAxis: {
      type: 'value',
      name: '重要性',
      axisLabel: { color: '#94a3b8' },
      splitLine: { lineStyle: { color: '#1e293b' } },
    },
    yAxis: {
      type: 'category',
      data: fi.map(f => f.feature).reverse(),
      axisLabel: { color: '#cbd5e1', fontSize: 12 },
      axisLine: { lineStyle: { color: '#334155' } },
    },
    series: [{
      type: 'bar',
      data: fi.map(f => +(f.importance * 100).toFixed(1)).reverse(),
      itemStyle: {
        color: {
          type: 'linear',
          x: 0, y: 0, x2: 1, y2: 0,
          colorStops: [
            { offset: 0, color: '#3b82f6' },
            { offset: 1, color: '#60a5fa' },
          ],
        },
        borderRadius: [0, 4, 4, 0],
      },
      barWidth: '60%',
      label: {
        show: true,
        position: 'right',
        formatter: '{c}%',
        color: '#94a3b8',
        fontSize: 11,
      },
    }],
  }
})
</script>

<style scoped>
.explanation-page {
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

.summary-text {
  font-size: 15px;
  color: #cbd5e1;
  line-height: 1.7;
}

.top-row {
  margin-bottom: 0;
}

/* Confidence */
.confidence-section {
  text-align: center;
  padding: 10px 0;
}

.big-confidence {
  font-size: 48px;
  font-weight: 800;
  color: #22c55e;
  line-height: 1;
}

.big-progress {
  margin: 16px 0;
}

.confidence-note {
  font-size: 13px;
  color: #64748b;
  margin-top: 8px;
}

/* Factors */
.factors-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.factor-item {
  display: flex;
  align-items: flex-start;
  gap: 12px;
}

.factor-num {
  width: 24px;
  height: 24px;
  border-radius: 50%;
  background: rgba(59, 130, 246, 0.2);
  color: #60a5fa;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
  font-weight: 700;
  flex-shrink: 0;
}

.factor-text {
  font-size: 14px;
  color: #cbd5e1;
  line-height: 1.6;
}

/* Feature chart */
.feature-chart-wrapper {
  height: 280px;
}

.feature-chart {
  width: 100%;
  height: 100%;
}

/* Weights */
.weight-subtitle {
  font-size: 14px;
  color: #94a3b8;
  margin-bottom: 12px;
  font-weight: 600;
}

.weight-bars {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.weight-row {
  display: flex;
  align-items: center;
  gap: 10px;
}

.weight-name {
  width: 140px;
  font-size: 13px;
  color: #cbd5e1;
  text-align: right;
  flex-shrink: 0;
}

.weight-track {
  flex: 1;
  height: 10px;
  background: rgba(15, 23, 42, 0.6);
  border-radius: 5px;
  overflow: hidden;
}

.weight-fill {
  height: 100%;
  border-radius: 5px;
  transition: width 0.6s ease;
}

.weight-fill-blue {
  background: linear-gradient(90deg, #3b82f6, #60a5fa);
}

.weight-fill-purple {
  background: linear-gradient(90deg, #8b5cf6, #a78bfa);
}

.weight-pct {
  width: 36px;
  font-size: 13px;
  font-weight: 700;
  color: #94a3b8;
}

.ensemble-note {
  margin-top: 24px;
  padding: 16px;
  background: rgba(15, 23, 42, 0.4);
  border-radius: 8px;
  border: 1px solid rgba(100, 140, 255, 0.1);
}

.ensemble-note h4 {
  font-size: 14px;
  color: #e2e8f0;
  margin-bottom: 8px;
}

.ensemble-note p {
  font-size: 13px;
  color: #94a3b8;
  line-height: 1.6;
}
</style>
