<template>
  <div class="prediction-form">
    <el-card shadow="never" class="form-card">
      <div class="form-row">
        <div class="form-item">
          <label class="form-label">城市</label>
          <el-select
            v-model="form.city"
            filterable
            placeholder="选择或搜索城市..."
            size="large"
            class="form-select"
            :disabled="loading"
            @change="onCityChange"
          >
            <el-option
              v-for="c in cities"
              :key="c"
              :label="c"
              :value="c"
            />
          </el-select>
        </div>

        <div class="form-item">
          <label class="form-label">预测日期</label>
          <input
            v-model="form.target_time"
            type="date"
            class="native-date-input"
            :disabled="loading"
          />
        </div>

        <div class="form-item form-item-btn">
          <el-button
            type="primary"
            size="large"
            :loading="loading"
            :disabled="!form.city"
            @click="onSubmit"
            class="predict-btn"
          >
            <el-icon v-if="!loading"><Search /></el-icon>
            {{ loading ? '预测中...' : '开始预测' }}
          </el-button>
        </div>
      </div>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { reactive } from 'vue'
import { Search } from '@element-plus/icons-vue'
import type { PredictRequest } from '../types'

interface Props {
  cities: string[]
  loading: boolean
}

interface Emits {
  (e: 'predict', request: PredictRequest): void
}

const props = defineProps<Props>()
const emit = defineEmits<Emits>()

const tomorrow = new Date()
tomorrow.setDate(tomorrow.getDate() + 1)
const y = tomorrow.getFullYear()
const m = String(tomorrow.getMonth() + 1).padStart(2, '0')
const d = String(tomorrow.getDate()).padStart(2, '0')

const form = reactive<PredictRequest>({
  city: '',
  target_time: `${y}-${m}-${d}`,
})

function onCityChange() {
  // city selected
}

function onSubmit() {
  emit('predict', {
    city: form.city,
    target_time: form.target_time || undefined,
  })
}
</script>

<style scoped>
.prediction-form {
  margin-bottom: 24px;
}

.form-card {
  background: rgba(30, 41, 59, 0.8) !important;
  border: 1px solid rgba(100, 140, 255, 0.15) !important;
  border-radius: 12px !important;
  backdrop-filter: blur(8px);
}

.form-card :deep(.el-card__body) {
  padding: 20px 24px;
}

.form-row {
  display: flex;
  gap: 20px;
  align-items: flex-end;
  flex-wrap: wrap;
}

.form-item {
  display: flex;
  flex-direction: column;
  gap: 6px;
  flex: 1;
  min-width: 200px;
}

.form-item-btn {
  flex: 0 0 auto;
  min-width: auto;
}

.form-label {
  font-size: 13px;
  font-weight: 600;
  color: #94a3b8;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.form-select {
  width: 100%;
}

.native-date-input {
  width: 100%;
  height: 42px;
  padding: 0 12px;
  background: rgba(15, 23, 42, 0.6);
  border: 1px solid rgba(100, 140, 255, 0.2);
  border-radius: 8px;
  color: #e2e8f0;
  font-size: 14px;
  outline: none;
  transition: border-color 0.3s;
}

.native-date-input:focus {
  border-color: #3b82f6;
}

.native-date-input:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.native-date-input::-webkit-calendar-picker-indicator {
  filter: invert(0.7);
  cursor: pointer;
}

.predict-btn {
  background: linear-gradient(135deg, #3b82f6, #2563eb) !important;
  border: none !important;
  padding: 12px 32px !important;
  font-weight: 600;
  letter-spacing: 0.5px;
  transition: all 0.3s;
}

.predict-btn:hover {
  background: linear-gradient(135deg, #60a5fa, #3b82f6) !important;
  transform: translateY(-1px);
  box-shadow: 0 4px 20px rgba(59, 130, 246, 0.4);
}
</style>
