<template>
  <div class="app-container">
    <el-container>
      <el-header class="app-header">
        <div class="header-content">
          <div class="logo-section">
            <span class="logo-icon">&#9728;</span>
            <h1 class="app-title">天气预测平台</h1>
            <span class="app-subtitle">European Weather Prediction</span>
          </div>
          <el-menu
            mode="horizontal"
            :default-active="activeRoute"
            router
            class="nav-menu"
          >
            <el-menu-item index="/">
              <el-icon><Odometer /></el-icon>
              <span>Dashboard</span>
            </el-menu-item>
            <el-menu-item index="/hourly">
              <el-icon><Timer /></el-icon>
              <span>Hourly</span>
            </el-menu-item>
            <el-menu-item index="/daily">
              <el-icon><Calendar /></el-icon>
              <span>Daily</span>
            </el-menu-item>
            <el-menu-item index="/explanation">
              <el-icon><DataAnalysis /></el-icon>
              <span>Explanation</span>
            </el-menu-item>
          </el-menu>
        </div>
      </el-header>

      <el-main class="app-main">
        <PredictionForm
          :cities="cities"
          :loading="loading"
          @predict="handlePredict"
        />
        <router-view
          v-if="predictionData"
          :data="predictionData"
          :loading="loading"
        />
        <el-empty
          v-else
          description="请输入城市名称和预测日期，开始预测"
          :image-size="200"
        />
      </el-main>

      <el-footer class="app-footer">
        <span>Weather Prediction Platform &copy; 2024 | Powered by Ensemble Learning</span>
      </el-footer>
    </el-container>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useRoute } from 'vue-router'
import { Odometer, Timer, Calendar, DataAnalysis } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'
import PredictionForm from './components/PredictionForm.vue'
import { fetchPrediction, fetchCities } from './api'
import type { PredictionResponse, PredictRequest } from './types'

const route = useRoute()
const activeRoute = computed(() => route.path)

const predictionData = ref<PredictionResponse | null>(null)
const cities = ref<string[]>([])
const loading = ref(false)

onMounted(async () => {
  try {
    const data = await fetchCities()
    cities.value = data.cities
  } catch {
    ElMessage.warning('无法获取城市列表，请确保后端已启动')
  }
})

async function handlePredict(request: PredictRequest) {
  loading.value = true
  try {
    const data = await fetchPrediction(request)
    predictionData.value = data
    ElMessage.success(`预测完成: ${data.city}`)
  } catch (e: any) {
    ElMessage.error(e?.response?.data?.detail || '预测请求失败')
  } finally {
    loading.value = false
  }
}
</script>

<style>
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
  min-height: 100vh;
}

.app-container {
  min-height: 100vh;
}

.app-header {
  background: rgba(15, 23, 42, 0.95) !important;
  backdrop-filter: blur(12px);
  border-bottom: 1px solid rgba(100, 140, 255, 0.15);
  padding: 0 24px;
  height: 64px !important;
}

.header-content {
  display: flex;
  align-items: center;
  justify-content: space-between;
  height: 100%;
  max-width: 1400px;
  margin: 0 auto;
}

.logo-section {
  display: flex;
  align-items: center;
  gap: 12px;
}

.logo-icon {
  font-size: 28px;
  color: #f59e0b;
}

.app-title {
  font-size: 20px;
  font-weight: 700;
  color: #e2e8f0;
  letter-spacing: 1px;
}

.app-subtitle {
  font-size: 12px;
  color: #64748b;
  margin-left: 4px;
}

.nav-menu {
  background: transparent !important;
  border: none !important;
}

.nav-menu .el-menu-item {
  color: #94a3b8;
  border-bottom: 2px solid transparent;
  transition: all 0.3s;
}

.nav-menu .el-menu-item:hover {
  color: #e2e8f0;
  background: rgba(100, 140, 255, 0.08) !important;
}

.nav-menu .el-menu-item.is-active {
  color: #60a5fa !important;
  border-bottom-color: #60a5fa !important;
  background: transparent !important;
}

.app-main {
  max-width: 1400px;
  margin: 0 auto;
  padding: 24px;
  min-height: calc(100vh - 120px);
}

.app-footer {
  text-align: center;
  color: #475569;
  font-size: 13px;
  height: 56px !important;
  background: rgba(15, 23, 42, 0.95) !important;
  border-top: 1px solid rgba(100, 140, 255, 0.1);
  display: flex;
  align-items: center;
  justify-content: center;
}
</style>
