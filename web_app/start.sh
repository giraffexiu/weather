#!/bin/bash
# WeatherAI 启动脚本
# 用法: bash start.sh

PROJECT="$(cd "$(dirname "$0")/.." && pwd)"

# 激活 conda 环境
source /opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh
conda activate model-train

echo "========================================="
echo "  WeatherAI 天气预测系统"
echo "  后端: FastAPI + Wide & Deep"
echo "  前端: Chart.js + 暗色主题"
echo "========================================="
echo ""

cd "$PROJECT"

# 启动后端
python web_app/backend/main.py
