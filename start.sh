#!/bin/bash
###############################################################################
# DeepSeek-OCR 一键启动脚本
# 支持：Python 后端 (FastAPI) + Vite 前端
# 作者：九天Hector
###############################################################################

set -e

# ✅ 确保 Node.js 路径加载
export PATH=/usr/local/lib/nodejs/node-v22.21.0-linux-x64/bin:$PATH

# 彩色输出
GREEN="\033[1;32m"
YELLOW="\033[1;33m"
RED="\033[1;31m"
RESET="\033[0m"

echo -e "${GREEN}============================================================${RESET}"
echo -e "🚀 ${YELLOW}正在启动 DeepSeek-OCR 项目...${RESET}"
echo -e "${GREEN}============================================================${RESET}"

# 初始化 Conda 环境
if ! command -v conda &> /dev/null; then
    echo -e "${RED}❌ 未检测到 Conda，请先安装 Miniconda 或 Anaconda。${RESET}"
    exit 1
fi

source $(conda info --base)/etc/profile.d/conda.sh

# 1️⃣ 激活虚拟环境
echo -e "${YELLOW}>>> Step 1. 激活 Conda 环境 deepseek-ocr${RESET}"
conda activate deepseek-ocr || { echo -e "${RED}❌ 无法激活 deepseek-ocr 环境。${RESET}"; exit 1; }

# 2️⃣ 启动后端服务
BACKEND_PORT=8002
if lsof -i:$BACKEND_PORT >/dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  后端端口 $BACKEND_PORT 已被占用，尝试关闭旧进程...${RESET}"
    fuser -k ${BACKEND_PORT}/tcp || true
fi

echo -e "${YELLOW}>>> Step 2. 启动后端服务 (Uvicorn)...${RESET}"
cd backend || cd .
nohup uvicorn main:app --host 0.0.0.0 --port ${BACKEND_PORT} --reload > ../backend.log 2>&1 &
BACK_PID=$!
disown $BACK_PID 2>/dev/null || true
echo -e "${GREEN}✅ 后端已启动 (PID: $BACK_PID)，日志写入 backend.log${RESET}"
cd ..

# 3️⃣ 启动前端服务
FRONTEND_PORT=3000
if lsof -i:$FRONTEND_PORT >/dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  前端端口 $FRONTEND_PORT 已被占用，尝试关闭旧进程...${RESET}"
    fuser -k ${FRONTEND_PORT}/tcp || true
fi

echo -e "${YELLOW}>>> Step 3. 启动前端服务 (Vite)...${RESET}"
cd frontend
nohup npm run dev -- --host > ../frontend.log 2>&1 &
FRONT_PID=$!
disown $FRONT_PID 2>/dev/null || true
echo -e "${GREEN}✅ 前端已启动 (PID: $FRONT_PID)，日志写入 frontend.log${RESET}"
cd ..

# 4️⃣ 启动完成信息
echo -e "${GREEN}============================================================${RESET}"
echo -e "${GREEN}🎉 DeepSeek-OCR 启动成功！${RESET}"
echo -e "🌐 后端接口地址: ${YELLOW}http://127.0.0.1:${BACKEND_PORT}${RESET}"
echo -e "🖥️  前端访问地址: ${YELLOW}http://127.0.0.1:${FRONTEND_PORT}${RESET}"
echo -e "🧾 后端日志: backend.log"
echo -e "🧾 前端日志: frontend.log"
echo -e ""
echo -e "${YELLOW}💡 提示: 服务已在后台运行，关闭终端不会影响服务运行${RESET}"
echo -e "${YELLOW}💡 如需停止服务，请使用: pkill -f 'uvicorn|vite'${RESET}"
echo -e "${GREEN}============================================================${RESET}"