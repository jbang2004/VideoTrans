#!/bin/bash

# 视频翻译后端服务启动脚本
set -e

echo "=== 启动视频翻译后端服务 ==="

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到 python3"
    exit 1
fi

# 检查必要的环境变量
if [ -z "$SUPABASE_URL" ] || [ -z "$SUPABASE_SERVICE_ROLE_KEY" ]; then
    echo "警告: 未设置 SUPABASE_URL 或 SUPABASE_SERVICE_ROLE_KEY 环境变量"
    echo "请确保 .env 文件存在并包含必要的配置"
fi

# 启动服务
echo "启动后端服务..."
cd "$(dirname "$0")"
python3 launcher.py

echo "=== 后端服务启动完成 ==="