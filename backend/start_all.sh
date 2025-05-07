#!/bin/bash

echo "===== 启动 VideoTrans 服务 (统一启动器) ====="

# 获取脚本所在的目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 启动统一的launcher.py
# 确保从 backend 目录的上下文执行，或者 launcher.py 处理好路径依赖
# Assuming launcher.py is in the same directory as this script (backend/)
python "$SCRIPT_DIR/launcher.py"

RET_CODE=$?

if [ $RET_CODE -eq 0 ]; then
  echo "Launcher.py 启动成功 (API服务通常在此阻塞运行)."
  echo "如需停止所有服务, 请在前台终止 launcher.py (Ctrl+C)."
else
  echo "Launcher.py 启动失败. 返回码: $RET_CODE"
fi