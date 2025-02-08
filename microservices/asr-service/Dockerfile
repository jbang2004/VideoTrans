FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 创建模型目录
RUN mkdir -p /app/models/SenseVoice \
    /app/models/CosyVoice \
    /app/models/MossFormer

# 复制应用代码
COPY ./asr-service /app/asr-service

# 安装Python依赖
COPY ./asr-service/requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# 设置工作目录
WORKDIR /app/asr-service

CMD ["python", "asr_worker.py"] 