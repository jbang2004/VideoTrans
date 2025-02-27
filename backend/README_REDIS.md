# Redis 队列系统使用指南

本文档介绍如何使用 Redis 队列系统进行视频翻译任务的处理。

## 系统架构

系统使用 Redis 作为任务队列和状态存储，实现了分布式的视频翻译处理流程。主要组件包括：

1. **任务状态管理**：使用 `TaskState` 类存储任务状态，可序列化到 Redis 和文件系统
2. **Redis 队列**：各个处理阶段之间通过 Redis 队列传递任务
3. **Worker 装饰器**：使用 `redis_worker_decorator` 简化 Worker 的实现
4. **Worker 启动器**：使用 `worker_launcher.py` 启动各个 Worker

## 依赖

- Python 3.7+
- Redis 服务器
- aioredis 库

## 安装依赖

```bash
pip install aioredis
```

## 启动 Redis 服务器

确保 Redis 服务器已启动：

```bash
# 检查 Redis 服务状态
systemctl status redis

# 如果未启动，启动 Redis 服务
systemctl start redis

# 或者使用 Docker 启动 Redis
docker run --name redis -p 6379:6379 -d redis
```

## 启动 Worker

可以使用 `worker_launcher.py` 启动所有或指定的 Worker：

```bash
# 启动所有 Worker
python worker_launcher.py

# 启动指定 Worker
python worker_launcher.py --worker segment
python worker_launcher.py --worker asr
python worker_launcher.py --worker translation
# ... 其他 Worker
```

## 任务处理流程

1. **初始化任务**：通过 `ViTranslator.trans_video()` 方法创建任务
2. **分段处理**：`SegmentWorker` 将视频分段并更新总段数
3. **语音识别**：`ASRWorker` 识别音频内容
4. **翻译**：`TranslationWorker` 翻译识别的文本
5. **模型输入**：`ModelInWorker` 准备 TTS 模型输入
6. **TTS 处理**：`TTSTokenWorker` 生成 TTS 所需的 token
7. **时长对齐**：`DurationWorker` 对齐音频时长
8. **音频生成**：`AudioGenWorker` 生成翻译后的音频
9. **混音**：`MixerWorker` 将生成的音频与原视频混合

## 测试

使用 `test_redis_queue.py` 测试 Redis 队列系统：

```bash
python test_redis_queue.py
```

## 监控队列

可以使用 Redis CLI 监控队列状态：

```bash
# 连接到 Redis
redis-cli

# 查看所有队列
keys *

# 查看队列长度
llen segment_init_queue
llen asr_queue
# ... 其他队列
```

## 故障排除

1. **连接 Redis 失败**：
   - 检查 Redis 服务是否运行
   - 检查 `REDIS_URL` 配置是否正确

2. **任务状态加载失败**：
   - 检查任务 ID 是否正确
   - 检查任务目录是否存在

3. **Worker 处理异常**：
   - 查看日志了解具体错误
   - 检查相关依赖是否安装正确

## 关键文件

- `utils/task_state.py`：任务状态类
- `utils/redis_utils.py`：Redis 操作工具
- `utils/redis_decorators.py`：Worker 装饰器
- `worker_launcher.py`：Worker 启动器
- `video_translator.py`：视频翻译主流程
