# CosySenseDubber 微服务架构

这是一个基于微服务的视频翻译与配音系统示例，将原先在 `backend/` 下的单体式代码拆分成多个服务，使用 **RabbitMQ** 进行消息队列通信。

## 系统概览

1. **api-gateway**: 提供 HTTP 对外接口（可自行扩展 REST API 来接收上传视频、调用后续流程）
2. **asr-service**: 语音识别 (ASR)
3. **translation-service**: 翻译
4. **modelin-service**: 模型输入处理
5. **tts-service**: TTS Token 生成
6. **aligner-service**: 时长对齐
7. **audio-gen-service**: 最终音频合成
8. **mixer-service**: 混音输出

每个服务通过 RabbitMQ 队列串联：  
`asr_queue -> translation_queue -> modelin_queue -> tts_queue -> duration_align_queue -> audio_gen_queue -> mixer_queue`

## 运行步骤

1. **安装 Docker 和 Docker Compose**
2. **进入本目录** `microservices/`
3. **启动**  
   ```bash
   docker-compose up -d
   ```
4. **验证**  
   - RabbitMQ 管理端口: http://localhost:15672 (默认 guest / guest)
   - API Gateway: http://localhost:8000

## 自定义逻辑

- 如需接收文件上传并将任务投递到 `asr_queue`，可在 `api-gateway/main.py` 中实现。
- 每个 Worker（例如 `asr_worker.py`）内可替换实际 ASR/TTS/翻译等逻辑。
- 公共工具和配置在 `shared/` 目录。

## 其他

该示例为简化版，核心点在于演示如何从单体拆分成多个独立服务，并通过消息队列编排处理流程。根据需要可自由扩展。

## 目录结构

```
microservices/
├─ docker-compose.yml    # Docker 编排配置
├─ shared/              # 共享代码
│   ├─ config.py        # 配置管理
│   ├─ connection.py    # RabbitMQ 连接
│   └─ sentence_tools.py # 句子数据结构
├─ api-gateway/         # API 网关服务
├─ asr-service/         # 语音识别服务
├─ translation-service/ # 翻译服务
├─ modelin-service/    # 模型输入服务
├─ tts-service/        # 语音合成服务
├─ aligner-service/    # 时长对齐服务
├─ audio-gen-service/  # 音频生成服务
└─ mixer-service/      # 混音服务
```

## 安装和运行

1. 安装 Docker 和 Docker Compose

2. 配置环境变量（可选）：
   ```bash
   export TRANSLATION_MODEL=deepseek  # 或 gemini
   export DEEPSEEK_API_KEY=your_key
   export GEMINI_API_KEY=your_key
   ```

3. 启动服务：
   ```bash
   cd microservices
   docker-compose up -d
   ```

4. 访问 API：
   - 上传视频：`POST http://localhost:8000/upload`
   - 查询状态：`GET http://localhost:8000/status/{task_id}`

## API 文档

### 上传视频

**请求**
```http
POST /upload
Content-Type: multipart/form-data

video: [视频文件]
target_language: "zh"  # 目标语言，默认中文
generate_subtitle: false  # 是否生成字幕，默认否
```

**响应**
```json
{
    "task_id": "uuid",
    "status": "submitted"
}
```

### 查询状态

**请求**
```http
GET /status/{task_id}
```

**响应**
```json
{
    "task_id": "uuid",
    "status": "processing"  // submitted, processing, completed, failed
}
```

## 开发说明

1. 每个服务都有自己的 `Dockerfile` 和 `requirements.txt`
2. 共享代码放在 `shared` 目录下
3. 使用 RabbitMQ 进行服务间通信
4. 所有服务共享 `storage` 目录用于存储文件

## 消息队列

服务间通过以下队列进行通信：

1. asr_queue: API Gateway -> ASR Service
2. translation_queue: ASR Service -> Translation Service
3. modelin_queue: Translation Service -> ModelIn Service
4. tts_queue: ModelIn Service -> TTS Service
5. duration_align_queue: TTS Service -> Aligner Service
6. audio_gen_queue: Aligner Service -> Audio Gen Service
7. mixer_queue: Audio Gen Service -> Mixer Service
