# VideoTrans 安装指南

## 系统要求

- Python 3.8+ (推荐 Python 3.10)
- Node.js 18+ 和 npm
- CUDA 支持的 GPU（用于语音分离、ASR 和 TTS 模型）
- FFmpeg 安装在系统上（必须）
- 至少 8GB 显存的 GPU（推荐 16GB+）
- 至少 16GB 系统内存

## 系统依赖安装

### Ubuntu/Debian

```bash
# 安装FFmpeg和其他系统依赖
sudo apt-get update
sudo apt-get install -y ffmpeg libsndfile1 python3-dev build-essential
```

### CentOS/RHEL

```bash
# 安装EPEL仓库
sudo yum install -y epel-release
# 安装FFmpeg和其他系统依赖
sudo yum install -y ffmpeg libsndfile-devel python3-devel gcc gcc-c++
```

## 后端安装

1. 安装Python依赖：

```bash
cd backend
pip install -r ../requirements.txt
```

2. 配置环境变量：

在 `.env` 文件中设置必要的 API 密钥和配置：

```
TRANSLATION_MODEL=deepseek  # 可选：deepseek, zhipu, gemini
DEEPSEEK_API_KEY=your_key_here
ZHIPUAI_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
```

3. 下载模型：
   
模型应该放在项目根目录下的 `models` 文件夹中，按照以下结构：

```
models/
  ├── ClearVoice/
  │   └── clearvoice.py  # 音频分离模型代码
  │   └── MossFormer2_SE_48K.ckpt  # 音频分离模型
  ├── SenseVoice/
  │   └── model.py  # ASR模型代码
  └── CosyVoice/
      ├── 模型文件
      └── third_party/
          └── Matcha-TTS/
              └── 模型文件
```

您可以从以下链接下载模型：
- ClearVoice: https://github.com/wenet-e2e/ClearVoice
- SenseVoice: https://github.com/iic-modelinghub/SenseVoice
- CosyVoice TTS模型: 请参考官方渠道获取

## 前端安装

1. 安装依赖：

```bash
cd frontend
npm install
```

2. 构建前端：

```bash
npm run build
```

## 启动服务

1. 启动后端服务：

```bash
cd backend
python api.py
```

2. 启动前端开发服务器（开发模式）：

```bash
cd frontend
npm run dev
```

或者启动生产模式服务：

```bash
cd frontend
npm run start
```

默认情况下，后端将在 `http://localhost:8000` 上运行，前端将在 `http://localhost:3000` 上运行。

## 常见问题

1. **CUDA错误**：确保您的CUDA驱动程序版本与PyTorch兼容。可以通过以下命令检查兼容性：
   ```
   python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
   ```

2. **内存不足**：
   - 在`config.py`中调整资源分配参数（如`MAX_PARALLEL_SEGMENTS`）
   - 减小`BATCH_SIZE`、`TRANSLATION_BATCH_SIZE`等参数
   - 调整各Actor的GPU分配比例，例如降低`COSYVOICE_ACTOR_NUM_GPUS`

3. **模型加载错误**：
   - 检查模型路径是否正确
   - 确保模型文件完整且未损坏
   - 检查`config.py`中的`SYSTEM_PATHS`设置是否正确

4. **FFmpeg错误**：
   - 确保FFmpeg已正确安装并添加到系统PATH中
   - 可通过运行`ffmpeg -version`检查安装状态

5. **依赖版本冲突**：
   - 建议使用虚拟环境隔离项目依赖：`python -m venv venv`
   - 如遇到兼容性问题，可尝试降级某些包版本 
如果服务器的npm版本太低，使用以下命令安装nvm
curl -o- https://cdn.jsdelivr.net/gh/nvm-sh/nvm@v0.40.2/install.sh | bash
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"  # This loads nvm
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"  # This loads nvm bash_completion
nvm install 20
nvm use 20