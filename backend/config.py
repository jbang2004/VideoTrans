# ================================
# File: config.py
# ================================
import os
from pathlib import Path
from dotenv import load_dotenv

# 获取 backend 目录路径
current_dir = Path(__file__).parent

# 加载 .env 文件（如果有的话）
env_path = current_dir / '.env'
load_dotenv(env_path)

# 获取项目相关目录
project_dir = current_dir.parent
storage_dir = project_dir / 'storage'


class Config:
    """
    仅保留一个配置类，固定 LOG_LEVEL="DEBUG"。
    原先按 FLASK_ENV 切换的逻辑全部移除。
    """

    # 服务器基础配置
    SERVER_HOST = "0.0.0.0"
    SERVER_PORT = 8000

    # 日志级别统一使用 DEBUG
    LOG_LEVEL = "DEBUG"

    # 存储相关配置
    BASE_DIR = storage_dir
    TASKS_DIR = BASE_DIR / "tasks"      # 任务工作目录
    PUBLIC_DIR = BASE_DIR / "public"    # 公共访问目录

    # 批处理配置
    BATCH_SIZE = 6

    # 音频配置
    TARGET_SPEAKER_AUDIO_DURATION = 8
    VAD_SR = 16000
    VOCALS_VOLUME = 0.7
    BACKGROUND_VOLUME = 0.3
    AUDIO_OVERLAP = 1024
    NORMALIZATION_THRESHOLD = 0.9

    # 视频分段配置
    SEGMENT_MINUTES = 5
    MIN_SEGMENT_MINUTES = 3

    # 模型配置：可从 .env 或环境变量中获取
    # 支持的翻译模型: glm4, gemini, deepseek
    TRANSLATION_MODEL = os.getenv("TRANSLATION_MODEL", "deepseek")
    ZHIPUAI_API_KEY = os.getenv("ZHIPUAI_API_KEY", "")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

    # 系统路径配置
    SYSTEM_PATHS = [
        str(current_dir / 'models' / 'CosyVoice'),
        str(current_dir / 'models' / 'ClearVoice'),
        str(current_dir / 'models' / 'CosyVoice' / 'third_party' / 'Matcha-TTS')
    ]

    # 额外可选属性：MODEL_DIR、MODEL_PATH
    MODEL_DIR = project_dir / "models"

    @property
    def MODEL_PATH(self) -> Path:
        return Path(self.MODEL_DIR)

    @property
    def BASE_PATH(self) -> Path:
        return self.BASE_DIR

    @property
    def TASKS_PATH(self) -> Path:
        return self.TASKS_DIR

    @property
    def PUBLIC_PATH(self) -> Path:
        return self.PUBLIC_DIR

    # 初始化必须存在的目录
    @classmethod
    def init_directories(cls):
        """初始化所有必要的目录"""
        directories = [
            cls.BASE_DIR,
            cls.TASKS_DIR,
            cls.PUBLIC_DIR,
            cls.PUBLIC_DIR / "playlists",
            cls.PUBLIC_DIR / "segments"
        ]
        for dir_path in directories:
            dir_path.mkdir(parents=True, exist_ok=True)
            os.chmod(str(dir_path), 0o755)

    # 句子分割与合并时使用的阈值等
    MAX_GAP_MS = 2000  # merge_sentences中的最大间隔时间
    SHORT_SENTENCE_MERGE_THRESHOLD_MS = 1000  # process_tokens中短句合并的时间差阈值
    MAX_TOKENS_PER_SENTENCE = 80  # tokens_timestamp_sentence中的最大token数限制
    MIN_SENTENCE_LENGTH = 4       # tokens_timestamp_sentence中的最小句子长度阈值
    SENTENCE_END_TOKENS = {9686, 9688, 9676, 9705, 9728, 9729, 20046, 24883, 24879}
    STRONG_END_TOKENS = {9688, 9676, 9705, 9729, 20046, 24883}  # 句号、感叹号、问号等
    WEAK_END_TOKENS = {9686, 9728, 24879}
    SPEAKER_AUDIO_TARGET_DURATION = 8.0  # 提取说话人音频的目标长度(秒)
    TRANSLATION_BATCH_SIZE = 50         # 每批翻译最大句子数
    MODELIN_BATCH_SIZE = 3              # 每批模型输入最大句子数
