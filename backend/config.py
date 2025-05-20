import os
from pathlib import Path
from dotenv import load_dotenv
import logging.config

current_dir = Path(__file__).parent
env_path = current_dir / '.env'
load_dotenv(env_path)

project_dir = current_dir.parent
storage_dir = project_dir / 'storage'

class Config:
    SERVER_HOST = "0.0.0.0"
    SERVER_PORT = 8000
    LOG_LEVEL = "DEBUG"

    BASE_DIR = storage_dir
    TASKS_DIR = BASE_DIR / "tasks"
    PUBLIC_DIR = BASE_DIR / "public"

    # ---> 将 Supabase 配置移到这里
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    # 统一使用 Service Role Key
    SUPABASE_KEY = SUPABASE_SERVICE_ROLE_KEY

    BATCH_SIZE = 6
    TARGET_SPEAKER_AUDIO_DURATION = 10
    VAD_SR = 16000
    VOCALS_VOLUME = 0.7
    BACKGROUND_VOLUME = 0.3
    AUDIO_OVERLAP = 1024
    SILENCE_FADE_MS = 25  # 静音边界淡变长度（毫秒）
    NORMALIZATION_THRESHOLD = 0.9
    
    # 目标采样率，统一设置为24000
    TARGET_SR = 24000

    SEGMENT_MINUTES = 5
    MIN_SEGMENT_MINUTES = 3

    TRANSLATION_MODEL = os.getenv("TRANSLATION_MODEL", "deepseek")
    ZHIPUAI_API_KEY = os.getenv("ZHIPUAI_API_KEY", "")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
    XAI_API_KEY = os.getenv("XAI_API_KEY", "")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

    SYSTEM_PATHS = [
        str(current_dir / 'models' / 'CosyVoice'),
        str(current_dir / 'models' / 'ClearVoice'),
        str(current_dir / 'models' / 'CosyVoice' / 'third_party' / 'Matcha-TTS')
    ]

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

    @classmethod
    def init_directories(cls):
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

    MAX_GAP_MS = 2000
    SHORT_SENTENCE_MERGE_THRESHOLD_MS = 1000
    MAX_TOKENS_PER_SENTENCE = 80
    MIN_SENTENCE_LENGTH = 4
    SENTENCE_END_TOKENS = {9686, 9688, 9676, 9705, 9728, 9729, 20046, 24883, 24879}
    STRONG_END_TOKENS = {9688, 9676, 9705, 9729, 20046, 24883}
    WEAK_END_TOKENS = {9686, 9728, 24879}
    SPEAKER_AUDIO_TARGET_DURATION = 20.0
    SPEAKER_AUDIO_MIN_DURATION = 5.0  # 最短音频持续时间（秒）
    TRANSLATION_BATCH_SIZE = 50
    TTS_BATCH_SIZE = 3
    # 控制同时处理多少个视频分段
    MAX_PARALLEL_SEGMENTS = 2

    # Actor资源配置
    CLEARVOICE_ACTOR_NUM_GPUS = 0.2  # 音频分离器
    ASR_ACTOR_NUM_GPUS = 0.2  # ASR模型
    TRANSLATOR_ACTOR_NUM_CPUS = 0.5  # 翻译Actor（CPU密集）
    MODELIN_ACTOR_NUM_CPUS = 0.5  # ModelIn处理Actor（CPU密集）
    MEDIA_MIXER_ACTOR_NUM_CPUS = 0.5  # 媒体混合Actor（CPU密集）

    # ASR流程配置
    ASR_BATCH_SIZE_S = 60  # 音频批处理大小(秒)
    ASR_USE_ITN = True     # 使用逆文本规范化
    ASR_MERGE_VAD = False  # 是否合并VAD结果

    COSYVOICE_MODEL_PATH = "models/CosyVoice/pretrained_models/CosyVoice2-0.5B"

# --- 全局日志配置 ---
LOG_DIR = storage_dir / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# 日志配置模板
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(levelname)s | %(asctime)s | %(name)s | L%(lineno)d | %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "standard",
            "filename": str(LOG_DIR / "app.log"),
            "maxBytes": 10 * 1024 * 1024,
            "backupCount": 5,
            "encoding": "utf-8",
        },
    },
    "root": {
        "level": "DEBUG",
        "handlers": ["console", "file"],
    },
    "loggers": {
        # 如需单独对第三方库或子系统配置，可在此添加
    },
}

def init_logging():
    """初始化全局日志配置，推荐在应用入口调用一次。"""
    logging.config.dictConfig(LOGGING_CONFIG)