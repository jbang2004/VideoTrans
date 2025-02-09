import os
from pathlib import Path
from dotenv import load_dotenv
import torch

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

    BATCH_SIZE = 6
    TARGET_SPEAKER_AUDIO_DURATION = 8
    VAD_SR = 16000
    VOCALS_VOLUME = 0.7
    BACKGROUND_VOLUME = 0.3
    AUDIO_OVERLAP = 1024
    NORMALIZATION_THRESHOLD = 0.9

    SEGMENT_MINUTES = 5
    MIN_SEGMENT_MINUTES = 3

    TRANSLATION_MODEL = os.getenv("TRANSLATION_MODEL", "deepseek")
    ZHIPUAI_API_KEY = os.getenv("ZHIPUAI_API_KEY", "")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

    SYSTEM_PATHS = [
        str(current_dir / 'models' / 'CosyVoice'),
        str(current_dir / 'models' / 'ClearVoice'),
        str(current_dir / 'models' / 'CosyVoice' / 'third_party' / 'Matcha-TTS')
    ]

    MODEL_DIR = project_dir / "models"

    # =========== 全局音频配置 ===========
    SAMPLE_RATE = 24000  # 全局采样率，从 CosyVoice 模型加载后会被更新
    VAD_SR = 16000
    VOCALS_VOLUME = 0.7
    BACKGROUND_VOLUME = 0.3
    AUDIO_OVERLAP = 1024
    NORMALIZATION_THRESHOLD = 0.9

    # =========== ASR 模型相关配置 ===========
    ASR_MODEL_DIR = MODEL_DIR / "SenseVoice"
    ASR_MODEL_NAME = "iic/SenseVoiceSmall"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # ASR 模型的其他参数
    ASR_MODEL_KWARGS = {
        "model": "iic/SenseVoiceSmall",
        "remote_code": "./models/SenseVoice/model.py",
        "vad_model": "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        "vad_kwargs": {"max_single_segment_time": 30000},
        "spk_model": "cam++",
        "trust_remote_code": True,
        "disable_update": True,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

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
    SPEAKER_AUDIO_TARGET_DURATION = 8.0
    TRANSLATION_BATCH_SIZE = 50
    MODELIN_BATCH_SIZE = 3
    # 控制同时处理多少个视频分段
    MAX_PARALLEL_SEGMENTS = 2
