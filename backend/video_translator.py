import sys
import logging
import asyncio
from pathlib import Path
import shutil
from typing import Dict, Optional
from core.translation.translator import Translator
from core.translation.glm4_client import GLM4Client
from core.translation.gemini_client import GeminiClient
from core.translation.deepseek_client import DeepSeekClient
from core.tts_token_gener import TTSTokenGenerator
from core.audio_gener import AudioGenerator
from utils.media_utils import MediaUtils
from core.media_mixer import MediaMixer
from core.hls_manager import HLSManager
from utils.sentence_logger import SentenceLogger
from core.audio_separator import ClearVoiceSeparator
from core.duration_aligner import DurationAligner
from core.timestamp_adjuster import TimestampAdjuster
from utils.decorators import worker_decorator, handle_errors
import os.path

# 导入自定义模块
from core.model_in import ModelIn
from models.CosyVoice.cosyvoice.cli.cosyvoice import CosyVoice
from core.auto_sense import SenseAutoModel as AutoModel

# 设置日志
logger = logging.getLogger(__name__)

class TaskState:
    """任务状态类"""
    def __init__(self, task_id: str, video_path: str, task_paths, hls_manager: HLSManager, target_language: str = "zh"):
        self.task_id = task_id
        self.video_path = video_path
        self.task_paths = task_paths
        self.hls_manager = hls_manager
        self.target_language = target_language
        self.sentence_counter = 0
        self.current_time = 0
        self.segment_counter = 0
        self.segment_media_files = {}  # 存储所有分段的媒体文件 {segment_index: media_files}
        
        # 任务相关的队列
        self.translation_queue = asyncio.Queue()
        self.modelin_queue = asyncio.Queue()
        self.tts_token_queue = asyncio.Queue()
        self.duration_align_queue = asyncio.Queue()
        self.audio_gen_queue = asyncio.Queue()
        self.mixing_queue = asyncio.Queue()
        
        # 用于监控第一段处理的完成状态
        self.mixing_complete = None

class ViTranslator:
    _instance = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, config=None):
        # 只在第一次初始化时加载模型
        if not self._initialized:
            self.logger = logging.getLogger(__name__)
            self.config = config
            self._init_models()
            self._initialized = True
        
        # 移除task_state的初始化，改为在trans_video时设置
        self.task_state = None

    def _init_models(self):
        """初始化所有需要的模型和工具"""
        self.logger.info("初始化模型（仅首次加载）...")
        
        # 初始化音频分离器
        self.audio_separator = ClearVoiceSeparator(
            model_name='MossFormer2_SE_48K'
        )
        
        # 初始化其他模型
        self.sense_model = AutoModel(
            config=self.config,
            model="iic/SenseVoiceSmall",
            remote_code="./models/SenseVoice/model.py",
            vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
            vad_kwargs={"max_single_segment_time": 30000},
            spk_model="cam++",
            trust_remote_code=True,
            disable_update=True,
            device="cuda"
        )
        
        # CosyVoice TTS模型
        # self.cosyvoice_model = CosyVoice2("models/CosyVoice/pretrained_models/CosyVoice2-0.5B")#CosyVoice-300M-25Hz
        self.cosyvoice_model = CosyVoice("models/CosyVoice/pretrained_models/CosyVoice-300M-25Hz")
        
        # 注入音频分离器
        self.media_utils = MediaUtils(
            config=self.config,
            audio_separator=self.audio_separator
        )
        
        # 初始化各个处理器
        self.model_in = ModelIn(self.cosyvoice_model.frontend)
        self.tts_generator = TTSTokenGenerator(self.cosyvoice_model.model, Hz=25)
        self.audio_generator = AudioGenerator(self.cosyvoice_model.model)
        self.duration_aligner = DurationAligner()
        self.timestamp_adjuster = TimestampAdjuster()

        # 初始化翻译器
        translation_model = self.config.TRANSLATION_MODEL.strip().strip('"').lower()
        self.logger.info(f"使用翻译模型: {repr(translation_model)}")
        
        try:
            if translation_model == "glm4":
                self.translator = Translator(GLM4Client(api_key=self.config.ZHIPUAI_API_KEY))
            elif translation_model == "gemini":
                self.translator = Translator(GeminiClient(api_key=self.config.GEMINI_API_KEY))
            elif translation_model == "deepseek":
                self.translator = Translator(DeepSeekClient(api_key=self.config.DEEPSEEK_API_KEY))
            else:
                import traceback
                error_location = traceback.extract_stack()[-1]
                error_msg = (
                    f"不支持的翻译模型: {repr(translation_model)}，支持的模型有: glm4, gemini, deepseek\n"
                    f"错误位置: {error_location.filename}:{error_location.lineno}\n"
                    f"配置来源: {self.config.__class__.__name__}"
                )
                self.logger.error(error_msg)
                raise ValueError(error_msg)
        except Exception as e:
            self.logger.error(f"初始化翻译器失败: {str(e)}", exc_info=True)
            raise

        self.mixer = MediaMixer(config=self.config, sample_rate=self.config.TARGET_SR)
        self.sentence_logger = SentenceLogger(self.config)
        
        self.logger.info("模型初始化完成")

    @worker_decorator(
        input_queue_attr='translation_queue',
        next_queue_attr='modelin_queue',
        worker_name='翻译工作者',
        mode='stream'
    )
    async def _translation_worker(self, sentences):
        """翻译工作者"""
        self.logger.debug(f"开始翻译 {len(sentences)} 个句子，目标语言: {self.task_state.target_language}")
        async for translated_batch in self.translator.translate_sentences(
            sentences, 
            batch_size=self.config.TRANSLATION_BATCH_SIZE,
            target_language=self.task_state.target_language
        ):
            yield translated_batch

    @worker_decorator(
        input_queue_attr='modelin_queue',
        next_queue_attr='tts_token_queue',
        worker_name='模型输入工作者',
        mode='stream'
    )
    async def _modelin_worker(self, sentences):
        """模型输入工作者"""
        self.logger.debug(f"开始处理 {len(sentences)} 个句子的模型输入")
        async for modelined_batch in self.model_in.modelin_maker(sentences, batch_size=self.config.MODELIN_BATCH_SIZE):
            yield modelined_batch

    @worker_decorator(
        input_queue_attr='tts_token_queue',
        next_queue_attr='duration_align_queue',
        worker_name='TTS Token 生成工作者'
    )
    async def _tts_token_worker(self, sentences):
        """TTS Token 生成工作者"""
        self.logger.debug(f"开始生成 {len(sentences)} 个句子的 TTS tokens")
        await self.tts_generator.tts_token_maker(sentences)
        return sentences

    @worker_decorator(
        input_queue_attr='duration_align_queue',
        next_queue_attr='audio_gen_queue',
        worker_name='时长对齐工作者'
    )
    async def _duration_align_worker(self, sentences):
        """时长对齐工作者"""
        self.logger.debug(f"开始对齐 {len(sentences)} 个句子的时长")
        self.duration_aligner.align_durations(sentences)
        return sentences

    @worker_decorator(
        input_queue_attr='audio_gen_queue',
        next_queue_attr='mixing_queue',
        worker_name='音频生成工作者'
    )
    async def _audio_generation_worker(self, sentences):
        """音频生成工作者"""
        self.logger.debug(f"开始生成 {len(sentences)} 个句子的音频")
        await self.audio_generator.vocal_audio_maker(sentences)
        
        self.task_state.current_time = self.timestamp_adjuster.update_timestamps(
            sentences, 
            start_time=self.task_state.current_time
        )
        
        if not self.timestamp_adjuster.validate_timestamps(sentences):
            self.logger.warning("检测到时间戳异常")
        
        return sentences

    @worker_decorator(
        input_queue_attr='mixing_queue',
        worker_name='混音工作者'
    )
    async def _mixing_worker(self, sentences):
        """混音工作者"""
        if not sentences:
            return None
            
        # 设置输出视频路径
        output_path = self.task_state.task_paths.segments_dir / f"segment_{self.task_state.segment_counter}.mp4"
        
        if await self.mixer.mixed_media_maker(
            sentences,
            task_state=self.task_state,
            output_path=str(output_path)
        ):
            await self.task_state.hls_manager.add_segment(str(output_path), self.task_state.segment_counter)
            
            # 如果是第一段且有监控队列，通知处理完成
            if self.task_state.segment_counter == 0 and self.task_state.mixing_complete:
                await self.task_state.mixing_complete.put(True)
            
            self.task_state.segment_counter += 1
            
        return None

    @handle_errors(logger)
    async def trans_video(self, video_path: str, task_id: str, task_paths, hls_manager: HLSManager, target_language: str = "zh") -> dict:
        """视频翻译主函数"""
        try:
            # 初始化任务状态，直接传入所有参数
            self.task_state = TaskState(
                task_id=task_id,
                video_path=video_path,
                task_paths=task_paths,
                hls_manager=hls_manager,
                target_language=target_language
            )
            
            self.logger.info(f"开始处理视频，任务ID: {self.task_state.task_id}，目标语言: {self.task_state.target_language}")
            
            # 获取视频时长
            duration = await self.media_utils.get_video_duration(self.task_state.video_path)
            
            # 获取音频分段
            segments = await self.media_utils.get_audio_segments(duration)
            self.logger.info(f"音频分段完成，共 {len(segments)} 个分段")

            # 启动工作者
            workers = [
                asyncio.create_task(self._translation_worker()),
                asyncio.create_task(self._modelin_worker()),
                asyncio.create_task(self._tts_token_worker()),
                asyncio.create_task(self._duration_align_worker()),
                asyncio.create_task(self._audio_generation_worker()),
                asyncio.create_task(self._mixing_worker())
            ]

            try:
                # 优先处理第一段
                if segments:
                    start, duration = segments[0]
                    self.logger.info(f"开始处理第一段: start={start}, duration={duration}")
                    await self._process_segment(0, start, duration, is_first=True)
                    
                    # 处理剩余分段
                    for i, (start, duration) in enumerate(segments[1:], 1):
                        self.logger.info(f"开始处理第 {i} 段: start={start}, duration={duration}")
                        await self._process_segment(i, start, duration)

            finally:
                # 发送停止信号
                await self.task_state.translation_queue.put(None)
                # 等待所有工作者完成
                await asyncio.gather(*workers, return_exceptions=True)

            # 检查是否有成功处理的片段
            if not self.task_state.hls_manager.has_segments:
                self.logger.error("没有成功处理任何视频片段")
                return {
                    'status': 'error',
                    'message': '视频处理失败：没有可用的音频片段'
                }

            # 标记播放列表为完成状态
            await self.task_state.hls_manager.finalize_playlist()
            return {'status': 'success'}

        except Exception as e:
            self.logger.error(f"视频处理失败: {str(e)}")
            return {
                'status': 'error',
                'message': f'处理失败: {str(e)}',
                'task_id': self.task_state.task_id
            }

    async def _process_segment(self, i: int, start: float, duration: float, is_first: bool = False, language:str="auto"):
        """处理单个分段"""
        try:
            # 使用 TaskPaths 中定义的 segments_dir
            segment_dir = self.task_state.task_paths.segments_dir / f"segment_{i}"
            segment_dir.mkdir(parents=True, exist_ok=True)
            
            # 分离当前分段的媒体文件并直接存储
            self.task_state.segment_media_files[i] = await self.media_utils.extract_segment(
                self.task_state.video_path,  # 使用 task_state.video_path 而不是 task_paths.video_path
                start,
                duration,
                self.task_state.task_paths.media_dir,
                i
            )
            
            # 语音识别
            sentences = self.sense_model.generate(
                input=self.task_state.segment_media_files[i]['vocals'],
                cache={},
                language=language,
                use_itn=True,
                batch_size_s=60,
                merge_vad=False
            )

            if not sentences:
                self.logger.warning(f"分段 {i} 未识别到句子")
                return

            # 为每个句子设置 ID 和分段信息
            for sentence in sentences:
                sentence.sentence_id = self.task_state.sentence_counter
                sentence.segment_index = i
                sentence.segment_start = start
                sentence.task_id = self.task_state.task_id  # 设置任务ID
                self.task_state.sentence_counter += 1
            
            self.logger.info(f"处理分段 {i} 完成，共 {len(sentences)} 个句子")

            # 如果是第一段，等待完整处理完成
            if is_first:
                self.task_state.mixing_complete = asyncio.Queue()
                await self.task_state.translation_queue.put(sentences)
                await self.task_state.mixing_complete.get()
                self.task_state.mixing_complete = None
            else:
                await self.task_state.translation_queue.put(sentences)

        except Exception as e:
            self.logger.error(f"处理视频段落 {i} 失败: {str(e)}")
            raise

async def main():
    vitrans = ViTranslator()
    try:
        video_path = "path/to/your/video.mp4"
        await vitrans.trans_video(video_path)
    except Exception as e:
        logger.error(f"处理失败: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())