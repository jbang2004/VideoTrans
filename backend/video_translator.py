import logging
from typing import List, Dict, Any
from pathlib import Path

from core.auto_sense import SenseAutoModel
from models.CosyVoice.cosyvoice.cli.cosyvoice import CosyVoice2
from core.translation.translator import Translator
from core.translation.deepseek_client import DeepSeekClient
from core.tts_token_gener import TTSTokenGenerator
from core.audio_gener import AudioGenerator
from core.timeadjust.duration_aligner import DurationAligner
from core.timeadjust.timestamp_adjuster import TimestampAdjuster
from core.media_mixer import MediaMixer
from utils.media_utils import MediaUtils
from pipeline_scheduler import PipelineScheduler
from core.audio_separator import ClearVoiceSeparator
from core.model_in import ModelIn
from utils.task_storage import TaskPaths
from config import Config
from utils.task_state import TaskState

logger = logging.getLogger(__name__)

class ViTranslator:
    """
    全局持有大模型(ASR/TTS/翻译)对象, 每次 trans_video 时创建新的 TaskState + PipelineScheduler
    """
    def __init__(self, config: Config = None):
        self.logger = logger
        self.config = config or Config()
        self._init_global_models()

    def _init_global_models(self):
        self.logger.info("[ViTranslator] 初始化模型和工具...")

        # 音频分离器
        self.audio_separator = ClearVoiceSeparator(model_name='MossFormer2_SE_48K')

        # ASR + VAD + Speaker
        self.sense_model = SenseAutoModel(
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

        # TTS 模型
        self.cosyvoice_model = CosyVoice2("models/CosyVoice/pretrained_models/CosyVoice2-0.5B")
        self.target_sr = self.cosyvoice_model.sample_rate

        # 媒体与管线相关工具
        self.media_utils = MediaUtils(config=self.config, audio_separator=self.audio_separator, target_sr=self.target_sr)
        self.model_in = ModelIn(self.cosyvoice_model)
        self.tts_generator = TTSTokenGenerator(self.cosyvoice_model, Hz=25)
        self.audio_generator = AudioGenerator(self.cosyvoice_model, sample_rate=self.target_sr)

        # 翻译模型
        translation_model = (self.config.TRANSLATION_MODEL or "deepseek").strip().lower()
        if translation_model == "deepseek":
            self.translator = Translator(DeepSeekClient(api_key=self.config.DEEPSEEK_API_KEY))
        else:
            raise ValueError(f"不支持的翻译模型：{translation_model}")

        # 其他处理
        self.duration_aligner = DurationAligner(
            model_in=self.model_in,
            simplifier=self.translator,
            tts_token_gener=self.tts_generator,
            max_speed=1.2
        )
        self.timestamp_adjuster = TimestampAdjuster(sample_rate=self.target_sr)
        self.mixer = MediaMixer(config=self.config, sample_rate=self.target_sr)

        self.logger.info("[ViTranslator] 初始化完成")

    async def trans_video(
        self,
        video_path: str,
        task_id: str,
        task_paths: TaskPaths,
        hls_manager=None,
        target_language="zh"
    ) -> Dict[str, Any]:
        """
        入口：对整段视频进行处理。包括分段、ASR、翻译、TTS、混音、生成 HLS 等。
        """
        self.logger.info(
            f"[trans_video] 开始处理视频: {video_path}, task_id={task_id}, target_language={target_language}"
        )

        # 初始化任务状态 + 管线
        task_state = TaskState(
            task_id=task_id,
            video_path=video_path,
            task_paths=task_paths,
            hls_manager=hls_manager,
            target_language=target_language
        )

        pipeline = PipelineScheduler(
            translator=self.translator,
            model_in=self.model_in,
            tts_token_generator=self.tts_generator,
            duration_aligner=self.duration_aligner,
            audio_generator=self.audio_generator,
            timestamp_adjuster=self.timestamp_adjuster,
            mixer=self.mixer,
            config=self.config
        )
        await pipeline.start_workers(task_state)

        try:
            # 1. 获取视频总时长
            duration = await self.media_utils.get_video_duration(video_path)
            # 2. 划分分段
            segments = await self.media_utils.get_audio_segments(duration)
            self.logger.info(f"总长度={duration:.2f}s, 分段数={len(segments)}, 任务ID={task_id}")

            if not segments:
                self.logger.warning(f"没有可用分段 -> 任务ID={task_id}")
                await pipeline.stop_workers(task_state)
                return {"status": "error", "message": "无法获取有效分段"}

            # 3. 遍历所有分段：提取、ASR、推送后续流水线
            for i, (seg_start, seg_dur) in enumerate(segments):
                await self._process_segment(pipeline, task_state, i, seg_start, seg_dur)

            # 4. 所有段结束后，停止流水线
            await pipeline.stop_workers(task_state)

            # 5. 如果有 HLS Manager，标记完成
            if hls_manager and hls_manager.has_segments:
                await hls_manager.finalize_playlist()
                self.logger.info(f"[trans_video] 任务ID={task_id} 完成并已生成HLS。")
                return {"status": "success", "message": "视频翻译完成"}
            else:
                self.logger.warning(f"任务ID={task_id} 没有可用片段写入HLS")
                return {"status": "error", "message": "没有成功写入HLS片段"}

        except Exception as e:
            self.logger.exception(f"[trans_video] 任务ID={task_id} 出错: {e}")
            return {"status": "error", "message": str(e)}

    async def _process_segment(
        self,
        pipeline: PipelineScheduler,
        task_state: TaskState,
        segment_index: int,
        start: float,
        seg_duration: float,
    ):
        """
        用于处理单个分段：提取音频/视频 -> ASR -> 推送后续翻译/合成的异步流水线
        """
        self.logger.info(
            f"[_process_segment] 任务ID={task_state.task_id}, segment_index={segment_index}, "
            f"start={start:.2f}, dur={seg_duration:.2f}"
        )
        # 1) 提取视频/音频/人声/背景音
        media_files = await self.media_utils.extract_segment(
            video_path=task_state.video_path,
            start=start,
            duration=seg_duration,
            output_dir=task_state.task_paths.processing_dir,
            segment_index=segment_index
        )
        task_state.segment_media_files[segment_index] = media_files

        # 2) 进行ASR（异步包装）
        sentences = await self.sense_model.generate_async(
            input=media_files['vocals'],
            cache={},
            language="auto",
            use_itn=True,
            batch_size_s=60,
            merge_vad=False
        )
        self.logger.info(f"[_process_segment] ASR识别到 {len(sentences)} 条句子, seg={segment_index}, TaskID={task_state.task_id}")

        if not sentences:
            return

        # 3) 给每条句子打上分段相关信息，再放入后续流水线
        for s in sentences:
            s.segment_index = segment_index
            s.segment_start = start
            s.task_id = task_state.task_id
            s.sentence_id = task_state.sentence_counter
            task_state.sentence_counter += 1

        await pipeline.push_sentences_to_pipeline(task_state, sentences)
