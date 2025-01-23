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

        self.audio_separator = ClearVoiceSeparator(model_name='MossFormer2_SE_48K')
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
        self.cosyvoice_model = CosyVoice2("models/CosyVoice/pretrained_models/CosyVoice2-0.5B")
        self.target_sr = self.cosyvoice_model.sample_rate

        self.media_utils = MediaUtils(config=self.config, audio_separator=self.audio_separator, target_sr=self.target_sr)
        self.model_in = ModelIn(self.cosyvoice_model)
        self.tts_generator = TTSTokenGenerator(self.cosyvoice_model, Hz=25)
        self.audio_generator = AudioGenerator(self.cosyvoice_model, sample_rate=self.target_sr)

        # 选择翻译模型(此处演示 DeepSeekClient)
        translation_model = (self.config.TRANSLATION_MODEL or "deepseek").strip().lower()
        if translation_model == "deepseek":
            self.translator = Translator(DeepSeekClient(api_key=self.config.DEEPSEEK_API_KEY))
        else:
            raise ValueError(f"不支持的翻译模型：{translation_model}")

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
        """对外主函数：翻译一个视频文件"""
        self.logger.info(
            f"[trans_video] 开始处理视频: {video_path}, task_id={task_id}, target_language={target_language}"
        )

        # 1) 构建 TaskState
        task_state = TaskState(
            task_id=task_id,
            video_path=video_path,
            task_paths=task_paths,
            hls_manager=hls_manager,
            target_language=target_language
        )

        # 2) 创建 PipelineScheduler 并启动 Worker
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
            # 3) 分段
            duration = await self.media_utils.get_video_duration(video_path)
            segments = await self.media_utils.get_audio_segments(duration)
            self.logger.info(f"总长度={duration:.2f}s, 分段数={len(segments)}, 任务ID={task_id}")

            if not segments:
                self.logger.warning(f"没有可用分段 -> 任务ID={task_id}")
                await pipeline.stop_workers(task_state)
                return {"status": "error", "message": "无法获取有效分段"}

            # 4) 先处理第一段
            first_start, first_dur = segments[0]
            await self._process_segment(pipeline, task_state, 0, first_start, first_dur, is_first_segment=True)
            # 等第一段完成(如果需要)
            await pipeline.wait_first_segment_done(task_state)

            # 处理后续分段
            for i, (seg_start, seg_dur) in enumerate(segments[1:], start=1):
                await self._process_segment(pipeline, task_state, i, seg_start, seg_dur)

            # 5) 所有分段都投递后，停止 Worker
            await pipeline.stop_workers(task_state)

            # 6) HLS
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
        is_first_segment: bool = False
    ):
        """
        处理单个分段: 提取 -> ASR -> 推到 pipeline
        """
        self.logger.info(
            f"[_process_segment] 任务ID={task_state.task_id}, segment_index={segment_index}, start={start:.2f}, dur={seg_duration:.2f}"
        )
        media_files = await self.media_utils.extract_segment(
            video_path=task_state.video_path,
            start=start,
            duration=seg_duration,
            output_dir=task_state.task_paths.processing_dir,
            segment_index=segment_index
        )
        task_state.segment_media_files[segment_index] = media_files

        # ASR
        sentences = self.sense_model.generate(
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

        # 设置附加属性
        for s in sentences:
            s.segment_index = segment_index
            s.segment_start = start
            s.task_id = task_state.task_id
            s.sentence_id = task_state.sentence_counter
            task_state.sentence_counter += 1

        # 推送到 pipeline
        await pipeline.push_sentences_to_pipeline(task_state, sentences, is_first_segment)
