# ---------------------------------
# backend/video_translator.py (节选完整示例)
# ---------------------------------
import logging
import os
from typing import List, Dict, Any
from pathlib import Path

from core.auto_sense import SenseAutoModel
from models.CosyVoice.cosyvoice.cli.cosyvoice import CosyVoice2
from core.translation.translator import Translator
from core.translation.deepseek_client import DeepSeekClient
from core.translation.gemini_client import GeminiClient
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

from utils.ffmpeg_utils import FFmpegTool

logger = logging.getLogger(__name__)

class ViTranslator:
    """
    全局持有大模型(ASR/TTS/翻译)对象, ...
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

        # 其他核心工具
        self.media_utils = MediaUtils(config=self.config, audio_separator=self.audio_separator, target_sr=self.target_sr)
        self.model_in = ModelIn(self.cosyvoice_model)
        self.tts_generator = TTSTokenGenerator(self.cosyvoice_model, Hz=25)
        self.audio_generator = AudioGenerator(self.cosyvoice_model, sample_rate=self.target_sr)

        # 翻译
        translation_model = (self.config.TRANSLATION_MODEL or "deepseek").strip().lower()
        if translation_model == "deepseek":
            self.translator = Translator(DeepSeekClient(api_key=self.config.DEEPSEEK_API_KEY))
        elif translation_model == "gemini":
            self.translator = Translator(GeminiClient(api_key=self.config.GEMINI_API_KEY))
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

        self.ffmpeg_tool = FFmpegTool()
        self.logger.info("[ViTranslator] 初始化完成")

    async def trans_video(
        self,
        video_path: str,
        task_id: str,
        task_paths: TaskPaths,
        hls_manager=None,
        target_language="zh",
        # =========== (新增) ===========
        generate_subtitle: bool = False,
    ) -> Dict[str, Any]:
        """
        入口：对整段视频进行处理。包括分段、ASR、翻译、TTS、混音、生成 HLS 等。
        generate_subtitle: 是否需要在最终生成的视频里烧制字幕
        """
        self.logger.info(
            f"[trans_video] 开始处理视频: {video_path}, task_id={task_id}, target_language={target_language}, generate_subtitle={generate_subtitle}"
        )

        # 初始化任务状态 + 管线
        task_state = TaskState(
            task_id=task_id,
            video_path=video_path,
            task_paths=task_paths,
            hls_manager=hls_manager,
            target_language=target_language,
            # =========== (新增) ===========
            generate_subtitle=generate_subtitle
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

            # 6. 现在合并 `_mixing_worker` 产出的所有 segment_xxx.mp4
            #    并在成功后自动删除它们
            final_video_path = await self._concat_segment_mp4s(task_state)
            if final_video_path is not None and final_video_path.exists():
                self.logger.info(f"翻译后的完整视频已生成: {final_video_path}")
                
                # 添加清理逻辑：
                import torch
                torch.cuda.empty_cache()
                self.logger.info("调用 torch.cuda.empty_cache()，已释放未使用的 GPU 显存")
                
                # 如果有临时目录需要清理（例如 task_state.task_paths 里存放了临时文件），可以进行删除：
                # import shutil
                # shutil.rmtree(task_state.task_paths.temp_dir, ignore_errors=True)
                # self.logger.info("已清理视频处理临时目录")

                return {
                    "status": "success",
                    "message": "视频翻译完成",
                    "final_video_path": str(final_video_path)
                }
            else:
                self.logger.warning("无法合并生成最终MP4文件")
                return {"status": "error", "message": "HLS完成，但无法合并出最终MP4"}

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
        # 1. 提取并分离人声/背景
        media_files = await self.media_utils.extract_segment(
            video_path=task_state.video_path,
            start=start,
            duration=seg_duration,
            output_dir=task_state.task_paths.processing_dir,
            segment_index=segment_index
        )
        task_state.segment_media_files[segment_index] = media_files

        # 2. ASR
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

        for s in sentences:
            s.segment_index = segment_index
            s.segment_start = start
            s.task_id = task_state.task_id
            s.sentence_id = task_state.sentence_counter
            task_state.sentence_counter += 1

        await pipeline.push_sentences_to_pipeline(task_state, sentences)

    async def _concat_segment_mp4s(self, task_state: TaskState) -> Path:
        """
        把 pipeline_scheduler _mixing_worker 产出的所有 segment_xxx.mp4
        用 ffmpeg concat 合并成 final_{task_state.task_id}.mp4
        如果成功再删除这些小片段。
        """
        if not task_state.merged_segments:
            self.logger.warning("无可合并的 segment MP4, 可能任务中断或没有生成混音段.")
            return None

        final_path = task_state.task_paths.output_dir / f"final_{task_state.task_id}.mp4"
        final_path.parent.mkdir(parents=True, exist_ok=True)

        list_txt = final_path.parent / f"concat_{task_state.task_id}.txt"
        with open(list_txt, 'w', encoding='utf-8') as f:
            for seg_mp4 in task_state.merged_segments:
                abs_path = Path(seg_mp4).resolve()
                f.write(f"file '{abs_path}'\n")

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(list_txt),
            "-c", "copy",
            str(final_path)
        ]
        try:
            self.logger.info(f"开始合并 {len(task_state.merged_segments)} 个MP4 -> {final_path}")
            await self.ffmpeg_tool.run_command(cmd)
            self.logger.info(f"合并完成: {final_path}")

            # 合并成功后，自动删除这些 segment
            for seg_mp4 in task_state.merged_segments:
                try:
                    Path(seg_mp4).unlink(missing_ok=True)
                    self.logger.debug(f"已删除分段文件: {seg_mp4}")
                except Exception as ex:
                    self.logger.warning(f"删除分段文件 {seg_mp4} 失败: {ex}")

            return final_path
        except Exception as e:
            self.logger.error(f"ffmpeg concat 失败: {e}")
            return None
        finally:
            if list_txt.exists():
                list_txt.unlink()
