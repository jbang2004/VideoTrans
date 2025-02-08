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
from pipeline_scheduler import PipelineScheduler
from core.audio_separator import ClearVoiceSeparator
from core.model_in import ModelIn
from utils.task_storage import TaskPaths
from config import Config
from utils.task_state import TaskState

from utils.ffmpeg_utils import FFmpegTool
from core.segment_worker import SegmentWorker
from core.asr_worker import ASRWorker
from core.video_segmenter import VideoSegmenter

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

        # 初始化工具类
        self.ffmpeg_tool = FFmpegTool()
        self.video_segmenter = VideoSegmenter(config=self.config, ffmpeg_tool=self.ffmpeg_tool)

        # 初始化各个Worker
        self.segment_worker = SegmentWorker(
            config=self.config,
            audio_separator=self.audio_separator,
            video_segmenter=self.video_segmenter,
            ffmpeg_tool=self.ffmpeg_tool,
            target_sr=self.target_sr
        )
        self.asr_worker = ASRWorker(sense_model=self.sense_model)
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

        self.logger.info("[ViTranslator] 初始化完成")

    async def trans_video(
        self,
        video_path: str,
        task_id: str,
        task_paths: TaskPaths,
        hls_manager=None,
        target_language="zh",
        generate_subtitle: bool = False,
    ) -> Dict[str, Any]:
        """
        入口：对整段视频进行处理。包括分段、ASR、翻译、TTS、混音、生成 HLS 等。
        """
        self.logger.info(
            f"[trans_video] 开始处理视频: {video_path}, task_id={task_id}, "
            f"target_language={target_language}, generate_subtitle={generate_subtitle}"
        )

        # 初始化任务状态
        task_state = TaskState(
            task_id=task_id,
            video_path=video_path,
            task_paths=task_paths,
            hls_manager=hls_manager,
            target_language=target_language,
            generate_subtitle=generate_subtitle
        )

        # 初始化流水线
        pipeline = PipelineScheduler(
            segment_worker=self.segment_worker,
            asr_worker=self.asr_worker,
            translator=self.translator,
            model_in=self.model_in,
            tts_token_generator=self.tts_generator,
            duration_aligner=self.duration_aligner,
            audio_generator=self.audio_generator,
            timestamp_adjuster=self.timestamp_adjuster,
            mixer=self.mixer,
            config=self.config
        )

        try:
            # 1. 启动所有worker
            await pipeline.start_workers(task_state)

            # 2. 触发视频分段初始化
            await task_state.segment_init_queue.put({
                'video_path': video_path,
                'task_id': task_id
            })

            # 3. 等待所有worker完成
            await pipeline.stop_workers(task_state)

            # 4. 检查是否有错误
            if task_state.errors:
                error_msg = f"处理过程中发生 {len(task_state.errors)} 个错误"
                self.logger.error(f"{error_msg} -> TaskID={task_id}")
                return {"status": "error", "message": error_msg, "errors": task_state.errors}

            # 5. 如果有HLS Manager，标记完成
            if hls_manager and hls_manager.has_segments:
                await hls_manager.finalize_playlist()
                self.logger.info(f"[trans_video] HLS生成完成 -> TaskID={task_id}")

            # 6. 合并所有segment MP4
            final_video_path = await self._concat_segment_mp4s(task_state)
            if final_video_path is not None and final_video_path.exists():
                self.logger.info(f"翻译后的完整视频已生成: {final_video_path}")

                # 清理资源
                import torch
                torch.cuda.empty_cache()
                self.logger.info("已释放未使用的GPU显存")

                return {
                    "status": "success",
                    "message": "视频翻译完成",
                    "final_video_path": str(final_video_path)
                }
            else:
                return {
                    "status": "error",
                    "message": "无法合并生成最终MP4文件"
                }

        except Exception as e:
            self.logger.exception(f"[trans_video] 处理失败: {e} -> TaskID={task_id}")
            return {"status": "error", "message": str(e)}

        finally:
            # 确保清理资源
            if 'pipeline' in locals():
                await pipeline.stop_workers(task_state)

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
