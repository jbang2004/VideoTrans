# ---------------------------------
# backend/video_translator.py (节选完整示例)
# ---------------------------------
import logging
from typing import Dict, Any
from pathlib import Path

from pipeline_scheduler import PipelineScheduler
from utils.task_storage import TaskPaths
from config import Config
from utils.task_state import TaskState
from utils.ffmpeg_utils import FFmpegTool

# 从各 worker 文件夹导入 Worker 类
from workers.asr_worker.worker import ASRWorker
from workers.segment_worker.worker import SegmentWorker
from workers.audio_gen_worker.worker import AudioGenWorker
from workers.mixer_worker.worker import MixerWorker
from workers.translation_worker.worker import TranslationWorker
from workers.modelin_worker.worker import ModelInWorker
from workers.tts_worker.worker import TTSTokenWorker
from workers.duration_worker.worker import DurationWorker

logger = logging.getLogger(__name__)

class ViTranslator:
    """
    视频翻译器：初始化所有模型、工具以及各个 Worker，通过 PipelineScheduler 协调整个处理流程。
    """

    def __init__(self, config: Config = None):
        self.logger = logger
        self.config = config or Config()
        self._init_global_models()

    def _init_global_models(self):
        self.logger.info("[ViTranslator] 初始化模型和工具...")
        
        # 初始化各 Worker 实例
        self.segment_worker = SegmentWorker(config=self.config)
        self.asr_worker = ASRWorker(config=self.config)
        self.translation_worker = TranslationWorker(config=self.config)
        self.modelin_worker = ModelInWorker(config=self.config)
        self.tts_token_worker = TTSTokenWorker(config=self.config)
        self.duration_worker = DurationWorker(config=self.config)
        self.audio_gen_worker = AudioGenWorker(config=self.config)
        self.mixer_worker = MixerWorker(config=self.config)

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
            task_paths=task_paths,
            hls_manager=hls_manager,
            target_language=target_language,
            generate_subtitle=generate_subtitle
        )

        # 初始化流水线
        pipeline = PipelineScheduler(
            segment_worker=self.segment_worker,
            asr_worker=self.asr_worker,
            translation_worker=self.translation_worker,
            modelin_worker=self.modelin_worker,
            tts_token_worker=self.tts_token_worker,
            duration_worker=self.duration_worker,
            audio_gen_worker=self.audio_gen_worker,
            mixer_worker=self.mixer_worker,
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

            # 6. 合并所有segment MP4s
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
            self.logger.warning("无可合并的 segment MP4，可能任务中断或没有生成混音段.")
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
            self.logger.info(f"开始合并 {len(task_state.merged_segments)} 个 MP4 -> {final_path}")
            # 使用 FFmpegTool 实例
            ffmpeg_tool = FFmpegTool()
            await ffmpeg_tool.run_command(cmd)
            self.logger.info(f"合并完成: {final_path}")

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

if __name__ == '__main__':
    print("ViTranslator 模块加载成功")
