import ray
from ray import serve
import logging
import torch
import numpy as np
from typing import Dict, Union, Tuple, Optional
from pathlib import Path
import time
import asyncio
from config import Config
import os
import soundfile as sf

from utils.ffmpeg_utils import extract_audio, extract_video
from models.ClearerVoice_Minimal.audio_enhancer import AudioEnhancer
from core.supabase_client import SupabaseClient

@serve.deployment(
    name="video_separator",
    ray_actor_options={"num_cpus": 1, "num_gpus": 0.3},
    logging_config={"log_level": "INFO"}
)
class VideoSeparator:
    """
    视频分离器，负责分割视频片段、提取音频并分离人声和背景音乐
    """
    def __init__(self, model_name='MossFormer2_SE_48K'):
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"初始化视频分离器: {model_name}")
        # 初始化音频增强器，并立即加载模型
        self.audio_enhancer = AudioEnhancer(model_name=model_name)
        self.config = Config()
        self.supabase_client = SupabaseClient(config=self.config)
    
    async def separate_video(
        self,
        video_path: str,
        output_dir: str,
        video_width: int,
        video_height: int,
        task_id: Optional[str] = None
    ) -> Dict[str, Union[str, float, int]]:
        """
        提取视频片段，分离人声和背景音乐
        
        Args:
            video_path: 视频文件路径
            output_dir: 输出目录
            video_width: 视频宽度
            video_height: 视频高度
            task_id: 任务ID，用于更新数据库状态
            
        Returns:
            Dict[str, Union[str, float, int]]: 包含分离后文件路径和视频尺寸的字典
            {
                'silent_video_path': 无声视频路径,
                'vocals_audio_path': 人声音频路径,
                'background_audio_path': 背景音乐路径,
                'video_width': 视频宽度,
                'video_height': 视频高度
            }
        """
        start_time = time.time()
        
        try:
            # 创建临时目录
            output_dir_path = Path(output_dir)
            output_dir_path.mkdir(parents=True, exist_ok=True)
            
            silent_video = str(output_dir_path / "video_silent.mp4")
            full_audio = str(output_dir_path / "audio_full.wav")
            vocals_audio = str(output_dir_path / "vocals.wav")
            background_audio = str(output_dir_path / "background.wav")

            # (0) 使用传入的 video_width 和 video_height
            self.logger.info(f"[{task_id if task_id else 'VideoSeparator'}] Using provided video resolution: {video_width}x{video_height}")

            # (1) 提取音频 & 视频（整段）
            # 获取目标采样率
            target_sr = self.config.TARGET_SR
            
            # 应用了音频预处理的提取
            await extract_audio(video_path, full_audio)
            
            # 提取无声视频
            await extract_video(video_path, silent_video)

            # (1.1) 对全音频进行归一化处理，但不改变采样率
            try:
                await asyncio.to_thread(
                    self._normalize_and_resample,
                    full_audio,
                    save_to_file=True  # 不指定target_sr，保持原采样率
                )
                self.logger.info("已对原始音频进行归一化处理")
            except Exception as e:
                self.logger.warning(f"原始音频归一化失败: {e}")

            # (2) 音频分离逻辑 - 使用AudioEnhancer进行语音增强
            self.logger.info(f"开始使用AudioEnhancer进行语音增强和背景分离")
            
            # 使用AudioEnhancer进行语音增强和背景分离
            success = await asyncio.to_thread(
                self.audio_enhancer.enhance_audio,
                input_path=full_audio,
                enhanced_path=vocals_audio,
                noise_path=background_audio
            )
            
            if success:
                self.logger.info("AudioEnhancer语音增强完成")
            else:
                self.logger.error("AudioEnhancer语音增强失败")
            
            if not success:
                self.logger.error("音频处理失败")
                # 更新任务状态
                if task_id:
                    asyncio.create_task(self.supabase_client.update_task(task_id, {
                        'status': 'error', 
                        'error_message': 'Video separation failed or no vocals detected'
                    }))
                return {}

            # (2.1) 处理背景音频：直接使用 _normalize_and_resample 方法处理文件
            try:
                # 直接传入背景音频文件路径和目标采样率
                await asyncio.to_thread(
                    self._normalize_and_resample,
                    background_audio,
                    target_sr=target_sr,
                    save_to_file=True
                )
                
                self.logger.info(f"背景音频已处理: 采样率={target_sr}Hz")
                    
            except Exception as e:
                self.logger.warning(f"背景音频处理失败: {e}")
                
            # (2.2) 处理人声音频：重采样为16000Hz
            try:
                # 处理人声音频
                await asyncio.to_thread(
                    self._normalize_and_resample,
                    vocals_audio,
                    target_sr=16000,  # 人声固定重采样到16000Hz
                    save_to_file=True
                )
                
                self.logger.info("人声音频已处理: 采样率=16000Hz")
                    
            except Exception as e:
                self.logger.warning(f"人声音频处理失败: {e}")
                
            # (3) 清理临时文件
            if Path(full_audio).exists():
                await asyncio.to_thread(os.remove, full_audio)

            # 返回结果
            media_files = {
                'silent_video_path': silent_video,
                'vocals_audio_path': vocals_audio,
                'background_audio_path': background_audio,
                'video_width': video_width,
                'video_height': video_height
            }

            # 更新媒体文件路径到数据库
            if task_id:
                try:
                    asyncio.create_task(self.supabase_client.update_task(task_id, media_files))
                    self.logger.info(f"[{task_id}] 已异步更新媒体文件路径到数据库")
                except Exception as e:
                    self.logger.warning(f"[{task_id}] 更新媒体文件路径到数据库失败: {e}")
            
            self.logger.debug(f"separate_video 完成，耗时 {time.time() - start_time:.2f}s")
            return media_files
            
        except Exception as e:
            self.logger.error(f"separate_video 执行出错，耗时 {time.time() - start_time:.2f}s, 错误: {e}")
            # 更新错误状态到数据库
            if task_id:
                try:
                    asyncio.create_task(self.supabase_client.update_task(task_id, {
                        'status': 'error', 
                        'error_message': f"Video separation error: {e}"
                    }))
                except Exception as e2:
                    self.logger.warning(f"[{task_id}] 更新错误状态到数据库失败: {e2}")
            raise
        finally:
            # 清理 GPU 缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
    def _normalize_and_resample(
        self,
        audio_input: Union[Tuple[int, np.ndarray], np.ndarray, str],
        target_sr: Optional[int] = None,
        save_to_file: bool = False
    ) -> np.ndarray:
        """
        重采样和归一化音频 - 同步方法
        
        Args:
            audio_input: 音频数据，可以是:
                         - 元组(采样率, 音频数据)
                         - 音频数据数组
                         - 音频文件路径
            target_sr: 目标采样率
            save_to_file: 是否将处理后的数据保存回原文件(当audio_input是文件路径时有效)
        
        Returns:
            np.ndarray: 处理后的音频数据
        """
        import torch
        import torchaudio
        
        resampled_audio = None
        original_file_path = None
        
        try:
            # 如果输入是文件路径，先读取数据
            if isinstance(audio_input, str):
                original_file_path = audio_input
                # 使用torchaudio代替soundfile读取音频
                try:
                    waveform, fs = torchaudio.load(audio_input)
                    # torchaudio返回的波形形状是 [channels, samples]
                    # 转为numpy数组，并且确保是单声道
                    audio_data = waveform.mean(dim=0).numpy() if waveform.shape[0] > 1 else waveform[0].numpy()
                except Exception as e:
                    self.logger.error(f"使用torchaudio读取音频失败: {e}")
                    raise
            elif isinstance(audio_input, tuple):
                fs, audio_data = audio_input
            else:
                fs = target_sr
                audio_data = audio_input

            # 确保音频数据是numpy数组
            if not isinstance(audio_data, np.ndarray):
                audio_data = np.asarray(audio_data, dtype=np.float32)
                
            # 转换为float32类型
            audio_data = audio_data.astype(np.float32)

            # 归一化音频
            max_val = np.abs(audio_data).max()
            if max_val > 0:
                audio_data = audio_data / max_val

            # 如果源采样率与目标采样率不一致, 用 torchaudio 进行重采样
            if fs != target_sr and target_sr is not None:
                audio_data = np.ascontiguousarray(audio_data)
                resampler = torchaudio.transforms.Resample(
                    orig_freq=fs,
                    new_freq=target_sr,
                    dtype=torch.float32
                )
                audio_tensor = torch.from_numpy(audio_data)[None, :]
                resampled_audio = resampler(audio_tensor)[0].numpy()
                
                # 删除临时张量
                del audio_tensor
                del resampler
                
                result_audio = resampled_audio
            else:
                result_audio = audio_data
            
            # 如果需要保存回文件
            if save_to_file and original_file_path:
                out_sr = target_sr if target_sr is not None else fs
                # 使用torchaudio保存音频
                audio_tensor = torch.from_numpy(result_audio).unsqueeze(0)  # [samples] -> [1, samples]
                torchaudio.save(original_file_path, audio_tensor, out_sr)
                del audio_tensor
            
            return result_audio
            
        except Exception as e:
            self.logger.error(f"音频重采样和归一化失败: {e}")
            raise
        finally:
            # 清理临时变量
            if resampled_audio is not None and resampled_audio is not audio_data:
                try:
                    del resampled_audio
                except:
                    pass
                
            # 确保GPU缓存被清理
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
