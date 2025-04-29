import ray
from ray import serve
import logging
import torch
import numpy as np
import soundfile as sf
from typing import Tuple, Dict, Union, Optional
from pathlib import Path
import time
import asyncio
from config import Config

from models.ClearerVoice.clearvoice import ClearVoice
from utils.ffmpeg_utils import extract_audio, extract_video

@serve.deployment(
    name="video_separator",
    ray_actor_options={"num_gpus": 0.3, "num_cpus": 0.5}
)
class VideoSeparator:
    """
    视频分离器，负责分割视频片段、提取音频并分离人声和背景音乐
    """
    def __init__(self, model_name='MossFormer2_SE_48K'):
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"初始化视频分离器: {model_name}")
        
        self.model_name = model_name
        self.clearvoice = ClearVoice(
            task='speech_enhancement',
            model_names=[model_name]
        )
        self.config = Config()
    
    async def separate_audio(self, input_path: str) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        分离音频中的人声和背景音
        
        Args:
            input_path: 输入音频文件路径
            
        Returns:
            Tuple[np.ndarray, np.ndarray, int]: (人声音频, 背景音频, 采样率)
        """
        # 使用asyncio.to_thread包装同步调用
        try:
            enhanced_audio, background_audio = await asyncio.to_thread(
                self.clearvoice,
                input_path=input_path,
                online_write=False,
                extract_noise=True
            )
            
            if self.model_name.endswith('16K'):
                sr = 16000
            elif self.model_name.endswith('48K'):
                sr = 48000
            else:
                sr = 48000
            
            return enhanced_audio, background_audio, sr
        except Exception as e:
            self.logger.error(f"音频分离失败: {e}, 输入路径: {input_path}")
            raise
        finally:
            # 主动清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    async def separate_video(
        self,
        video_path: str,
        output_dir: str,
        target_sr: int
    ) -> Dict[str, Union[str, float]]:
        """
        提取视频片段，分离人声和背景音乐
        
        Args:
            video_path: 视频文件路径
            output_dir: 输出目录
            target_sr: 目标采样率
            
        Returns:
            Dict[str, Union[str, float]]: 包含分离后文件路径的字典
            {
                'video': 无声视频路径,
                'vocals': 人声音频路径,
                'background': 背景音乐路径,
                'duration': 实际片段时长
            }
        """
        start_time = time.time()
        temp_files = {}
        vocals = None
        background = None
        
        try:
            # 创建临时目录
            output_dir_path = Path(output_dir)
            output_dir_path.mkdir(parents=True, exist_ok=True)
            
            silent_video = str(output_dir_path / "video_silent.mp4")
            full_audio = str(output_dir_path / "audio_full.wav")
            vocals_audio = str(output_dir_path / "vocals.wav")
            background_audio = str(output_dir_path / "background.wav")

            # (1) 提取音频 & 视频（整段）
            await extract_audio(video_path, full_audio)
            await extract_video(video_path, silent_video)

            # (2) 分离人声
            vocals, background, sr = await self.separate_audio(full_audio)

            # 检查是否成功分离
            if vocals is None or background is None:
                self.logger.error("音频分离失败，未能产生有效的vocals或background")
                return {}

            # (3) 重采样和归一化 - 使用asyncio.to_thread包装同步函数调用
            background_resampled = await asyncio.to_thread(self._normalize_and_resample, (sr, background), target_sr)
            
            # 显式删除原始background以节省内存
            if background is not vocals:  # 确保它们不是同一个对象
                del background
                background = None
                
            # 写入人声/背景音频 - 使用asyncio.to_thread避免阻塞
            await asyncio.to_thread(sf.write, vocals_audio, vocals, sr, subtype='FLOAT')
            await asyncio.to_thread(sf.write, background_audio, background_resampled, target_sr, subtype='FLOAT')
            
            # 删除重采样后的背景音频（因为已写入文件）
            del background_resampled

            segment_duration = len(vocals) / sr

            # 删除原始整段音频
            full_audio_path = Path(full_audio)
            if full_audio_path.exists():
                await asyncio.to_thread(full_audio_path.unlink, missing_ok=True)

            temp_files = {
                'video': silent_video,
                'vocals': vocals_audio,
                'background': background_audio,
                'duration': segment_duration
            }

            # 返回临时文件路径
            elapsed = time.time() - start_time
            self.logger.debug(f"separate_video 完成，耗时 {elapsed:.2f}s")
            return temp_files
            
        except Exception as e:
            elapsed = time.time() - start_time
            self.logger.error(f"separate_video 执行出错，耗时 {elapsed:.2f}s, 错误: {e}")
            
            # 清理已生成的临时文件
            for file_path in temp_files.values():
                if isinstance(file_path, str) and Path(file_path).exists():
                    try:
                        await asyncio.to_thread(Path(file_path).unlink)
                    except Exception as clean_error:
                        self.logger.error(f"清理临时文件失败: {clean_error}, 文件: {file_path}")
            
            raise
        finally:
            # 确保大型音频数据被清理
            large_variables = ['vocals', 'background']
            for var_name in large_variables:
                if var_name in locals() and locals()[var_name] is not None:
                    del locals()[var_name]
                
            # 添加GPU缓存清理
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def _normalize_and_resample(
        self,
        audio_input: Union[Tuple[int, np.ndarray], np.ndarray],
        target_sr: int = None
    ) -> np.ndarray:
        """
        重采样和归一化音频 - 同步方法
        
        Args:
            audio_input: 音频数据，可以是元组(采样率, 音频数据)或者直接是音频数据
            target_sr: 目标采样率
        
        Returns:
            np.ndarray: 处理后的音频数据
        """
        import torch
        import torchaudio
        
        resampled_audio = None
        
        try:
            if isinstance(audio_input, tuple):
                fs, audio_data = audio_input
            else:
                fs = target_sr
                audio_data = audio_input

            audio_data = audio_data.astype(np.float32)

            max_val = np.abs(audio_data).max()
            if max_val > 0:
                audio_data = audio_data / max_val

            # 如果多通道, 转单通道
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=-1)

            # 如果源采样率与目标采样率不一致, 用 torchaudio 进行重采样
            if fs != target_sr:
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
                
                return resampled_audio
            
            return audio_data
            
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