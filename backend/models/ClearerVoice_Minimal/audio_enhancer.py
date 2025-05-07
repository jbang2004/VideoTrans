import os
import pathlib
import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
# 使用相对于文件的路径导入ClearVoice，确保无论从哪里调用都能正确导入
import sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))
from clearvoice import ClearVoice


class AudioEnhancer:
    """
    音频增强器类：提供音频处理和增强功能
    """
    def __init__(self, model_name='MossFormer2_SE_48K'):
        """
        初始化音频增强器
        
        参数:
            model_name: 要加载的模型名称，默认为'MossFormer2_SE_48K'
        """
        # 获取项目根目录（当前文件所在目录）
        self.script_dir = pathlib.Path(__file__).parent.absolute()
        self.checkpoints_dir = os.path.join(self.script_dir, "checkpoints")
        
        # 支持的SE模型列表
        self.supported_models = ['MossFormer2_SE_48K', 'FRCRN_SE_16K', 'MossFormerGAN_SE_16K']
        
        # 验证模型名称
        if model_name not in self.supported_models:
            raise ValueError(f"不支持的模型: {model_name}。支持的模型有: {', '.join(self.supported_models)}")
        
        # 设置当前模型名称
        self.model_name = model_name
        
        # 直接加载模型
        print(f"正在加载模型: {model_name}...")
        
        # 创建模型实例
        self.model = ClearVoice(task='speech_enhancement', model_names=[model_name])
        
        # 设置checkpoint路径
        model_checkpoint_dir = os.path.join(self.checkpoints_dir, model_name)
        self.model.models[0].args.checkpoint_dir = model_checkpoint_dir
        
        print(f"模型 {model_name} 加载完成")
    
    def write_audio_file(self, audio_data, output_path, sample_rate, channels=1, sample_width=2):
        """
        将音频数据写入文件
        
        参数:
            audio_data: numpy数组格式的音频数据
            output_path: 输出文件路径
            sample_rate: 采样率
            channels: 通道数
            sample_width: 采样位宽 (2=16bit, 4=32bit)
        """
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 调整音频格式
        if not isinstance(audio_data, np.ndarray):
            audio_data = np.array(audio_data)
        
        # 处理多维数组
        if len(audio_data.shape) > 1:
            if audio_data.shape[0] == 1:  # 如果是单通道格式 [1, samples]
                audio_data = audio_data[0, :]
            elif audio_data.shape[0] == 2 and channels == 2:  # 如果是双通道格式 [2, samples]
                left_channel = audio_data[0, :]
                right_channel = audio_data[1, :]
                audio_data = np.vstack((left_channel, right_channel)).T
        
        # 设置音频位深
        if sample_width == 4:  # 32位浮点
            MAX_WAV_VALUE = 2147483648.0
            np_type = np.int32
        else:  # 16位整数
            MAX_WAV_VALUE = 32768.0
            np_type = np.int16
        
        # 调整音频幅度并转换类型
        audio_data = audio_data * MAX_WAV_VALUE
        audio_data = audio_data.astype(np_type)
        
        # 使用AudioSegment创建音频段并导出
        audio_segment = AudioSegment(
            audio_data.tobytes(),  # 原始音频数据
            frame_rate=sample_rate,  # 采样率
            sample_width=sample_width,  # 采样位宽
            channels=channels  # 通道数
        )
        
        # 获取文件扩展名并导出
        ext = os.path.splitext(output_path)[1][1:].lower()
        audio_format = 'ipod' if ext in ['m4a', 'aac'] else ext if ext else 'wav'
        audio_segment.export(output_path, format=audio_format)
    
    def enhance_audio(self, input_path, enhanced_path, noise_path, model_name=None):
        """
        语音增强：对输入音频进行增强，并分离出背景噪声
        
        参数:
            input_path: 输入音频文件路径
            enhanced_path: 增强音频输出路径
            noise_path: 背景噪声输出路径
            model_name: 已废弃参数，保留是为了向后兼容
        
        返回:
            成功与否的布尔值
        """
        try:
            # 如果传入model_name且与实例化时不同，给出警告
            if model_name is not None and model_name != self.model_name:
                print(f"警告: 当前实例已加载模型 {self.model_name}，忽略参数中的模型名称 {model_name}")
            
            # 确保输出目录存在
            os.makedirs(os.path.dirname(enhanced_path), exist_ok=True)
            os.makedirs(os.path.dirname(noise_path), exist_ok=True)
            
            # 确定采样率
            model_sr = 16000 if "16K" in self.model_name else 48000
            
            # 预处理音频 (转为单声道)
            audio_arr, sr0 = librosa.load(input_path, sr=None, mono=True)
            
            # 创建临时文件
            temp_dir = os.path.dirname(input_path)
            temp_mono_path = os.path.join(temp_dir, f"temp_mono_{os.path.basename(input_path)}")
            sf.write(temp_mono_path, audio_arr, sr0)
            
            # 执行语音增强
            output_audio = self.model(input_path=temp_mono_path, online_write=False)
            
            # 提取增强音频
            if isinstance(output_audio, dict) and 'enhanced' in output_audio:
                enhanced = output_audio['enhanced']
                # 如果模型同时返回了噪声
                noise = output_audio.get('noise', None)
            else:
                # 处理其他返回格式
                if isinstance(output_audio, dict):
                    enhanced = next(iter(output_audio.values()))
                else:
                    enhanced = output_audio
                noise = None
            
            # 保存增强音频
            self.write_audio_file(enhanced, enhanced_path, model_sr)
            
            # 处理噪声音频
            if noise is None:
                # 如果模型没有返回噪声，手动计算
                orig, _ = librosa.load(temp_mono_path, sr=model_sr, mono=True)
                enh, _ = librosa.load(enhanced_path, sr=model_sr, mono=True)
                min_len = min(len(orig), len(enh))
                noise = orig[:min_len] - enh[:min_len]
                noise = noise.reshape(1, -1)  # 调整形状
            
            # 保存噪声音频
            self.write_audio_file(noise, noise_path, model_sr)
            
            # 清理临时文件
            if os.path.exists(temp_mono_path):
                os.remove(temp_mono_path)
                
            return True
        
        except Exception as e:
            print(f"增强音频时出错: {e}")
            return False 