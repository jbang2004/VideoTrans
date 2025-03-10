import logging
import numpy as np
import ray

@ray.remote(num_cpus=0.1)
def generate_audio(sentences, cosyvoice_actor, sample_rate=None):
    """
    生成音频（Ray Task版本）
    
    Args:
        sentences: 要处理的句子列表
        cosyvoice_actor: CosyVoice模型Actor引用
        sample_rate: 采样率，如果为None则使用Actor的采样率
        
    Returns:
        List[Sentence]: 处理后的句子列表
    """
    logger = logging.getLogger("audio_generator")
    
    if not sentences:
        logger.warning("generate_audio: 收到空的句子列表")
        return sentences
    
    # 获取采样率（如果未指定）
    if sample_rate is None:
        sample_rate = ray.get(cosyvoice_actor.get_sample_rate.remote())
    
    logger.info(f"开始处理 {len(sentences)} 个句子的音频生成")
    
    try:
        # 处理句子列表
        sentences_with_audio = []
        for sentence in sentences:
            try:
                # 获取所需参数
                model_input = sentence.model_input
                tts_token_id = model_input.get('tts_token_id')
                speaker_feature_id = model_input.get('speaker_feature_id')
                
                if not tts_token_id or not speaker_feature_id:
                    logger.info(f"缺少必要的参数，仅生成空波形 (TTS Token ID: {tts_token_id})")
                    sentence.generated_audio = np.zeros(0, dtype=np.float32)
                    sentences_with_audio.append(sentence)
                    continue
                
                # 获取语速
                speed = sentence.speed if sentence.speed else 1.0
                
                # 调用Actor生成音频，使用缓存的特征
                audio = ray.get(cosyvoice_actor.generate_audio.remote(
                    tts_token_id, speaker_feature_id, speed
                ))
                
                # 添加首句静音
                if sentence.is_first and sentence.start > 0:
                    silence_samples = int(sentence.start * sample_rate / 1000)
                    audio = np.concatenate([np.zeros(silence_samples, dtype=np.float32), audio])
                
                # 添加尾部静音
                if sentence.silence_duration > 0:
                    silence_samples = int(sentence.silence_duration * sample_rate / 1000)
                    audio = np.concatenate([audio, np.zeros(silence_samples, dtype=np.float32)])
                
                # 更新句子
                sentence.generated_audio = audio
                logger.info(f"音频生成完成 (TTS Token ID: {tts_token_id}, 长度: {len(audio)}样本)")
                sentences_with_audio.append(sentence)
                
            except Exception as e:
                logger.error(f"句子音频生成失败: {e}")
                # 设置空音频
                sentence.generated_audio = np.zeros(0, dtype=np.float32)
                sentences_with_audio.append(sentence)
        
        logger.info(f"音频生成完成，处理了 {len(sentences_with_audio)} 个句子")
        return sentences_with_audio
        
    except Exception as e:
        logger.error(f"批量音频生成失败: {e}")
        raise
