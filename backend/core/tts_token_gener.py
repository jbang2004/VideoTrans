import logging
import ray

@ray.remote(num_cpus=0.1)
def generate_tts_tokens(sentences, cosyvoice_actor):
    """
    生成TTS tokens（Ray Task版本）
    
    Args:
        sentences: 要处理的句子列表
        cosyvoice_actor: CosyVoice模型Actor引用
        
    Returns:
        List[Sentence]: 处理后的句子列表
    """
    logger = logging.getLogger("tts_token_generator")
    
    if not sentences:
        logger.warning("generate_tts_tokens: 收到空的句子列表")
        return sentences
        
    logger.info(f"处理 {len(sentences)} 个句子生成TTS tokens")
    
    try:
        # 处理句子列表
        processed_sentences = []
        for sentence in sentences:
            try:
                # 获取所需特征ID
                model_input = sentence.model_input
                text_feature_id = model_input.get('text_feature_id')
                speaker_feature_id = model_input.get('speaker_feature_id')
                
                if not text_feature_id or not speaker_feature_id:
                    raise ValueError(f"缺少必要的特征ID: text_feature_id={text_feature_id}, speaker_feature_id={speaker_feature_id}")
                
                # 获取可能存在的tts_token_id，用于ID复用
                existing_tts_id = model_input.get('tts_token_id')
                
                # 生成新的TTS token特征
                tts_token_id, duration = ray.get(cosyvoice_actor.generate_tts_tokens_and_cache.remote(
                    text_feature_id, speaker_feature_id, existing_tts_id
                ))
                
                # 更新句子
                model_input['tts_token_id'] = tts_token_id
                sentence.duration = duration
                
                logger.info(f"TTS token 生成完成 (ID={tts_token_id}, 时长={duration:.2f}ms)")
                processed_sentences.append(sentence)
                
            except Exception as e:
                logger.error(f"句子处理失败: {e}")
                # 清理可能已创建的TTS token
                old_tts_token_id = model_input.get('tts_token_id')
                if old_tts_token_id:
                    try:
                        ray.get(cosyvoice_actor.cleanup_tts_tokens.remote(old_tts_token_id))
                    except Exception as cleanup_error:
                        logger.warning(f"清理TTS token失败: {cleanup_error}")
                
                # 仍然添加到结果列表，保持原始顺序
                processed_sentences.append(sentence)
        
        logger.info(f"TTS tokens 生成完成，处理了 {len(processed_sentences)} 个句子")
        return processed_sentences
        
    except Exception as e:
        logger.error(f"批量生成TTS token失败: {e}")
        raise