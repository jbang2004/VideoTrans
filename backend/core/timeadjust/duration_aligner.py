import logging
import ray
import torch

@ray.remote(num_cpus=0.1)
def align_durations(tts_token_ref, simplifier=None, model_in_actor=None, cosyvoice_actor=None, max_speed=1.1):
    """
    对句子进行时长对齐处理（Ray Task版本）
    
    Args:
        tts_token_ref: TTS token生成任务的引用或句子列表
        simplifier: 简化器Actor引用
        model_in_actor: 模型输入Actor引用
        cosyvoice_actor: CosyVoice模型Actor引用
        max_speed: 最大语速阈值
        
    Returns:
        List[Sentence]: 处理后的句子列表
    """
    logger = logging.getLogger("duration_aligner")
    
    # 获取输入句子
    sentences = ray.get(tts_token_ref) if not isinstance(tts_token_ref, list) else tts_token_ref
    
    if not sentences:
        logger.warning("align_durations: 收到空的句子列表")
        return sentences
    
    logger.info(f"开始处理 {len(sentences)} 个句子的时长对齐")
    
    try:
        # 第一次对齐
        sentences = _align_batch(sentences, logger)
        
        # 查找语速过快的句子
        fast_indices = [i for i, sentence in enumerate(sentences) if sentence.speed > max_speed]
        
        # 如果有语速过快的句子，进行精简处理
        if fast_indices and simplifier and model_in_actor and cosyvoice_actor:
            logger.info(f"发现 {len(fast_indices)} 个语速过快的句子，进行精简...")
            
            # 提取需要精简的句子
            fast_sentences = [sentences[idx] for idx in fast_indices]
            
            # 1. 使用TranslatorActor对语速过快的句子进行精简
            simplified_sentences = None
            for batch_ref in ray.get(simplifier.simplify_sentences.remote(
                fast_sentences,
                target_speed=max_speed
            )):
                simplified_sentences = batch_ref
            
            if simplified_sentences:
                # 2. 使用model_in_actor处理简化后的句子
                refined_sentences = []
                
                for batch_ref in ray.get(model_in_actor.modelin_maker.remote(
                    simplified_sentences,
                    reuse_speaker=True,
                    batch_size=3
                )):
                    # 3. 使用generate_tts_tokens生成新的TTS token
                    from core.tts_token_gener import generate_tts_tokens
                    processed_batch = ray.get(generate_tts_tokens.remote(
                        batch_ref,
                        cosyvoice_actor
                    ))
                    refined_sentences.extend(processed_batch)
                
                # 4. 创建新的句子列表，替换精简后的句子
                if refined_sentences:
                    # 创建结果列表的副本
                    result_sentences = sentences.copy()
                    
                    # 用精简后的句子更新结果列表
                    for i, orig_idx in enumerate(fast_indices):
                        if i < len(refined_sentences):
                            result_sentences[orig_idx] = refined_sentences[i]
                            logger.info(f"更新句子[{orig_idx}]: {sentences[orig_idx].trans_text} -> {refined_sentences[i].trans_text}")
                    
                    # 对更新后的句子再次对齐
                    logger.info("精简完成，进行最终对齐...")
                    result_sentences = _align_batch(result_sentences, logger)
                    return result_sentences
            
            logger.warning("精简过程未能生成有效句子，保持原句子")
        
        logger.info(f"时长对齐处理完成，共处理 {len(sentences)} 个句子")
        return sentences
        
    except Exception as e:
        logger.error(f"时长对齐处理失败: {e}")
        raise

def _align_batch(sentences, logger):
    """
    同批次句子进行时长对齐的内部实现
    """
    if not sentences:
        return sentences

    # 创建一个新的句子列表（深拷贝原始句子）
    aligned_sentences = []
    for s in sentences:
        # 在Python中，通常需要实现自定义的深拷贝
        # 这里简化处理，假设句子对象有必要的属性可以直接访问
        aligned_s = s  # 在实际应用中，这里应该是深拷贝
        aligned_s.diff = aligned_s.duration - aligned_s.target_duration
        aligned_sentences.append(aligned_s)

    # 计算需要调整的时间差
    total_diff_to_adjust = sum(s.diff for s in aligned_sentences)
    positive_diff_sum = sum(x.diff for x in aligned_sentences if x.diff > 0)
    negative_diff_sum_abs = sum(abs(x.diff) for x in aligned_sentences if x.diff < 0)
    current_time = aligned_sentences[0].start

    for s in aligned_sentences:
        s.adjusted_start = current_time
        diff = s.diff
        # 确保初始speed不为0
        s.speed = 1.0
        s.silence_duration = 0.0
        s.adjusted_duration = s.duration

        if total_diff_to_adjust != 0:
            if total_diff_to_adjust > 0 and diff > 0:
                if positive_diff_sum > 0:
                    proportion = diff / positive_diff_sum
                    adjustment = total_diff_to_adjust * proportion
                    s.adjusted_duration = s.duration - adjustment
                    s.speed = s.duration / max(s.adjusted_duration, 0.001)
            elif total_diff_to_adjust < 0 and diff < 0:
                if negative_diff_sum_abs > 0:
                    proportion = abs(diff) / negative_diff_sum_abs
                    total_needed = abs(total_diff_to_adjust) * proportion
                    max_slowdown = s.duration * 0.07
                    slowdown = min(total_needed, max_slowdown)
                    s.adjusted_duration = s.duration + slowdown
                    s.speed = s.duration / max(s.adjusted_duration, 0.001)
                    s.silence_duration = total_needed - slowdown
                    if s.silence_duration > 0:
                        s.adjusted_duration += s.silence_duration

        s.diff = s.duration - s.adjusted_duration
        current_time += s.adjusted_duration

        logger.info(
            f"对齐后: {s.trans_text}, duration: {s.duration}, "
            f"target_duration: {s.target_duration}, diff: {s.diff}, "
            f"speed: {s.speed}, silence_duration: {s.silence_duration}"
        )
    
    return aligned_sentences
