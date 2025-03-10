import logging
import ray
from typing import List

@ray.remote(num_cpus=0.1)
def adjust_timestamps(sentences, sample_rate, start_time=None):
    """
    更新和验证句子的时间戳信息（Ray Task版本）
    
    Args:
        sentences: 要处理的句子列表
        sample_rate: 采样率
        start_time: 起始时间（毫秒），如果为None则使用第一个句子的开始时间
        
    Returns:
        List: 处理后的句子列表（已更新时间戳）
    """
    logger = logging.getLogger("timestamp_adjuster")
    
    if not sentences:
        logger.warning("adjust_timestamps: 收到空的句子列表")
        return sentences
    
    logger.info(f"开始处理 {len(sentences)} 个句子的时间戳调整")
    
    # 更新时间戳
    current_time = start_time if start_time is not None else sentences[0].start
    
    for sentence in sentences:
        # 计算实际音频长度（毫秒）
        if sentence.generated_audio is not None:
            actual_duration = (len(sentence.generated_audio) / sample_rate) * 1000
        else:
            actual_duration = 0
            logger.warning(f"句子 {sentence.sentence_id} 没有生成音频")
        
        # 更新时间戳
        sentence.adjusted_start = current_time
        sentence.adjusted_duration = actual_duration
        
        # 更新差异值
        sentence.diff = sentence.duration - actual_duration
        
        # 更新下一个句子的开始时间
        current_time += actual_duration
    
    # 验证时间戳（保留验证逻辑，但不使用is_valid变量）
    validation_issues = 0
    for i in range(len(sentences) - 1):
        current = sentences[i]
        next_sentence = sentences[i + 1]
        
        # 验证时间连续性
        expected_next_start = current.adjusted_start + current.adjusted_duration
        if abs(next_sentence.adjusted_start - expected_next_start) > 1:  # 允许1毫秒的误差
            logger.error(
                f"时间戳不连续 - 句子 {current.sentence_id} 结束时间: {expected_next_start:.2f}ms, "
                f"句子 {next_sentence.sentence_id} 开始时间: {next_sentence.adjusted_start:.2f}ms"
            )
            validation_issues += 1
            
        # 验证时长有效性
        if current.adjusted_duration <= 0:
            logger.error(f"句子 {current.sentence_id} 的时长无效: {current.adjusted_duration:.2f}ms")
            validation_issues += 1
    
    # 记录最终状态
    if validation_issues > 0:
        logger.warning(f"时间戳调整完成，处理了 {len(sentences)} 个句子，结束时间: {current_time:.2f}ms，发现 {validation_issues} 个问题")
    else:
        logger.info(f"时间戳调整完成，处理了 {len(sentences)} 个句子，结束时间: {current_time:.2f}ms，验证通过")
    
    # 只返回处理后的句子列表
    return sentences