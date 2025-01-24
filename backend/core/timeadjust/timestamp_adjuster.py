import logging
from typing import List, Optional
from dataclasses import dataclass

class TimestampAdjuster:
    """句子时间戳调整器"""
    
    def __init__(self, sample_rate: int):
        self.logger = logging.getLogger(__name__)
        self.sample_rate = sample_rate
        
    def update_timestamps(self, sentences: List, start_time: float = None) -> float:
        """更新句子的时间戳信息
        
        Args:
            sentences: 要更新的句子列表
            start_time: 起始时间（毫秒），如果为 None 则使用第一个句子的开始时间
            
        Returns:
            float: 最后一个句子结束的时间点（毫秒）
        """
        if not sentences:
            return start_time if start_time is not None else 0
            
        # 使用传入的起始时间或第一个句子的开始时间
        current_time = start_time if start_time is not None else sentences[0].start
        
        for i, sentence in enumerate(sentences):
            # 计算实际音频长度（毫秒）
            if sentence.generated_audio is not None:
                actual_duration = (len(sentence.generated_audio) / self.sample_rate) * 1000
            else:
                actual_duration = 0
                self.logger.warning(f"句子 {sentence.sentence_id} 没有生成音频")
            
            # 更新时间戳
            sentence.adjusted_start = current_time
            sentence.adjusted_duration = actual_duration
            
            # 更新差异值
            sentence.diff = sentence.duration - actual_duration
            
            # 更新下一个句子的开始时间
            current_time += actual_duration
            
        return current_time
        
    def validate_timestamps(self, sentences: List) -> bool:
        """验证时间戳的连续性和有效性
        
        Args:
            sentences: 要验证的句子列表
            
        Returns:
            bool: 时间戳是否有效
        """
        if not sentences:
            return True
            
        for i in range(len(sentences) - 1):
            current = sentences[i]
            next_sentence = sentences[i + 1]
            
            # 验证时间连续性
            expected_next_start = current.adjusted_start + current.adjusted_duration
            if abs(next_sentence.adjusted_start - expected_next_start) > 1:  # 允许1毫秒的误差
                self.logger.error(
                    f"时间戳不连续 - 句子 {current.sentence_id} 结束时间: {expected_next_start:.2f}ms, "
                    f"句子 {next_sentence.sentence_id} 开始时间: {next_sentence.adjusted_start:.2f}ms"
                )
                return False
                
            # 验证时长有效性
            if current.adjusted_duration <= 0:
                self.logger.error(f"句子 {current.sentence_id} 的时长无效: {current.adjusted_duration:.2f}ms")
                return False
                
        return True