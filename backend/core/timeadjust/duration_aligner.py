import logging

class DurationAligner:
    def __init__(self, model_in=None, simplifier=None, tts_token_gener=None, max_speed=1.1):
        self.model_in = model_in
        self.simplifier = simplifier
        self.tts_token_gener = tts_token_gener
        self.max_speed = max_speed
        self.logger = logging.getLogger(__name__)

    async def align_durations(self, sentences):
        """时长对齐处理"""
        if not sentences:
            self.logger.warning("待处理的句子列表为空")
            return
        
        self._align_batch(sentences)
        self.logger.info("第一轮对齐完成")
        
        # 检查需要重新处理的句子
        sentences_to_retry = [s for s in sentences if getattr(s, 'speed', 1.0) > self.max_speed]
        
        if sentences_to_retry:
            self.logger.info(f"发现{len(sentences_to_retry)}个句子速度过快(>{self.max_speed}):")
            for s in sentences_to_retry:
                self.logger.info(f"句子速度: {s.speed:.2f}, 文本: {s.trans_text}")
                
            self.logger.info("开始重试处理速度过快的句子...")
            for sentence in sentences_to_retry:
                await self._retry_sentence(sentence)
                
            # 重新对齐整个批次
            self._align_batch(sentences)

    def _align_batch(self, sentences):
        """原有的对齐逻辑"""
        total_diff_to_adjust = sum(s.diff for s in sentences)
        current_time = sentences[0].start
        
        for s in sentences:
            s.adjusted_start = current_time  # 先设置当前句子的开始时间
            diff = s.diff  # 提前计算 diff

            if total_diff_to_adjust == 0:
                # 整组句子的实际时长和目标时长相等，无需调整
                s.adjusted_duration = s.duration
                s.diff = 0
            elif total_diff_to_adjust > 0:
                # 整体偏长，按 diff 比例缩短每个句子的时长
                positive_diff_sum = sum(s.diff for s in sentences if s.diff > 0)
                if positive_diff_sum > 0 and diff > 0:
                    proportion = diff / positive_diff_sum
                    adjustment = total_diff_to_adjust * proportion  # 调整量基于整体偏差
                    s.adjusted_duration = s.duration - adjustment
                    s.diff = s.duration - s.adjusted_duration
                    s.speed = s.duration / s.adjusted_duration if s.adjusted_duration != 0 else 1
                else:
                    s.adjusted_duration = s.duration
                    s.diff = 0  # 明确指出，未调整的句子 diff 为 0
            else:  # total_diff_to_adjust < 0
                # 整体偏短，先尝试通过降低速度来补偿，必要时再添加静音
                negative_diff_sum_abs = sum(abs(s.diff) for s in sentences if s.diff < 0)
                if negative_diff_sum_abs > 0 and diff < 0:
                    # 计算当前句子需要补偿的时长
                    proportion = abs(diff) / negative_diff_sum_abs
                    total_needed_duration = abs(total_diff_to_adjust) * proportion

                    # 通过降低速度来补偿（最多降低10%）
                    max_slowdown = s.duration * 0.1
                    slowdown_duration = min(total_needed_duration, max_slowdown)
                    
                    # 设置新的时长和速度
                    s.adjusted_duration = s.duration + slowdown_duration
                    s.speed = s.duration / s.adjusted_duration
                    
                    # 如果降速后仍需要补偿，添加静音
                    s.silence_duration = total_needed_duration - slowdown_duration
                    if s.silence_duration > 0:
                        s.adjusted_duration += s.silence_duration
                else:
                    # 不需要调整的句子
                    s.adjusted_duration = s.duration
                    s.speed = 1.0
                    s.silence_duration = 0
                
                s.diff = s.duration - s.adjusted_duration

            current_time += s.adjusted_duration  # 更新 current_time

    async def _retry_sentence(self, sentence):
        """重试处理单个句子"""
        try:
            self.logger.info(f"开始精简句子: {sentence.trans_text}")
            
            # 获取简化的翻译
            simplified_text = await self.simplifier.simplify_text(sentence.trans_text)
            self.logger.info(f"句子精简结果: \n原文: {sentence.trans_text}\n精简: {simplified_text}")
            
            sentence.trans_text = simplified_text
            
            # 直接使用 model_in 更新文本特征
            self.model_in.update_text_features(sentence)
            
            # 重新生成语音 token，复用原有的 UUID
            self.tts_token_gener._generate_tts_single(sentence, sentence.model_input.get('uuid'))
            
            self.logger.info(f"句子重试成功，字数变化: {len(sentence.trans_text)}/{len(sentence.raw_text)}")
            return True
            
        except Exception as e:
            self.logger.error(f"句子重试失败: {str(e)}")
            return False