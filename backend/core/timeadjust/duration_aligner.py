import logging

class DurationAligner:
    def __init__(self, model_in=None, simplifier=None, tts_token_gener=None, max_speed=1.1):
        """
        :param model_in: 用于更新文本特征的对象(例如 ModelIn)
        :param simplifier: 用于做文本精简的翻译器/简化器(例如 Translator)
        :param tts_token_gener: 用于重新生成TTS token的对象(例如 TTSTokenGenerator)
        :param max_speed: 允许的最大语速阈值, 超过则需要做文本精简
        """
        self.model_in = model_in
        self.simplifier = simplifier
        self.tts_token_gener = tts_token_gener
        self.max_speed = max_speed
        self.logger = logging.getLogger(__name__)

    async def align_durations(self, sentences):
        """
        核心入口：对一批句子进行时长对齐。
        1) 首先做一次批量对齐(_align_batch)。
        2) 若发现有句子速度超出max_speed，则进行文本精简(_retry_sentences_batch)，然后再做一次批量对齐。
        """
        if not sentences:
            return

        # 第一次对齐
        self._align_batch(sentences)

        # 检查语速, 找出需要精简的句子
        retry_sentences = []
        for sentence in sentences:
            if sentence.speed > self.max_speed:
                retry_sentences.append(sentence)

        # 如果确实存在“语速过快”的句子，执行文本精简+重试
        if retry_sentences:
            self.logger.info(f"发现 {len(retry_sentences)} 个句子语速过快，尝试批量精简并重试...")
            success = await self._retry_sentences_batch(retry_sentences)
            if success:
                # 精简后，重新进行一次整批对齐
                self._align_batch(sentences)
            else:
                self.logger.warning("文本精简过程失败，保持第一次对齐结果不变")

    def _align_batch(self, sentences):
        """
        对整批句子做一次分摊式的时长对齐:
        1) 先对每个句子: diff = (s.duration - s.target_duration), 以保证对齐基于原目标时长
        2) 计算整批 total_diff_to_adjust
        3) 分别对正diff/负diff做比例调配(压缩或扩展)
        4) 更新 adjusted_duration / speed / silence_duration 等
        """
        if not sentences:
            return

        # --- 关键改动：先为每个句子重置 diff 为 (当前TTS时长 - 目标时长) ---
        for s in sentences:
            s.diff = s.duration - s.target_duration

        # 计算整批句子的总差值
        total_diff_to_adjust = sum(s.diff for s in sentences)
        current_time = sentences[0].start

        for s in sentences:
            # 将当前句子的起始时间先记录为 current_time
            s.adjusted_start = current_time

            diff = s.diff
            if total_diff_to_adjust == 0:
                # 如果整批差值为 0, 无需再动
                s.adjusted_duration = s.duration
                s.diff = 0

            elif total_diff_to_adjust > 0:
                # 整体时长过长，需要压缩
                positive_diff_sum = sum(x.diff for x in sentences if x.diff > 0)
                if positive_diff_sum > 0 and diff > 0:
                    proportion = diff / positive_diff_sum
                    adjustment = total_diff_to_adjust * proportion
                    s.adjusted_duration = s.duration - adjustment
                    s.diff = s.duration - s.adjusted_duration
                    if s.adjusted_duration != 0:
                        s.speed = s.duration / s.adjusted_duration
                    else:
                        s.speed = 1.0
                else:
                    # 该句并不需要压缩
                    s.adjusted_duration = s.duration
                    s.diff = 0

            else:
                # total_diff_to_adjust < 0 => 整体时长不足，需要扩展
                negative_diff_sum_abs = sum(abs(x.diff) for x in sentences if x.diff < 0)
                if negative_diff_sum_abs > 0 and diff < 0:
                    proportion = abs(diff) / negative_diff_sum_abs
                    total_needed_duration = abs(total_diff_to_adjust) * proportion

                    # 最多只能加10%时长用于减速
                    max_slowdown = s.duration * 0.1
                    slowdown_duration = min(total_needed_duration, max_slowdown)

                    # 减速部分
                    s.adjusted_duration = s.duration + slowdown_duration
                    if s.adjusted_duration != 0:
                        s.speed = s.duration / s.adjusted_duration
                    else:
                        s.speed = 1.0

                    # 剩下的部分当静音插入
                    s.silence_duration = total_needed_duration - slowdown_duration
                    if s.silence_duration > 0:
                        s.adjusted_duration += s.silence_duration
                else:
                    # 不需要扩展
                    s.adjusted_duration = s.duration
                    s.speed = 1.0
                    s.silence_duration = 0

                s.diff = s.duration - s.adjusted_duration

            # 累加当前句子的时长, 用于为下一句设定 adjusted_start
            current_time += s.adjusted_duration

    async def _retry_sentences_batch(self, sentences):
        """
        对速度过快的句子做文本精简 -> 重新生成TTS token & duration
        成功返回 True, 失败返回 False
        """
        try:
            # 1. 收集要精简的句子文本
            texts_to_simplify = {str(i): s.trans_text for i, s in enumerate(sentences)}

            # 2. 调用简化器(翻译器)的 simplify 功能
            simplified_texts = await self.simplifier.simplify(texts_to_simplify)

            # 3. 更新每个句子的文本, 并且重新生成 TTS(从而更新 s.duration 等)
            for i, sentence in enumerate(sentences):
                simplified_text = simplified_texts.get(str(i))
                if simplified_text:
                    self.logger.info(f"句子精简结果:\n原文: {sentence.trans_text}\n精简: {simplified_text}")
                    sentence.trans_text = simplified_text
                    # 重新更新文本特征
                    self.model_in.update_text_features(sentence)
                    # 重新生成TTS token/音频(新的 sentence.duration、diff 等会被更新)
                    self.tts_token_gener._generate_tts_single(sentence, sentence.model_input.get('uuid'))
                    self.logger.info(f"句子重试成功，字数变化: {len(sentence.trans_text)}/{len(sentence.raw_text)}")
                else:
                    self.logger.warning(f"句子精简失败，保持原文: {sentence.trans_text}")

            return True

        except Exception as e:
            self.logger.error(f"批量句子重试失败: {str(e)}")
            return False