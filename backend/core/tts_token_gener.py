import logging
import asyncio
import uuid
import torch
import ray

from utils import concurrency

class TTSTokenGenerator:
    def __init__(self, cosyvoice_model_actor, Hz=25):
        """
        Args:
            cosyvoice_model_actor: CosyVoice模型Actor引用
            Hz: token频率
        """
        self.cosyvoice_actor = cosyvoice_model_actor
        self.Hz = Hz
        self.logger = logging.getLogger(__name__)

    async def tts_token_maker(self, sentences, reuse_uuid=False):
        """生成TTS tokens（调用Actor）"""
        try:
            # 确保sentences不是协程
            if asyncio.iscoroutine(sentences):
                self.logger.warning("收到协程对象而非句子列表，尝试等待...")
                sentences = await sentences
            
            tasks = []
            for s in sentences:
                # 检查单个句子是否为协程
                if asyncio.iscoroutine(s):
                    self.logger.warning(f"句子{id(s)}是协程，等待...")
                    s = await s
                
                current_uuid = (
                    s.model_input.get('uuid') if reuse_uuid and s.model_input.get('uuid')
                    else str(uuid.uuid1())
                )
                # 直接将协程函数添加到任务列表
                tasks.append(self._generate_tts_single_async(s, current_uuid))

            processed = await asyncio.gather(*tasks)

            for sen in processed:
                if not sen.model_input.get('segment_speech_tokens'):
                    self.logger.error(f"TTS token 生成失败: {sen.trans_text}")

            return processed

        except Exception as e:
            self.logger.error(f"TTS token 生成失败: {e}")
            raise

    async def _generate_tts_single_async(self, sentence, main_uuid):
        """异步调用Actor生成TTS tokens"""
        # 检查sentence是否为协程
        if asyncio.iscoroutine(sentence):
            self.logger.warning(f"在_generate_tts_single_async中接收到协程，等待...")
            sentence = await sentence
        
        # 使用concurrency.run_sync而非直接调用
        try:
            # 使用concurrency.run_sync实现异步生成
            return await concurrency.run_sync(
                self._generate_tts_single, sentence, main_uuid
            )
        except Exception as e:
            self.logger.error(f"处理失败 (UUID={main_uuid}): {e}")
            raise

    def _generate_tts_single(self, sentence, main_uuid):
        """同步调用Actor生成TTS tokens"""
        model_input = sentence.model_input
        segment_tokens_list = []
        segment_uuids = []
        total_token_count = 0

        try:
            for i, (text, text_len) in enumerate(zip(model_input['text'], model_input['text_len'])):
                seg_uuid = f"{main_uuid}_seg_{i}"
                
                # 调用Actor生成TTS tokens
                prompt_text = model_input.get('prompt_text', torch.zeros(1, 0, dtype=torch.int32))
                llm_prompt_speech_token = model_input.get('llm_prompt_speech_token', torch.zeros(1, 0, dtype=torch.int32))
                llm_embedding = model_input.get('llm_embedding', torch.zeros(0, 192))
                
                seg_tokens = ray.get(self.cosyvoice_actor.generate_tts_tokens.remote(
                    text, prompt_text, llm_prompt_speech_token, llm_embedding, seg_uuid
                ))

                segment_tokens_list.append(seg_tokens)
                segment_uuids.append(seg_uuid)
                total_token_count += len(seg_tokens)

            total_duration_s = total_token_count / self.Hz
            sentence.duration = total_duration_s * 1000

            model_input['segment_speech_tokens'] = segment_tokens_list
            model_input['segment_uuids'] = segment_uuids
            model_input['uuid'] = main_uuid

            self.logger.debug(
                f"TTS token 生成完成 (UUID={main_uuid}, 时长={total_duration_s:.2f}s, "
                f"段数={len(segment_uuids)})"
            )
            return sentence

        except Exception as e:
            self.logger.error(f"生成失败 (UUID={main_uuid}): {e}")
            # 调用Actor清理
            for seg_uuid in segment_uuids:
                ray.get(self.cosyvoice_actor.cleanup_tts_tokens.remote(seg_uuid))
            raise