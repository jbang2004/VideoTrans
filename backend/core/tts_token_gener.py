import logging
import asyncio
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

    async def tts_token_maker(self, sentences):
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
                
                # 直接将协程函数添加到任务列表
                tasks.append(self._generate_tts_single_async(s))

            processed = await asyncio.gather(*tasks)

            for sen in processed:
                if not sen.model_input.get('tts_token_id'):
                    self.logger.error(f"TTS token 生成失败: {sen.trans_text}")

            return processed

        except Exception as e:
            self.logger.error(f"TTS token 生成失败: {e}")
            raise

    async def _generate_tts_single_async(self, sentence):
        """异步调用Actor生成TTS tokens"""
        # 检查sentence是否为协程
        if asyncio.iscoroutine(sentence):
            self.logger.warning(f"在_generate_tts_single_async中接收到协程，等待...")
            sentence = await sentence
        
        # 使用concurrency.run_sync而非直接调用
        try:
            # 使用concurrency.run_sync实现异步生成
            return await concurrency.run_sync(
                self._generate_tts_single, sentence
            )
        except Exception as e:
            self.logger.error(f"处理失败: {e}")
            raise

    def _generate_tts_single(self, sentence):
        """同步调用Actor生成TTS tokens"""
        model_input = sentence.model_input
        
        try:
            # 获取特征ID
            text_feature_id = model_input.get('text_feature_id')
            speaker_feature_id = model_input.get('speaker_feature_id')
            
            if not text_feature_id or not speaker_feature_id:
                raise ValueError(f"缺少必要的特征ID: text_feature_id={text_feature_id}, speaker_feature_id={speaker_feature_id}")
            
            # 获取可能存在的tts_token_id，用于ID复用
            existing_tts_id = model_input.get('tts_token_id')
            
            # 生成新的TTS token特征，如果有现有的tts_token_id则复用ID
            tts_token_id, duration = ray.get(self.cosyvoice_actor.generate_tts_tokens_and_cache.remote(
                text_feature_id, speaker_feature_id, existing_tts_id
            ))
            
            # 更新model_input
            model_input['tts_token_id'] = tts_token_id
            sentence.duration = duration

            self.logger.debug(
                f"TTS token 生成完成 (ID={tts_token_id}, 时长={duration:.2f}ms)"
            )
            return sentence

        except Exception as e:
            self.logger.error(f"生成失败: {e}")
            # 清理可能已创建的TTS token
            old_tts_token_id = model_input.get('tts_token_id')
            if old_tts_token_id:
                try:
                    ray.get(self.cosyvoice_actor.cleanup_tts_tokens.remote(old_tts_token_id))
                except Exception as cleanup_error:
                    self.logger.warning(f"清理TTS token失败: {cleanup_error}")
            raise