import logging
import asyncio
import uuid
import torch
import numpy as np
from typing import List
from .concurrency import run_sync
from shared.sentence_tools import Sentence

logger = logging.getLogger(__name__)

class TTSTokenGenerator:
    """
    根据文本生成 TTS token 并更新句子相关信息，
    参考 backend/core/tts_token_gener.py 的实现。
    """
    def __init__(self, cosyvoice_model, Hz=25):
        self.cosyvoice_model = cosyvoice_model.model
        self.Hz = Hz
        self.logger = logging.getLogger(__name__)

    async def tts_token_maker(self, sentences: List[Sentence], reuse_uuid: bool = False) -> List[Sentence]:
        tasks = []
        for s in sentences:
            current_uuid = s.model_input.get('uuid') if reuse_uuid and s.model_input.get('uuid') else str(uuid.uuid1())
            tasks.append(asyncio.create_task(self._generate_tts_single_async(s, current_uuid)))
        processed = await asyncio.gather(*tasks)
        for sen in processed:
            if not sen.model_input.get('segment_speech_tokens'):
                self.logger.error(f"TTS token generation failed for: {sen.trans_text}")
        return processed

    async def _generate_tts_single_async(self, sentence: Sentence, main_uuid: str):
        return await run_sync(self._generate_tts_single, sentence, main_uuid)

    def _generate_tts_single(self, sentence: Sentence, main_uuid: str):
        model_input = sentence.model_input
        segment_tokens_list = []
        segment_uuids = []
        total_token_count = 0
        try:
            for i, (text, text_len) in enumerate(zip(model_input.get('text', []), model_input.get('text_len', []))):
                seg_uuid = f"{main_uuid}_seg_{i}"
                with self.cosyvoice_model.lock:
                    self.cosyvoice_model.tts_speech_token_dict[seg_uuid] = []
                    self.cosyvoice_model.llm_end_dict[seg_uuid] = False
                    if hasattr(self.cosyvoice_model, 'mel_overlap_dict'):
                        self.cosyvoice_model.mel_overlap_dict[seg_uuid] = None
                    self.cosyvoice_model.hift_cache_dict[seg_uuid] = None
                self.cosyvoice_model.llm_job(
                    text,
                    model_input.get('prompt_text', torch.zeros(1, 0, dtype=torch.int32)),
                    model_input.get('llm_prompt_speech_token', torch.zeros(1, 0, dtype=torch.int32)),
                    model_input.get('llm_embedding', torch.zeros(0, 192)),
                    seg_uuid
                )
                seg_tokens = self.cosyvoice_model.tts_speech_token_dict[seg_uuid]
                segment_tokens_list.append(seg_tokens)
                segment_uuids.append(seg_uuid)
                total_token_count += len(seg_tokens)
            total_duration_s = total_token_count / self.Hz
            sentence.duration = total_duration_s * 1000
            model_input['segment_speech_tokens'] = segment_tokens_list
            model_input['segment_uuids'] = segment_uuids
            model_input['uuid'] = main_uuid
            self.logger.debug(f"TTS tokens generated: UUID={main_uuid}, duration={total_duration_s:.2f}s, segments={len(segment_uuids)}")
            return sentence
        except Exception as e:
            self.logger.error(f"TTS token generation error (UUID={main_uuid}): {e}")
            with self.cosyvoice_model.lock:
                for seg_uuid in segment_uuids:
                    self.cosyvoice_model.tts_speech_token_dict.pop(seg_uuid, None)
                    self.cosyvoice_model.llm_end_dict.pop(seg_uuid, None)
                    self.cosyvoice_model.hift_cache_dict.pop(seg_uuid, None)
                    if hasattr(self.cosyvoice_model, 'mel_overlap_dict'):
                        self.cosyvoice_model.mel_overlap_dict.pop(seg_uuid, None)
            raise

class AudioGenerator:
    """
    根据 TTS tokens 生成语音波形，
    参考 backend/core/audio_gener.py 的实现。
    """
    def __init__(self, cosyvoice_model, sample_rate: int = None):
        self.cosyvoice_model = cosyvoice_model.model
        self.sample_rate = sample_rate or cosyvoice_model.sample_rate
        self.logger = logging.getLogger(__name__)

    async def vocal_audio_maker(self, sentences: List[Sentence]) -> None:
        tasks = [self._generate_single_async(s) for s in sentences]
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            self.logger.error(f"Audio generation failed: {e}")
            raise

    async def _generate_single_async(self, sentence: Sentence):
        try:
            audio_np = await run_sync(self._generate_audio_single, sentence)
            sentence.generated_audio = audio_np
        except Exception as e:
            self.logger.error(f"Audio generation failed for sentence (UUID: {sentence.model_input.get('uuid', 'unknown')}): {e}")
            sentence.generated_audio = None

    def _generate_audio_single(self, sentence: Sentence):
        model_input = sentence.model_input
        self.logger.debug(f"Generating audio for sentence (UUID: {model_input.get('uuid', 'unknown')})")
        try:
            segment_audio_list = []
            tokens_list = model_input.get('segment_speech_tokens', [])
            uuids_list = model_input.get('segment_uuids', [])
            if not tokens_list or not uuids_list:
                self.logger.debug(f"Empty tokens for UUID: {model_input.get('uuid', 'unknown')}, generating empty waveform.")
                segment_audio_list.append(np.zeros(0, dtype=np.float32))
            else:
                for i, (tokens, seg_uuid) in enumerate(zip(tokens_list, uuids_list)):
                    if not tokens:
                        segment_audio_list.append(np.zeros(0, dtype=np.float32))
                        continue
                    token2wav_kwargs = {
                        'token': torch.tensor(tokens).unsqueeze(dim=0),
                        'token_offset': 0,
                        'finalize': True,
                        'prompt_token': model_input.get('flow_prompt_speech_token', torch.zeros(1, 0, dtype=torch.int32)),
                        'prompt_feat': model_input.get('prompt_speech_feat', torch.zeros(1, 0, 80)),
                        'embedding': model_input.get('flow_embedding', torch.zeros(0)),
                        'uuid': seg_uuid,
                        'speed': sentence.speed if sentence.speed else 1.0
                    }
                    segment_output = self.cosyvoice_model.token2wav(**token2wav_kwargs)
                    segment_audio = segment_output.cpu().numpy()
                    if segment_audio.ndim > 1:
                        segment_audio = segment_audio.mean(axis=0)
                    segment_audio_list.append(segment_audio)
                    self.logger.debug(f"Segment {i+1}/{len(uuids_list)} generated: duration {len(segment_audio)/self.sample_rate:.2f}s")
            if segment_audio_list:
                final_audio = np.concatenate(segment_audio_list)
            else:
                final_audio = np.zeros(0, dtype=np.float32)
            if sentence.is_first and sentence.start > 0:
                silence_samples = int(sentence.start * self.sample_rate / 1000)
                final_audio = np.concatenate([np.zeros(silence_samples, dtype=np.float32), final_audio])
            if sentence.silence_duration > 0:
                silence_samples = int(sentence.silence_duration * self.sample_rate / 1000)
                final_audio = np.concatenate([final_audio, np.zeros(silence_samples, dtype=np.float32)])
            self.logger.debug(f"Audio generation complete (UUID: {model_input.get('uuid', 'unknown')}, duration: {len(final_audio)/self.sample_rate:.2f}s)")
            return final_audio
        except Exception as e:
            self.logger.error(f"Audio generation error (UUID: {model_input.get('uuid', 'unknown')}): {e}")
            raise
