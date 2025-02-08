import logging
import os
import torch
import numpy as np
import asyncio
from functools import partial
from typing import List

# 导入 funasr 相关模块，请确保已安装 funasr
from funasr.auto.auto_model import AutoModel as BaseAutoModel, prepare_data_iterator
from funasr.utils.load_utils import load_audio_text_image_video
from funasr.utils.vad_utils import slice_padding_audio_samples
from funasr.models.campplus.utils import sv_chunk, postprocess
from funasr.models.campplus.cluster_backend import ClusterBackend

from shared.sentence_tools import Sentence, get_sentences
from .concurrency import run_sync

logger = logging.getLogger(__name__)

class SenseAutoModel(BaseAutoModel):
    """
    封装 ASR/VAD/SD（speaker diarization）功能。
    参考 backend/core/auto_sense.py 的实现，对外提供 generate_async 接口。
    
    注意：实际使用时请根据具体的 funasr 模型接口进行调整。
    """
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.logger = logging.getLogger(__name__)
        self.config = config
        if self.spk_model is not None:
            self.cb_model = ClusterBackend().to(kwargs.get("device", "cpu"))
            spk_mode = kwargs.get("spk_mode", "punc_segment")
            if spk_mode not in ["default", "vad_segment", "punc_segment"]:
                self.logger.error("spk_mode 应该是 'default', 'vad_segment' 或 'punc_segment'之一。")
            self.spk_mode = spk_mode
        self.kwargs = kwargs
        self.vad_kwargs = kwargs.get("vad_kwargs", {})

    def combine_results(self, restored_data, vadsegments):
        result = {}
        for j, data in enumerate(restored_data):
            for k, v in data.items():
                if k.startswith("timestamp"):
                    result.setdefault(k, [])
                    for t in v:
                        t[0] += vadsegments[j][0]
                        t[1] += vadsegments[j][0]
                    result[k].extend(v)
                elif k == "spk_embedding":
                    if k not in result:
                        result[k] = v
                    else:
                        result[k] = torch.cat([result[k], v], dim=0)
                elif "token" in k:
                    result.setdefault(k, [])
                    result[k].extend(v)
                else:
                    result[k] = result.get(k, 0) + v
        return result

    def inference_with_vad(self, input_audio, input_len=None, **cfg):
        """
        根据 VAD 对输入进行分段并进行 ASR 推理，参考 backend 实现。
        """
        kwargs = self.kwargs
        self.tokenizer = kwargs.get("tokenizer")
        from funasr.utils.misc import deep_update
        deep_update(self.vad_kwargs, cfg)
        res = self.inference(input_audio, input_len=input_len, model=self.vad_model, kwargs=self.vad_kwargs, **cfg)
        model = self.model
        deep_update(kwargs, cfg)
        kwargs["batch_size"] = max(int(kwargs.get("batch_size_s", 300)) * 1000, 1)
        batch_size_threshold_ms = int(kwargs.get("batch_size_threshold_s", 60)) * 1000

        key_list, data_list = prepare_data_iterator(input_audio, input_len=input_len, data_type=kwargs.get("data_type", None))
        results_ret_list = []
        for i, item in enumerate(res):
            key, vadsegments = item["key"], item["value"]
            input_i = data_list[i]
            fs = kwargs["frontend"].fs if hasattr(kwargs["frontend"], "fs") else 16000
            speech = load_audio_text_image_video(input_i, fs=fs, audio_fs=kwargs.get("fs", 16000))
            self.logger.debug(f"Audio length: {len(speech)} samples")
            if len(speech) < 400:
                self.logger.warning(f"Audio too short ({len(speech)} samples) for utt: {key}")
            sorted_data = sorted([(seg, idx) for idx, seg in enumerate(vadsegments)], key=lambda x: x[0][1] - x[0][0])
            if not sorted_data:
                self.logger.info(f"No VAD segments for utt: {key}")
                continue
            results_sorted = []
            all_segments = []
            beg_idx, end_idx = 0, 1
            max_len_in_batch = 0
            for j in range(len(sorted_data)):
                sample_length = sorted_data[j][0][1] - sorted_data[j][0][0]
                potential_batch_length = max(max_len_in_batch, sample_length) * (j + 1 - beg_idx)
                if (j < len(sorted_data) - 1 and 
                    sample_length < batch_size_threshold_ms and 
                    potential_batch_length < kwargs["batch_size"]):
                    max_len_in_batch = max(max_len_in_batch, sample_length)
                    end_idx += 1
                    continue
                speech_j, _ = slice_padding_audio_samples(speech, len(speech), sorted_data[beg_idx:end_idx])
                results = self.inference(speech_j, input_len=None, model=model, kwargs=kwargs, **cfg)
                if self.spk_model is not None:
                    for _b, speech_segment in enumerate(speech_j):
                        vad_segment = sorted_data[beg_idx:end_idx][_b][0]
                        segments = sv_chunk([[vad_segment[0] / 1000.0, vad_segment[1] / 1000.0, np.array(speech_segment)]])
                        all_segments.extend(segments)
                        spk_res = self.inference([seg[2] for seg in segments], input_len=None, model=self.spk_model, kwargs=kwargs, **cfg)
                        results[_b]["spk_embedding"] = spk_res[0]["spk_embedding"]
                beg_idx, end_idx = end_idx, end_idx + 1
                max_len_in_batch = sample_length
                results_sorted.extend(results)
            if len(results_sorted) != len(sorted_data):
                self.logger.info(f"Incomplete ASR results for utt: {key}")
                continue
            restored_data = [0] * len(sorted_data)
            for j, (_, idx) in enumerate(sorted_data):
                restored_data[idx] = results_sorted[j]
            combined = self.combine_results(restored_data, vadsegments)
            
            # 处理说话人分离结果
            if self.spk_model is not None and kwargs.get("return_spk_res", True):
                all_segments.sort(key=lambda x: x[0])
                spk_embedding = combined["spk_embedding"]
                labels = self.cb_model(spk_embedding.cpu(), oracle_num=kwargs.get("preset_spk_num", None))
                sv_output = postprocess(all_segments, None, labels, spk_embedding.cpu())
                
                if "timestamp" not in combined:
                    self.logger.error(f"speaker diarization 依赖于时间戳对于 utt: {key}")
                    sentence_list = []
                else:
                    # 使用 get_sentences 函数将 ASR 结果转换为 Sentence 对象列表
                    sentence_list = get_sentences(
                        tokens=combined["token"],
                        timestamps=combined["timestamp"],
                        speech=speech,
                        tokenizer=self.tokenizer,
                        sd_time_list=sv_output,
                        sample_rate=fs,
                        config=self.config
                    )
                    results_ret_list = sentence_list
            else:
                sentence_list = []

        return results_ret_list

    def generate(self, input_audio, input_len=None, **cfg):
        """执行 ASR 推理并返回 Sentence 对象列表"""
        results = self.inference_with_vad(input_audio, input_len, **cfg)
        return results

    async def generate_async(self, input_audio, input_len=None, **cfg):
        """异步执行 ASR 推理并返回 Sentence 对象列表"""
        func = partial(self.generate, input_audio, input_len, **cfg)
        return await run_sync(func)
