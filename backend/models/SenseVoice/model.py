import time
import torch
from torch import nn
import torchaudio
from funasr.utils.load_utils import load_audio_text_image_video, extract_fbank
from funasr.models.sense_voice.model import SenseVoiceSmall as BaseSenseVoiceSmall
from funasr.register import tables

@tables.register("model_classes", "SenseVoiceSmall")
class Model(BaseSenseVoiceSmall):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 这里可以添加或修改需要的属性
        
    def inference(
        self,
        data_in,
        data_lengths=None,
        key: list = ["wav_file_tmp_name"],
        tokenizer=None,
        frontend=None,
        **kwargs,
    ):

        meta_data = {}
        if (
            isinstance(data_in, torch.Tensor) and kwargs.get("data_type", "sound") == "fbank"
        ):  # fbank
            speech, speech_lengths = data_in, data_lengths
            if len(speech.shape) < 3:
                speech = speech[None, :, :]
            if speech_lengths is None:
                speech_lengths = speech.shape[1]
        else:
            # extract fbank feats
            time1 = time.perf_counter()
            audio_sample_list = load_audio_text_image_video(
                data_in,
                fs=frontend.fs,
                audio_fs=kwargs.get("fs", 16000),
                data_type=kwargs.get("data_type", "sound"),
                tokenizer=tokenizer,
            )
            time2 = time.perf_counter()
            meta_data["load_data"] = f"{time2 - time1:0.3f}"
            speech, speech_lengths = extract_fbank(
                audio_sample_list, data_type=kwargs.get("data_type", "sound"), frontend=frontend
            )
            time3 = time.perf_counter()
            meta_data["extract_feat"] = f"{time3 - time2:0.3f}"
            meta_data["batch_data_time"] = (
                speech_lengths.sum().item() * frontend.frame_shift * frontend.lfr_n / 1000
            )

        speech = speech.to(device=kwargs["device"])
        speech_lengths = speech_lengths.to(device=kwargs["device"])

        language = kwargs.get("language", "auto")
        language_query = self.embed(
            torch.LongTensor([[self.lid_dict[language] if language in self.lid_dict else 0]]).to(
                speech.device
            )
        ).repeat(speech.size(0), 1, 1)

        use_itn = kwargs.get("use_itn", False)
        textnorm = kwargs.get("text_norm", None)
        if textnorm is None:
            textnorm = "withitn" if use_itn else "woitn"
        textnorm_query = self.embed(
            torch.LongTensor([[self.textnorm_dict[textnorm]]]).to(speech.device)
        ).repeat(speech.size(0), 1, 1)
        speech = torch.cat((textnorm_query, speech), dim=1)
        speech_lengths += 1

        event_emo_query = self.embed(torch.LongTensor([[1, 2]]).to(speech.device)).repeat(
            speech.size(0), 1, 1
        )
        input_query = torch.cat((language_query, event_emo_query), dim=1)
        speech = torch.cat((input_query, speech), dim=1)
        speech_lengths += 3

        # Encoder
        encoder_out, encoder_out_lens = self.encoder(speech, speech_lengths)
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]

        # c. Passed the encoder result and the beam search
        ctc_logits = self.ctc.log_softmax(encoder_out)
        if kwargs.get("ban_emo_unk", False):
            ctc_logits[:, :, self.emo_dict["unk"]] = -float("inf")
        
        # 在这里添加自定义的处理逻辑
        b, n, d = encoder_out.size()  # 假设我们需要访问编码器输出
        
        results = []
        for i in range(b):
            # 修改或添加结果
            x = ctc_logits[i, 4 : encoder_out_lens[i].item(), :]
            yseq = x.argmax(dim=-1)
            yseq = torch.unique_consecutive(yseq, dim=-1)
            mask = yseq != self.blank_id
            
            # 计算时间戳
            original_speech_length = speech_lengths[i].item() - 4
            sample_duration = original_speech_length * frontend.frame_shift * frontend.lfr_n
            sample_frames = encoder_out_lens[i].item() - 4
            
            alignment, scores = torchaudio.functional.forced_align(
                x.unsqueeze(0), 
                yseq[mask].unsqueeze(0), 
                None, 
                None, 
                blank=self.blank_id
            )
            token_spans = torchaudio.functional.merge_tokens(alignment[0], scores[0])
            
            tokens, timestamps = self.get_token_timestamps(
                token_spans, 
                sample_duration, 
                sample_frames, 
                tokenizer
            )
            
            # 更新结果
            result_dict = {
                "token": tokens,
                "timestamp": timestamps,
            }
            results.append(result_dict)

        return results, meta_data

    def get_token_timestamps(self, token_spans, sample_duration, sample_frames, tokenizer):
        """
        计算每个 token 的时间戳并解码。
        
        参数:
        token_spans: forced_align 输出的 TokenSpan 列表
        sample_duration: 音频样本的总时长（毫秒）
        sample_frames: 音频样本的总帧数
        tokenizer: 用于解码的分词器
        
        返回:
        元组 (tokens, timestamps)
        tokens: token列表
        timestamps: 时间戳列表，格式为 [[start1, end1], [start2, end2], ...]，单位为毫秒
        """
        ms_per_frame = sample_duration / sample_frames
        tokens = []
        timestamps = []
        for span in token_spans:
            start_ms = int(span.start * ms_per_frame)
            end_ms = int(span.end * ms_per_frame)
            tokens.append(span.token)
            timestamps.append([start_ms, end_ms])
        
        return tokens, timestamps