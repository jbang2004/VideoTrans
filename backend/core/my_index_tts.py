import os
import re
import time
import sentencepiece as spm
import torch
import torchaudio

from backend.models.IndexTTS.indextts.infer import IndexTTS
from backend.models.IndexTTS.indextts.utils.feature_extractors import MelSpectrogramFeatures
from backend.models.IndexTTS.indextts.utils.common import tokenize_by_CJK_char


# --- Simplified Path Logic ---
# Assume the script is run from the project root or the paths are relative to the project root.
# Alternatively, calculate relative to this file's location if needed, but root-relative is often cleaner.
_MY_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(_MY_DIR) # Assumes core is directly under backend

CHECKPOINTS_DIR = os.path.join(_BACKEND_DIR, 'models', 'IndexTTS', 'checkpoints')
CFG_PATH = os.path.join(CHECKPOINTS_DIR, 'config.yaml')
MODEL_DIR =     CHECKPOINTS_DIR
# --- End Simplified Path Logic ---


class MyIndexTTS(IndexTTS):
    def __init__(self, cfg_path=CFG_PATH, model_dir=MODEL_DIR, is_fp16=True, device=None):
        """
        继承并修改 IndexTTS.

        Args:
            cfg_path (str): 配置文件的路径.
            model_dir (str): 模型目录的路径.
            is_fp16 (bool): 是否使用 fp16.
            device (str): 使用的设备 (e.g., 'cuda', 'cpu'). 如果为 None, 会自动检测 CUDA 或 MPS.
                          注意: 这里我们优先使用 'cuda' 而不是 'cuda:0'.
        """
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda' # 使用 'cuda' 而不是 'cuda:0'
            elif torch.backends.mps.is_available(): # Corrected MPS check
                device = 'mps'
            else:
                device = 'cpu'

        super().__init__(cfg_path=cfg_path, model_dir=model_dir, is_fp16=is_fp16, device=device)

        # === 在初始化时加载 Tokenizer ===
        self.tokenizer = spm.SentencePieceProcessor()

        self.tokenizer.load(self.bpe_path)
        print(f">> Tokenizer loaded from: {self.bpe_path}")
        print(f">> MyIndexTTS initialized on device: {self.device} with FP16: {self.is_fp16}")


    def infer(self, audio_prompt, text, output_path):
        # 覆写 infer 方法，使其结构更接近原始版本，但使用 self.tokenizer
        print(f"Origin text: {text}")
        text = self.preprocess_text(text)
        print(f"Normalized text: {text}")

        audio, sr = torchaudio.load(audio_prompt)
        audio = torch.mean(audio, dim=0, keepdim=True)
        if audio.shape[0] > 1:
            audio = audio[0].unsqueeze(0)

        if sr != 24000:
             audio = torchaudio.transforms.Resample(sr, 24000)(audio)
        cond_mel = MelSpectrogramFeatures()(audio).to(self.device)

        print(f"cond_mel shape: {cond_mel.shape}", "dtype:", cond_mel.dtype)

        auto_conditioning = cond_mel

        punctuation = ["!", "?", ".", ";", "！", "？", "。", "；"]
        pattern = r"(?<=[{0}])\s*".format("".join(re.escape(p) for p in punctuation))
        sentences = [s for s in re.split(pattern, text) if s.strip()]
        if not sentences:
             print("Warning: No sentences found after splitting text.")
             # Consider raising an error or returning an empty result indicator
             return
        print("Sentences:", sentences)

        top_p = .8
        top_k = 30
        temperature = 1.0
        autoregressive_batch_size = 1
        length_penalty = 0.0
        num_beams = 3
        repetition_penalty = 10.0
        max_mel_tokens = 600
        sampling_rate = 24000

        wavs = []
        print(">> Start inference...")
        start_time = time.time()

        for sent in sentences:
            print(f"Processing sentence: {sent}")
            cleand_text = tokenize_by_CJK_char(sent)
            print("Cleaned text:", cleand_text)

            # 使用 self.tokenizer
            text_tokens = torch.IntTensor(self.tokenizer.encode(cleand_text)).unsqueeze(0).to(self.device)

            if text_tokens.numel() == 0:
                print(f"Warning: Empty token sequence for sentence: {sent}")
                continue

            print(f"text_tokens shape: {text_tokens.shape}, type: {text_tokens.dtype}")

            text_len = torch.IntTensor([text_tokens.size(1)]).to(self.device)

            with torch.no_grad():
                # GPT Part 1: Generate codes
                with torch.amp.autocast(self.device, enabled=self.is_fp16, dtype=self.dtype):
                    codes = self.gpt.inference_speech(
                        auto_conditioning,
                        text_tokens,
                        cond_mel_lengths=torch.tensor([auto_conditioning.shape[-1]], device=self.device),
                        do_sample=True,
                        top_p=top_p,
                        top_k=top_k,
                        temperature=temperature,
                        num_return_sequences=autoregressive_batch_size,
                        length_penalty=length_penalty,
                        num_beams=num_beams,
                        repetition_penalty=repetition_penalty,
                        max_generate_length=max_mel_tokens,
                    )
                #codes = codes[:, :-2]
                codes, code_lens = self.remove_long_silence(codes, silent_token=52, max_consecutive=30)
                print(codes, type(codes))
                print(f"fix codes shape: {codes.shape}, codes type: {codes.dtype}")
                print(f"code len: {code_lens}")

                # GPT Part 2: Generate latent
                with torch.amp.autocast(self.device, enabled=self.is_fp16, dtype=self.dtype):
                    latent = self.gpt(
                        auto_conditioning,
                        text_tokens,
                        text_len, # Pass text_len directly (positional 3)
                        codes,    # Pass codes directly (positional 4)
                        code_lens * self.gpt.mel_length_compression, # Pass computed lengths (positional 5)
                        cond_mel_lengths=torch.tensor([auto_conditioning.shape[-1]], device=self.device),
                        return_latent=True,
                        clip_inputs=False
                    )

                if latent is None or latent.numel() == 0:
                     print(f"Warning: Latent generation failed for sentence: {sent}. Skipping.")
                     continue

            # BigVGAN: Generate waveform (outside autocast)
            with torch.no_grad():
                # Pass latent directly, only transpose auto_conditioning
                wav, _ = self.bigvgan(latent, auto_conditioning.transpose(1, 2))
            wav = wav.squeeze(1).cpu()

            # Post-process and append
            wav = 32767.0 * wav
            wav = torch.clamp(wav, -32767.0, 32767.0)
            print(f"Generated wav part shape: {wav.shape}")
            wavs.append(wav)

        end_time = time.time()
        elapsed_time = end_time - start_time
        minutes, seconds = divmod(int(elapsed_time), 60)
        milliseconds = int((elapsed_time - int(elapsed_time)) * 1000)
        print(f">> Inference done. Time: {minutes:02d}:{seconds:02d}.{milliseconds:03d}")

        if not wavs:
             print("Error: No audio generated.")
             # Consider raising an error or returning None
             return

        print(">> Saving wav file...")
        try:
            final_wav = torch.cat(wavs, dim=1)
            torchaudio.save(output_path, final_wav.to(torch.int16), sampling_rate)
            print(f">> Wav file saved to: {output_path}")
        except Exception as e:
            print(f"Error saving wav file: {e}")


# 使用示例 (可以放在另一个文件中)
if __name__ == "__main__":
    prompt_wav="test_data/input.wav"
    prompt_wav="testwav/spk_1744181067_1.wav"
    #text="晕 XUAN4 是 一 种 GAN3 觉"
    #text='大家好，我现在正在bilibili 体验 ai 科技，说实话，来之前我绝对想不到！AI技术已经发展到这样匪夷所思的地步了！'
    text="There is a vehicle arriving in dock number 7?"

    tts = MyIndexTTS(cfg_path=CFG_PATH, model_dir=MODEL_DIR, is_fp16=True)
    tts.infer(audio_prompt=prompt_wav, text=text, output_path="gen.wav")
