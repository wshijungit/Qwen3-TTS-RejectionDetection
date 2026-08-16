# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""把 Base 权重里的 speaker_encoder 单独拆出来当特征提取器。

**为什么要拆**：`speaker_encoder` 只在 `tts_model_type == "base"` 时才实例化
（modeling_qwen3_tts.py:1845）。VoiceDesign 和 CustomVoice 的权重里
speaker_encoder 张量数都是 **0** —— 想在 VoiceDesign 上用 speaker 槽位，
向量只能从 Base 那份权重里来。

**为什么能离线算**：官方 sft_12hz.py 那行是
`model.speaker_encoder(ref_mels).detach()` —— encoder 全程冻结、不接收梯度。
既然冻结，离线算完存盘与在线现算**数学上完全等价**，而离线的好处是：

  - 训练时模型里根本不含 speaker_encoder（显存/计算/checkpoint 都不变）
  - 产出的 ckpt 仍是干净的 VoiceDesign，推理侧不用改结构

**维度对得上**（已核）：Base 的 `enc_dim = 2048` == VoiceDesign talker
`hidden_size = 2048`，两边 codec 特殊 id（think/bos/eos/pad）也逐个相同。

**注意**：Base 的 embedding 空间与 VoiceDesign 的 talker 是否还兼容，
**没有被验证过** —— VoiceDesign 是从 Base 继续训的，可能已经漂了。
这一格是靠微调重新学出来的，不是拿来即用。
"""
import glob
import os

import numpy as np
import torch

SPK_DIM = 2048          # Base 的 enc_dim；与 VoiceDesign talker hidden_size 相同
SPK_SR = 24000          # extract_speaker_embedding 里硬性 assert 的采样率
SPK_ITEMSIZE = SPK_DIM * 2      # 一行 fp16 的字节数，续跑对齐要用


def load_speaker_encoder(base_model_path, device="cpu", dtype=torch.float32):
    """只加载 speaker_encoder 那 76 个张量，不碰 1.7B 的 talker。"""
    from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
    from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSSpeakerEncoder
    from safetensors import safe_open

    cfg = Qwen3TTSConfig.from_pretrained(base_model_path)
    if getattr(cfg, "speaker_encoder_config", None) is None:
        raise SystemExit(
            f"{base_model_path} 没有 speaker_encoder_config —— 这不是 Base 权重。\n"
            f"speaker_encoder 只存在于 tts_model_type == 'base' 的那份里，"
            f"VoiceDesign / CustomVoice 都是 0 张量。")

    enc = Qwen3TTSSpeakerEncoder(cfg.speaker_encoder_config)
    prefix = "speaker_encoder."
    sd, found = {}, 0
    for shard in sorted(glob.glob(os.path.join(base_model_path, "*.safetensors"))):
        with safe_open(shard, framework="pt") as f:
            for k in f.keys():
                if k.startswith(prefix):
                    sd[k[len(prefix):]] = f.get_tensor(k)
                    found += 1
    if not found:
        raise SystemExit(f"{base_model_path} 里没有 speaker_encoder.* 张量")
    missing, unexpected = enc.load_state_dict(sd, strict=False)
    # 宁可当场崩：少加载一层权重，产出的向量看起来正常但毫无意义，
    # 而这批向量要喂进 43.5w 条训练数据，事后无法从 loss 上看出来。
    if missing or unexpected:
        raise SystemExit(f"speaker_encoder 权重对不上：missing={missing} unexpected={unexpected}")
    enc.eval().to(device=device, dtype=dtype)
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc


@torch.inference_mode()
def extract_spk(enc, audio, sr, device="cpu", dtype=torch.float32):
    """与 modeling 的 extract_speaker_embedding 逐参数一致（:1988-2001）。"""
    from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram

    if sr != SPK_SR:
        raise ValueError(f"speaker_encoder 只支持 {SPK_SR}Hz，收到 {sr}")
    if isinstance(audio, np.ndarray):
        audio = torch.from_numpy(audio)
    mels = mel_spectrogram(
        audio.reshape(1, -1).float(),
        n_fft=1024, num_mels=128, sampling_rate=SPK_SR,
        hop_size=256, win_size=1024, fmin=0, fmax=12000,
    ).transpose(1, 2)
    return enc(mels.to(device=device, dtype=dtype))[0].float().cpu()


def open_spk_memmap(path, n_rows):
    """按 [N, 2048] fp16 读回。训练侧用 memmap —— 43.5w 条也只有 1.78GB，
    且不进常驻内存（audio_codes 存 Python list 撑到 78GB 的教训）。"""
    size = os.path.getsize(path)
    if size != n_rows * SPK_ITEMSIZE:
        raise SystemExit(
            f"{path} 有 {size / SPK_ITEMSIZE:.1f} 行，但 jsonl 是 {n_rows} 行 —— "
            f"两者必须逐行对应。删掉重抽，或确认用的是同一份 jsonl。")
    return np.memmap(path, dtype=np.float16, mode="r", shape=(n_rows, SPK_DIM))
