# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""VoiceDesign 微调用的 Dataset。

与官方 dataset.py（Base 模型 / 音色克隆）的区别，逐条对应 VoiceDesign 的推理路径
`Qwen3TTSForConditionalGeneration.generate`（modeling_qwen3_tts.py:2076-2185）：

1. **instruct 前缀**：instruct 经 `<|im_start|>user\\n{ins}<|im_end|>\\n` 包装后
   tokenize，作为变长前缀排在最前，只占文本通道，codec 通道留空
   （推理侧 :2076-2080 就是把它 embed 后 append 到 talker 输入序列最前）。
2. **speaker 槽位可选**：不给 `spk` 时沿用 VoiceDesign 原本的
   `speaker_embed is None` 路径（:2213-2215），codec 前缀里没有 speaker 那一格。
   给了 `spk` 则在 head 与 `[codec_pad, codec_bos]` 之间插一格，与推理侧的
   `cat(emb0, speaker_embed.view(1,1,-1), emb1)`（:2216-2218）逐格对齐，块宽 5→6。
   **该格的 embedding 是被整格覆盖的，不是相加**（官方 sft_12hz.py:91 同）。
   官方 dataset.py 把位置写死成 `[:, 6, :]`，这里不行 —— instruct 是变长前缀，
   每条样本的绝对位置都不同，故 collate 额外吐 `spk_pos`。
3. **无 ref_audio**：不做音色克隆，也不在训练图里跑 speaker_encoder ——
   spk 向量在数据准备阶段用 Base 的 encoder 离线抽好（见 spk_encoder.py）。
   官方那行本就是 `.detach()`，encoder 全程冻结，离线与在线等价。
4. **带 language 的 codec 前缀**：语言非 Auto 时推理侧走
   `[think, think_bos, language_id, think_eos]`（:2141-2147），而不是 Auto 的
   `[nothink, think_bos, think_eos]`。两者加上 `[codec_pad, codec_bos]` 后
   分别是 6/5 长，去掉末位 codec_bos 参与对齐，故块宽 5/4。

序列布局（block 宽度 nb：带语言 5、带语言+spk 6、Auto 4、Auto+spk 5）：

    位置    0 .. n_ins-1     | n_ins .. +3   | +nb 个对齐槽        | 文本 ...        | 音频码 ...
    文本    instruct tokens  | role(3 token) | pad×(nb-1), bos     | text, eos, pad  | pad
    codec   (空, mask off)   | (空, mask off)| think/lang/spk/pad  | codec_pad, bos  | codec_0

对齐块的 codec 逐格（带语言 + spk 的情况）：

    think, think_bos, lang, think_eos, [spk], codec_pad
                                        ^ 这一格的 embedding 被 spk 向量整格覆盖
"""
import numpy as np
import torch
from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
from torch.utils.data import Dataset


# 序列长度向上取整到该值的倍数。
#
# 昇腾的算子是**按形状编译**的：形状每变一次就现场编译一次内核（CPU 编译、
# NPU 等喂）。而 collate 原本按 batch 内最长样本 padding，每个 batch 长度都不同
# —— 实测 12 条真实样本产生 12 种形状，等于每步都在重编译。这正是 NPU 上
# 1.2s/step（batch=1）的主因，而 1.7B 在该规模下的真实计算量远小于此。
#
# 取整到 64 的倍数后，形状收敛到少数几档，编译结果可跨 step 复用。
# 代价是多算一点 padding（平均多 32 格，相对 100 量级的序列约 30%），
# 但省下的编译时间远大于此。CUDA 上无此问题（动态形状原生支持），
# 设为 1 即退化回原行为。
LENGTH_BUCKET = 64


class VoiceDesignTTSDataset(Dataset):
    def __init__(self, data_list, processor, config: Qwen3TTSConfig, language="Chinese",
                 lag_num=-1, length_bucket=LENGTH_BUCKET, spk=None):
        self.data_list = data_list
        self.processor = processor
        self.lag_num = lag_num
        self.config = config
        self.language = language
        self.length_bucket = max(1, int(length_bucket))
        # spk: None 或 [N, 2048] 的 memmap/数组，行号与 data_list 一一对应。
        # 给了就在 codec 前缀里多插一格 speaker（块宽 5→6）。
        self.spk = spk
        if spk is not None and len(spk) != len(data_list):
            raise ValueError(f"spk 有 {len(spk)} 行，data_list 有 {len(data_list)} 条，必须逐行对应")

    def __len__(self):
        return len(self.data_list)

    # ---- 与推理侧逐字一致，不要自己改格式 ----
    # modeling 侧对应 Qwen3TTSModel._build_assistant_text / _build_instruct_text
    def _build_assistant_text(self, text: str) -> str:
        return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"

    def _build_instruct_text(self, instruct: str) -> str:
        return f"<|im_start|>user\n{instruct}<|im_end|>\n"

    def _tokenize_texts(self, text) -> torch.Tensor:
        out = self.processor(text=text, return_tensors="pt", padding=True)
        ids = out["input_ids"]
        return ids.unsqueeze(0) if ids.dim() == 1 else ids

    def _codec_prefix(self):
        """返回 (codec 前缀 token 列表, 块宽 nb, speaker 相对位置)。

        与推理侧 :2180-2218 一致。speaker 那一格夹在 head 和 [codec_pad, codec_bos]
        中间：`cat(emb0, speaker_embed.view(1,1,-1), emb1)`。
        """
        tc = self.config.talker_config
        lang = (self.language or "auto").lower()
        if lang == "auto":
            head = [tc.codec_nothink_id, tc.codec_think_bos_id, tc.codec_think_eos_id]
        else:
            if lang not in tc.codec_language_id:
                raise NotImplementedError(f"Language {self.language} not implemented")
            head = [
                tc.codec_think_id,
                tc.codec_think_bos_id,
                tc.codec_language_id[lang],
                tc.codec_think_eos_id,
            ]
        # speaker 槽位占的 token id 无所谓 —— 它的 embedding 会被整格**覆盖**成
        # spk 向量（官方 sft_12hz.py:91 的 input_codec_embedding[:, 6, :] = ... 也是
        # 赋值而非相加）。填 codec_pad 只是为了不落在词表外。
        spk_rel = len(head) if self.spk is not None else -1
        mid = [tc.codec_pad_id] if self.spk is not None else []
        # 推理侧把 [codec_pad, codec_bos] 接在后面再丢掉末位 codec_bos 参与对齐
        prefix = head + mid + [tc.codec_pad_id]
        return prefix, len(prefix), spk_rel

    def __getitem__(self, idx):
        item = self.data_list[idx]
        text = item["text"]
        instruct = item.get("instruct", "") or ""
        audio_codes = torch.tensor(item["audio_codes"], dtype=torch.long)

        text_ids = self._tokenize_texts(self._build_assistant_text(text))
        # 去掉尾部的 <|im_end|>\n<|im_start|>assistant\n（5 token），与官方一致
        text_ids = text_ids[:, :-5]

        instruct_ids = (
            self._tokenize_texts(self._build_instruct_text(instruct))
            if instruct
            else torch.zeros((1, 0), dtype=torch.long)
        )

        out = {
            "text_ids": text_ids,          # 1, t   (role 3 token + 正文)
            "instruct_ids": instruct_ids,  # 1, n   (可为空)
            "audio_codes": audio_codes,    # t, 16
        }
        if self.spk is not None:
            # memmap 的行是 fp16，转 float32 交给上层（autocast 会再降回 bf16）
            out["spk"] = torch.from_numpy(np.asarray(self.spk[idx], dtype=np.float32))
        return out

    def collate_fn(self, batch):
        assert self.lag_num == -1
        tc = self.config.talker_config
        codec_prefix, nb, spk_rel = self._codec_prefix()

        lens = [
            b["instruct_ids"].shape[1] + b["text_ids"].shape[1] + b["audio_codes"].shape[0]
            for b in batch
        ]
        need = max(lens) + nb + 4                # end = lens+nb+3，留 1 格余量
        # 向上取整到分桶边界，让昇腾的编译结果能跨 step 复用（见 LENGTH_BUCKET）
        bk = self.length_bucket
        b_, t = len(batch), ((need + bk - 1) // bk) * bk

        input_ids = torch.zeros((b_, t, 2), dtype=torch.long)
        codec_ids = torch.zeros((b_, t, 16), dtype=torch.long)
        text_embedding_mask = torch.zeros((b_, t), dtype=torch.bool)
        codec_embedding_mask = torch.zeros((b_, t), dtype=torch.bool)
        codec_mask = torch.zeros((b_, t), dtype=torch.bool)
        attention_mask = torch.zeros((b_, t), dtype=torch.long)
        codec_0_labels = torch.full((b_, t), -100, dtype=torch.long)
        # instruct 是变长前缀，speaker 那一格的绝对位置**每条样本都不同**，
        # 不能像官方 sft_12hz.py 那样写死 [:, 6, :]。
        spk_pos = torch.zeros((b_,), dtype=torch.long) if spk_rel >= 0 else None
        spk_vec = torch.zeros((b_, 2048), dtype=torch.float32) if spk_rel >= 0 else None

        for i, data in enumerate(batch):
            ins_ids = data["instruct_ids"]
            text_ids = data["text_ids"]
            codecs = data["audio_codes"]
            codec_0 = codecs[:, 0]

            n_ins = ins_ids.shape[1]
            n_txt = text_ids.shape[1]       # = 3 (role) + 正文长度
            n_cod = codec_0.shape[0]

            # ---- 1. instruct 前缀：只走文本通道，codec 通道留空 ----
            if n_ins:
                input_ids[i, :n_ins, 0] = ins_ids[0]
                text_embedding_mask[i, :n_ins] = True
                # codec_embedding_mask 保持 False —— 推理侧这段也只有文本 embedding

            # ---- 2. role: <|im_start|>assistant\n ----
            r = n_ins
            input_ids[i, r:r + 3, 0] = text_ids[0, :3]

            # ---- 3. 对齐块：文本 pad×(nb-1)+bos，codec 为 think/language 前缀 ----
            a = r + 3
            input_ids[i, a:a + nb - 1, 0] = self.config.tts_pad_token_id
            input_ids[i, a + nb - 1, 0] = self.config.tts_bos_token_id
            input_ids[i, a:a + nb, 1] = torch.tensor(codec_prefix, dtype=torch.long)
            if spk_rel >= 0:
                spk_pos[i] = a + spk_rel
                spk_vec[i] = data["spk"]

            # ---- 4. 正文 + tts_eos，codec 侧**全部** codec_pad（含 eos 那格）----
            s = a + nb                      # 正文起点
            n_body = n_txt - 3              # 去掉 role 的正文长度
            input_ids[i, s:s + n_body, 0] = text_ids[0, 3:]
            input_ids[i, s + n_body, 0] = self.config.tts_eos_token_id
            input_ids[i, s:s + n_body + 1, 1] = tc.codec_pad_id

            # ---- 4b. codec_bos 独占一格，文本侧是 tts_pad ----
            # 推理侧（modeling:2224-2245 的 non-streaming 分支）把这两段分开拼：
            #   ① text_emb(正文)+tts_eos  叠  codec_pad × (n_body+1)
            #   ② tts_pad_embed           叠  emb(codec_bos)
            # 若把 tts_eos 与 codec_bos 并成一格，推理时模型会在一个训练中从未
            # 出现过的 (tts_pad, codec_bos) 组合上被要求吐第一帧 codec，且
            # (tts_eos, codec_pad) 也从未共现——不报错，只表现为训完推理就崩。
            # 官方 dataset.py:174/175 vs :190/191 也是分两格的。
            bpos = s + n_body + 1
            input_ids[i, bpos, 0] = self.config.tts_pad_token_id
            input_ids[i, bpos, 1] = tc.codec_bos_id

            # ---- 5. 音频码 ----
            c = bpos + 1
            input_ids[i, c:c + n_cod, 1] = codec_0
            input_ids[i, c + n_cod, 1] = tc.codec_eos_token_id
            input_ids[i, c:c + n_cod + 1, 0] = self.config.tts_pad_token_id
            codec_ids[i, c:c + n_cod, :] = codecs

            end = c + n_cod + 1
            # talker.forward 在**同一位置**算 CE（modeling 侧用不 shift 的裸 CE），
            # 所以 label 要摆在因果意义上的前一格：codec_bos 那格预测第一个音频码，
            # 最后一个音频码那格预测 codec_eos。
            # 三处必须成套——collate 左移一格 + 模型不 shift + 训练脚本不手工 shift。
            # 官方 sft_12hz.py 是「手工 shift + HF loss_function 再 shift」= 双 shift，
            # 不报错但训不出东西。
            lab = c - 1                       # = codec_bos 所在位置
            codec_0_labels[i, lab:lab + n_cod] = codec_0
            codec_0_labels[i, lab + n_cod] = tc.codec_eos_token_id

            text_embedding_mask[i, :end] = True
            codec_embedding_mask[i, a:end] = True   # 无 speaker 空洞
            codec_mask[i, c:c + n_cod] = True
            attention_mask[i, :end] = True

        out = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "text_embedding_mask": text_embedding_mask.unsqueeze(-1),
            "codec_embedding_mask": codec_embedding_mask.unsqueeze(-1),
            "codec_0_labels": codec_0_labels,
            "codec_ids": codec_ids,
            "codec_mask": codec_mask,
        }
        if spk_rel >= 0:
            out["spk_pos"] = spk_pos
            out["spk_vec"] = spk_vec
        return out
