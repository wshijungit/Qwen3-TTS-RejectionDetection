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
"""VoiceDesign 单阶段微调：instruct（自然语言风格描述）→ 控制说话腔调。

与官方 sft_12hz.py（Base 模型 / 音色克隆）的三处关键差异：

1. **补上 text_projection**。官方脚本用 `talker.model.text_embedding(ids)` 直接
   当输入 embedding，但推理侧走的是 `talker.text_projection(get_text_embeddings()(ids))`
   （modeling_qwen3_tts.py:1978/2079/2124/2177…共 7 处，全在推理路径）。
   实测两者不等价：范数 0.46 → 1.98，余弦相似度 0.0058（近乎正交）。
   训练不对齐推理，训出来的东西推理时用不上。

2. **不做手工 shift**。官方是 `inputs[:, :-1]` + `labels[:, 1:]`，而模型侧
   HF 的 loss_function 还会再 shift 一次 = 双 shift。本脚本约定：
   collate 把 label 摆到因果前一格，模型侧用不 shift 的裸 CE，脚本不动。
   三处成套，改一处就错。

3. **无 speaker_encoder / ref_audio**。VoiceDesign 的 speaker_embed 为 None
   （modeling:2165-2171），codec 前缀里没有 speaker 槽位，音色完全由 instruct 决定。

单阶段：不新增 text_head、不训 <think> 生成。instruct 只作输入侧条件，
输出端仍只有 codec。多阶段（text-head 解耦）留给后续。
"""
import argparse
import json
import os
import shutil

import torch
from accelerate import Accelerator
from dataset_voicedesign import VoiceDesignTTSDataset
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


def _sub_talker_loss(talker, codec_ids, codec_mask, hidden_states):
    """sub-talker（1..15 号码本）的 loss。

    label 在 collate 里已左移一格，这里对应地用 hidden[:-1] 预测 codec[1:]。
    """
    target = codec_mask[:, 1:]
    hs = hidden_states[:, :-1][target]
    ids = codec_ids[:, 1:][target]
    if hs.numel() == 0 or ids.numel() == 0:
        return hidden_states.new_zeros(())
    _, loss = talker.forward_sub_talker_finetune(ids, hs)
    return loss


def train():
    parser = argparse.ArgumentParser(
        description="Qwen3-TTS VoiceDesign 单阶段微调（instruct 控腔调）"
    )
    parser.add_argument("--init_model_path", type=str,
                        default="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
    parser.add_argument("--output_model_path", type=str, default="output")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--sub_talker_loss_weight", type=float, default=0.3)
    parser.add_argument("--language", type=str, default="Chinese",
                        help="决定 codec 前缀走 think+language 还是 Auto 的 nothink")
    parser.add_argument("--max_steps", type=int, default=-1, help="调试用，>0 时提前停")
    parser.add_argument("--attn", type=str, default="flash_attention_2",
                        help="NPU 上 flash_attention_2 不可用，改 sdpa 或 eager")
    parser.add_argument("--log_every", type=int, default=10)
    args = parser.parse_args()

    accelerator = Accelerator(gradient_accumulation_steps=args.grad_accum,
                              mixed_precision="bf16")

    MODEL_PATH = args.init_model_path
    qwen3tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, attn_implementation=args.attn,
    )
    config = AutoConfig.from_pretrained(MODEL_PATH)

    with open(args.train_jsonl) as f:
        train_data = [json.loads(l) for l in f if l.strip()]
    dataset = VoiceDesignTTSDataset(train_data, qwen3tts.processor, config,
                                    language=args.language)
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=dataset.collate_fn)

    optimizer = AdamW(qwen3tts.model.parameters(), lr=args.lr, weight_decay=0.01)
    model, optimizer, train_dataloader = accelerator.prepare(
        qwen3tts.model, optimizer, train_dataloader
    )
    model.train()

    gstep = 0
    for epoch in range(args.num_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                talker = _unwrap(model).talker
                dev = next(model.parameters()).device

                input_ids = batch["input_ids"].to(dev)
                codec_ids = batch["codec_ids"].to(dev)
                text_embedding_mask = batch["text_embedding_mask"].to(dev)
                codec_embedding_mask = batch["codec_embedding_mask"].to(dev)
                attention_mask = batch["attention_mask"].to(dev)
                codec_0_labels = batch["codec_0_labels"].to(dev)
                codec_mask = batch["codec_mask"].to(dev)

                # 文本通道：必须过 text_projection，与推理侧一致（见文件头 §1）
                input_text_embedding = talker.text_projection(
                    talker.model.text_embedding(input_ids[:, :, 0])
                ) * text_embedding_mask
                # codec 通道：VoiceDesign 无 speaker 槽位，不注入 speaker_embedding
                input_codec_embedding = (
                    talker.model.codec_embedding(input_ids[:, :, 1]) * codec_embedding_mask
                )
                input_embeddings = input_text_embedding + input_codec_embedding

                for i in range(1, 16):
                    emb = talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
                    input_embeddings = input_embeddings + emb * codec_mask.unsqueeze(-1)

                # 不手工 shift —— label 已在 collate 里左移（见文件头 §2）
                outputs = talker(
                    inputs_embeds=input_embeddings,
                    attention_mask=attention_mask,
                    labels=codec_0_labels,
                    output_hidden_states=True,
                )
                hidden_states = outputs.hidden_states[0][-1]
                loss = outputs.loss + args.sub_talker_loss_weight * _sub_talker_loss(
                    talker, codec_ids, codec_mask, hidden_states
                )

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            if step % args.log_every == 0:
                accelerator.print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")
            gstep += 1
            if args.max_steps > 0 and gstep >= args.max_steps:
                accelerator.print(f"到达 --max_steps {args.max_steps}，提前结束")
                break
        if args.max_steps > 0 and gstep >= args.max_steps:
            break

        if accelerator.is_main_process:
            out_dir = os.path.join(args.output_model_path, f"checkpoint-epoch-{epoch}")
            shutil.copytree(MODEL_PATH, out_dir, dirs_exist_ok=True)

            # 保持 voice_design —— 官方脚本这里硬写 custom_voice，产出的 ckpt 会被
            # qwen3_tts_model.py:686 的 != "voice_design" 直接拒掉
            with open(os.path.join(MODEL_PATH, "config.json"), encoding="utf-8") as f:
                cfg = json.load(f)
            cfg["tts_model_type"] = "voice_design"
            with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2, ensure_ascii=False)

            sd = {k: v.detach().to("cpu")
                  for k, v in _unwrap(model).state_dict().items()
                  if not k.startswith("speaker_encoder")}   # VoiceDesign 用不到
            save_file(sd, os.path.join(out_dir, "model.safetensors"))
            accelerator.print(f"已保存 {out_dir}")


if __name__ == "__main__":
    train()
