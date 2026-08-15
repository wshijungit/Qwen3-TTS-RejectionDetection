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


class V2TrainStep(torch.nn.Module):
    """把「手工拼 embedding + talker 前向 + loss」整个封进一个 forward。

    **不要退回直接调 `model.talker(...)`。** accelerate 的两件事都只作用于被
    `prepare` 的模块的 `forward`：
      - `mixed_precision="bf16"` 的 autocast 包的是顶层 forward；
      - 多卡时 DDP 的梯度 allreduce 由 `DDP.forward` 里的
        `Reducer::prepare_for_backward()` 触发。
    绕过 forward 直接调子模块，**autocast 不生效、DDP 也完全不同步**
    （不报错、loss 照降，每张卡各训各的）。实测：
        model.talker(x)               -> fp32（未 autocast）
        with acc.autocast(): talker(x) -> bf16
    """

    def __init__(self, qwen_model, sub_talker_weight):
        super().__init__()
        self.model = qwen_model
        self.sub_talker_weight = sub_talker_weight

    def forward(self, input_ids, codec_ids, text_embedding_mask,
                codec_embedding_mask, attention_mask, codec_0_labels, codec_mask):
        talker = self.model.talker

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
            use_cache=False,
        )
        hidden_states = outputs.hidden_states[0][-1]
        return outputs.loss + self.sub_talker_weight * _sub_talker_loss(
            talker, codec_ids, codec_mask, hidden_states
        )


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
    parser.add_argument("--attn", type=str, default="sdpa",
                        help="CUDA 上可传 flash_attention_2；NPU 上不可用，保持 sdpa/eager")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument(
        "--length-bucket", type=int, default=64,
        help="序列长度向上取整到该倍数。昇腾按形状编译算子，长度每变一次就重编译一次；"
             "分桶后形状收敛到少数几档，编译可跨 step 复用。CUDA 上无此问题，设 1 即关闭",
    )
    parser.add_argument(
        "--dtype", choices=["bf16", "fp32"], default="fp32",
        help="权重精度。bf16 = 参数/梯度/Adam 动量全 bf16（AdamW 用 zeros_like 建 "
             "state，参数 bf16 则动量也是 bf16），1.7B 静态约 12.7GiB，但 exp_avg_sq "
             "用 8 位尾数累积梯度平方，lr 2e-5 下更新量有下溢风险；"
             "fp32 = fp32 权重 + autocast bf16 计算（标准混合精度），静态约 25GiB，"
             "910B2 的 64GB 吃得下，收敛更稳。显存不紧就用 fp32")
    args = parser.parse_args()

    accelerator = Accelerator(gradient_accumulation_steps=args.grad_accum,
                              mixed_precision="bf16")

    MODEL_PATH = args.init_model_path
    load_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    qwen3tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH, torch_dtype=load_dtype, attn_implementation=args.attn,
    )
    config = AutoConfig.from_pretrained(MODEL_PATH)

    with open(args.train_jsonl) as f:
        train_data = [json.loads(l) for l in f if l.strip()]
    dataset = VoiceDesignTTSDataset(train_data, qwen3tts.processor, config,
                                    language=args.language,
                                    length_bucket=args.length_bucket)
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=dataset.collate_fn)

    step_module = V2TrainStep(qwen3tts.model, args.sub_talker_loss_weight)
    optimizer = AdamW(step_module.parameters(), lr=args.lr, weight_decay=0.01)
    model, optimizer, train_dataloader = accelerator.prepare(
        step_module, optimizer, train_dataloader
    )
    model.train()

    gstep = 0
    stop = False
    for epoch in range(args.num_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                # 走 wrapper 的 forward —— autocast 与 DDP 同步都靠它
                loss = model(**{k: v for k, v in batch.items()})

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
                stop = True
                break

        # 无论是正常跑完一个 epoch 还是被 --max_steps 截断，都要落盘，
        # 否则调试跑（--max_steps 50）训完磁盘上什么都没有
        _save(accelerator, model, args, MODEL_PATH, epoch)
        if stop:
            break


def _save(accelerator, model, args, MODEL_PATH, epoch):
    if accelerator.is_main_process:
        out_dir = os.path.join(args.output_model_path, f"checkpoint-epoch-{epoch}")
        shutil.copytree(MODEL_PATH, out_dir, dirs_exist_ok=True)

        # 保持 voice_design —— 官方脚本这里硬写 custom_voice，产出的 ckpt 会被
        # qwen3_tts_model.py:686 的 != "voice_design" 直接拒掉
        with open(os.path.join(MODEL_PATH, "config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
        cfg["tts_model_type"] = "voice_design"
        # 训练用的 language 决定 codec 前缀（Chinese→think+lang 5 格 /
        # Auto→nothink 4 格）。generate_voice_design 的 language 默认是 Auto，
        # 与此不一致就是静默失配，故记进 config 供推理侧对齐。
        cfg["v2_train_language"] = args.language
        with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)

        # _unwrap(model) 是 V2TrainStep，真模型在它的 .model 上
        real = _unwrap(model).model
        # 无论以什么精度训，一律 bf16 落盘：推理侧默认 bf16，且省一半磁盘。
        # 三元必须整体括起来，否则 dict comprehension 里优先级会错。
        sd = {k: (v.detach().cpu().to(torch.bfloat16) if v.is_floating_point()
                  else v.detach().cpu())
              for k, v in real.state_dict().items()
              if not k.startswith("speaker_encoder")}   # VoiceDesign 下本就为 None，防御性保留
        # 若源目录是分片 checkpoint，index.json 会让下面写的单文件失效，
        # 加载时静默读回原始权重 → 微调结果丢失
        idx = os.path.join(out_dir, "model.safetensors.index.json")
        if os.path.exists(idx):
            os.remove(idx)
        save_file(sd, os.path.join(out_dir, "model.safetensors"),
                  metadata={"format": "pt"})
        accelerator.print(f"已保存 {out_dir}")


if __name__ == "__main__":
    train()
