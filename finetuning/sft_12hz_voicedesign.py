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

3. **speaker 槽位默认没有，给 --spk-file 才有**。不给时走 VoiceDesign 原本的
   `speaker_embed is None` 路径（modeling:2213-2215），codec 前缀里没有那一格，
   音色完全由 instruct 决定；给了则插一格并**整格覆盖**其 embedding
   （逐格布局见 dataset_voicedesign.py 文件头 §2）。训练图里始终不含
   speaker_encoder —— 向量在数据准备阶段离线抽好（spk_encoder.py）。

单阶段：不新增 text_head、不训 <think> 生成。instruct 只作输入侧条件，
输出端仍只有 codec。多阶段（text-head 解耦）留给后续。
"""
import argparse
import json
import os
import sys
import re
import shutil

# 仓库根必须顶到 sys.path 最前，**且必须在任何 qwen_tts 相关 import 之前**：
# qwen_tts 若是 editable 安装且指向别的副本（开发机实测 pip -e 指到了参考仓库
# /home/swx/workspace/Qwen3-TTS），就会 import 到那一份 —— 它没有本仓库对
# modeling 的三处改动（py3.9 注解、_compute_token_ce_loss 不二次 shift、
# to() 搬 speech_tokenizer），不报错，只是训练目标悄悄变成错位一格的任务。
#
# **顺序是关键**：dataset_voicedesign 自己就会 import qwen_tts，一旦它先执行，
# qwen_tts 已进 sys.modules，之后再改 sys.path 完全无效（子模块沿父包
# __path__ 解析）。守卫放在它下面 = 死代码 —— 本项目已经这样错过一次。
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch
from accelerate import Accelerator
from dataset_voicedesign import VoiceDesignTTSDataset
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

# 守卫可能因为 import 顺序被绕过（本项目已经踩过一次：守卫写在 dataset_voicedesign
# 之后 = 死代码）。所以再加一道**运行时断言** —— 不管是什么原因加载错了副本，
# 这里当场崩，而不是让训练用二次 shift 的 loss 跑完 19-21 小时。
def _assert_repo_modeling():
    import qwen_tts.core.models.modeling_qwen3_tts as _m

    if not hasattr(_m, "_compute_token_ce_loss"):
        raise SystemExit(
            f"加载到的 qwen_tts 不是本仓库那份：\n  {_m.__file__}\n"
            f"  期望在 {_REPO_ROOT} 之下。\n"
            "它缺少 _compute_token_ce_loss —— 会退回 HF 的 loss_function，"
            "而 collate 已把 label 左移一格，二者叠加就是**二次 shift**："
            "不报错、loss 照降，但训出来的东西是错的。\n"
            "多半是 pip -e 装到了别的副本。设 PYTHONPATH 指向本仓库根即可。")


_assert_repo_modeling()
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

    def __init__(self, qwen_model, sub_talker_weight, spk_drop_prob=0.0,
                 sub_talker_frame_bucket=0):
        super().__init__()
        self.model = qwen_model
        self.sub_talker_weight = sub_talker_weight
        self.spk_drop_prob = spk_drop_prob
        self.sub_talker_frame_bucket = sub_talker_frame_bucket

    def forward(self, input_ids, codec_ids, text_embedding_mask,
                codec_embedding_mask, attention_mask, codec_0_labels, codec_mask,
                spk_pos=None, spk_vec=None):
        talker = self.model.talker

        # 文本通道：必须过 text_projection，与推理侧一致（见文件头 §1）
        input_text_embedding = talker.text_projection(
            talker.model.text_embedding(input_ids[:, :, 0])
        ) * text_embedding_mask
        input_codec_embedding = (
            talker.model.codec_embedding(input_ids[:, :, 1]) * codec_embedding_mask
        )
        # speaker 那一格：**整格覆盖**，不是相加。推理侧是
        # cat(emb0, speaker_embed.view(1,1,-1), emb1)（modeling:2216-2218）——
        # 那一格里只有 spk 向量，没有任何 codec token 的 embedding 掺进去。
        # 写成 += 会让训练与推理在这一格上不一致，且不报错。
        if spk_pos is not None:
            b = torch.arange(input_ids.shape[0], device=input_ids.device)
            v = spk_vec.to(input_codec_embedding.dtype)
            if self.spk_drop_prob > 0 and self.training:
                # 按概率整条置零 = 退回「没有 speaker」的行为，让模型别把
                # 音色当成唯一线索、instruct 退化成摆设。默认 0（用户要求先跑
                # 最简版本）—— **drop=0 时训完必须永远给 spk，不给就是失配**。
                keep = (torch.rand(v.shape[0], device=v.device) >= self.spk_drop_prob)
                v = v * keep.unsqueeze(-1).to(v.dtype)
            input_codec_embedding = input_codec_embedding.index_put(
                (b, spk_pos), v, accumulate=False)
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
            talker, codec_ids, codec_mask, hidden_states,
            frame_bucket=self.sub_talker_frame_bucket,
        )


# 显存水位记录：NPU 侧观察到 ~7MB/步 的逐步爬升（§21），要判断某个改动有没有
# 止住它，必须能看到**斜率**而不只是峰值。
#
# 用**滑动窗口**斜率，不是"自基准点的累计平均"：首版用了累计平均，结果早期
# 一次性的编译开销把均值长期拖高——NPU 侧实测水位数百步纹丝不动（54.24→54.25GB），
# 累计斜率却还显示 +7.8MB/步，害人以为仍在漏。窗口斜率只看最近这一段，
# 早期尖峰滚出窗口就不再影响判断。
_MEM_HIST = []
_MEM_WIN = 20          # 参与拟合的最近日志点数


def _mem_note(gstep):
    """返回 ' | mem 54.25GB (近 20 点 +0.01MB/步)'；设备不支持则返回空串。"""
    try:
        import torch as _t

        if hasattr(_t, "npu") and _t.npu.is_available():
            free, total = _t.npu.mem_get_info(0)
        elif _t.cuda.is_available():
            free, total = _t.cuda.mem_get_info(0)
        else:
            return ""
        used = (total - free) / 2 ** 20        # MiB
    except Exception:
        return ""
    _MEM_HIST.append((gstep, used))
    if len(_MEM_HIST) > _MEM_WIN:
        _MEM_HIST.pop(0)
    if len(_MEM_HIST) < 3:
        return f" | mem {used / 1024:.2f}GB"
    # 最小二乘拟合窗口内的 (step, MiB)，比首末两点差稳得多
    n = len(_MEM_HIST)
    sx = sum(p[0] for p in _MEM_HIST); sy = sum(p[1] for p in _MEM_HIST)
    sxx = sum(p[0] * p[0] for p in _MEM_HIST); sxy = sum(p[0] * p[1] for p in _MEM_HIST)
    den = n * sxx - sx * sx
    slope = (n * sxy - sx * sy) / den if den else 0.0
    return f" | mem {used / 1024:.2f}GB (近 {n} 点 {slope:+.2f}MB/步)"


def _sub_talker_loss(talker, codec_ids, codec_mask, hidden_states, frame_bucket=0):
    """sub-talker（1..15 号码本）的 loss。

    label 在 collate 里已左移一格，这里对应地用 hidden[:-1] 预测 codec[1:]。

    **frame_bucket：把帧数向上取整到它的倍数**（0/1 = 关闭，保持原行为）。

    为什么需要：`hidden_states[:, :-1][target]` 是布尔掩码索引，产出
    `[N_frames, hidden]` —— N_frames 是这一批音频的**实际总帧数**，与主路径的
    LENGTH_BUCKET 无关，**完全没有分桶**。按真实时长分布估算（batch=8），
    54000 步里会出现约 **524 种不同形状**，而昇腾按形状编译算子，
    每种形状要单独编一次内核并占一份 workspace。
    524 > 默认 `ACLNN_CACHE_LIMIT=300`，缓存还会颠簸 —— 这与 NPU 侧观察到的
    「~7MB/步 稳定泄漏、与 sdpa/eager 无关、CUDA 上复现不了」完全吻合。
    分桶到 64 的倍数后形状降到 **12 种**。

    padding 行对 loss **零影响**：codec_ids 那份补 0（embedding 查表需要合法
    下标，不能填 -100），labels 那份补 -100，而 `_compute_token_ce_loss` 用的是
    `ignore_index=-100` + `reduction='mean'`，被忽略的项分子分母都不计入。
    """
    target = codec_mask[:, 1:]
    hs = hidden_states[:, :-1][target]
    ids = codec_ids[:, 1:][target]
    if hs.numel() == 0 or ids.numel() == 0:
        return hidden_states.new_zeros(())
    labels = None
    if frame_bucket and frame_bucket > 1:
        pad = (-hs.shape[0]) % frame_bucket
        if pad:
            hs = torch.cat([hs, hs.new_zeros(pad, hs.shape[1])], dim=0)
            ids = torch.cat([ids, ids.new_zeros(pad, ids.shape[1])], dim=0)
        labels = ids[:, 1:].clone()
        if pad:
            labels[-pad:] = -100
    _, loss = talker.forward_sub_talker_finetune(ids, hs, labels=labels)
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
        "--save-total-limit", type=int, default=2,
        help="最多保留几个 step checkpoint（epoch checkpoint 不受限）。"
             "每个 ckpt 约 4.3GB，全量 batch=8 一个 epoch 有 5 万多步，"
             "save-every=500 就是 108 个 = 约 470GB —— 不限量必写爆磁盘",
    )
    parser.add_argument(
        "--save-every", type=int, default=0,
        help="每 N 个 micro-batch 步存一次（0=只在 epoch 末存）。注意计的是 "
             "dataloader 步、不是优化器步，grad_accum>1 时两者差 grad_accum 倍。"
             "43.5w 条即使 batch=8 一个 epoch 也是小时级，中途崩就丢整个 epoch",
    )
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
    parser.add_argument(
        "--optim-state", type=str, default=None,
        help="optimizer 状态文件路径。**分块训练必须给** —— 本脚本的热启是权重级的，"
             "不给这个参数，每块开头都是全新的 AdamW（m=0, v=0）：此时二阶矩估计还没"
             "建立，前若干步的更新量接近 lr·sign(g)，远大于稳态，等于每块挨一次冲击。"
             "5000 步一块、10 万步总量就是 20 多次。给了则块末写、块初读，"
             "使分块训练等价于连续训练。fp32 下约 13.6GB（1.7B × 2 个动量），"
             "每次覆盖写，只占一份")
    parser.add_argument(
        "--sub-talker-frame-bucket", type=int, default=0,
        help="把 sub-talker 的帧数向上取整到该倍数（0=关闭）。"
             "sub-talker 走布尔掩码索引，形状 = 这批音频的实际总帧数，"
             "与 --length-bucket 无关、完全动态：batch=8 下约 524 种形状，"
             "超过默认 ACLNN_CACHE_LIMIT=300。昇腾按形状编译算子，"
             "这是 NPU 侧「~7MB/步 显存泄漏」的头号嫌疑。设 64 可降到 12 种。"
             "padding 行 label 填 -100，对 loss 零影响（已实测逐位相同）")
    parser.add_argument(
        "--spk-file", type=str, default=None,
        help="prepare_data.py --spk_out 产出的裸 fp16 文件（[N,2048]，行与 train_jsonl 逐行对应）。"
             "给了就在 codec 前缀里多插一格 speaker，块宽 5→6")
    parser.add_argument(
        "--spk-drop-prob", type=float, default=0.0,
        help="训练时按此概率把 spk 向量**置零**（注意：不是去掉那一格）。默认 0。"
             "无论 drop 与否，序列永远是 6 格块 —— 而推理侧不传 speaker 走的是 5 格块，"
             "两者不是一回事。所以训完做「纯 instruct」推理时仍要带这一格、传全零向量；"
             "完全不传 spk 永远是失配。调到 0.1-0.3 可让同一个 ckpt 兼顾两种用法")
    args = parser.parse_args()

    accelerator = Accelerator(gradient_accumulation_steps=args.grad_accum,
                              mixed_precision="bf16")

    MODEL_PATH = args.init_model_path
    load_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    qwen3tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH, torch_dtype=load_dtype, attn_implementation=args.attn,
    )
    config = AutoConfig.from_pretrained(MODEL_PATH)

    # audio_codes 压成 numpy int16 再存：码值 ≤3071，int16 足够。
    # Python list of int 每个元素是独立对象，43.5w 条实测约 31KB/条 = 13GB/进程，
    # 8 卡 DDP 各自加载一份 ≈ 104GB/节点，再叠上每 rank 约 7GB 的模型 CPU 副本，
    # 很可能直接把节点内存打爆。压成 int16 后约 2.5KB/条 = 1.05GB/进程。
    # Dataset 侧 torch.tensor(..., dtype=torch.long) 吃 numpy int16 无需改动。
    train_data = []
    with open(args.train_jsonl, encoding="utf-8") as f:   # encoding 必须给：C locale 下读中文会崩
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if "audio_codes" in d:
                d["audio_codes"] = np.asarray(d["audio_codes"], dtype=np.int16)
            train_data.append(d)
    # spk 用 memmap 打开：43.5w × 2048 fp16 = 1.78GB，不进常驻内存。
    # 行数与 jsonl 对不上会当场崩（open_spk_memmap 里查），不会静默错位。
    spk = None
    if args.spk_file:
        from spk_encoder import open_spk_memmap

        spk = open_spk_memmap(args.spk_file, len(train_data))
        accelerator.print(f"speaker 向量 {spk.shape} ← {args.spk_file}"
                          f"（drop_prob={args.spk_drop_prob}）")
    dataset = VoiceDesignTTSDataset(train_data, qwen3tts.processor, config,
                                    language=args.language,
                                    length_bucket=args.length_bucket, spk=spk)
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=dataset.collate_fn)

    step_module = V2TrainStep(qwen3tts.model, args.sub_talker_loss_weight,
                              spk_drop_prob=args.spk_drop_prob,
                              sub_talker_frame_bucket=args.sub_talker_frame_bucket)
    optimizer = AdamW(step_module.parameters(), lr=args.lr, weight_decay=0.01)
    model, optimizer, train_dataloader = accelerator.prepare(
        step_module, optimizer, train_dataloader
    )
    model.train()

    # 恢复 optimizer 状态（分块训练用）。放在 prepare 之后 —— 此时 optimizer
    # 里的 param 引用已经是 prepare 后的那些，state_dict 的 key 才对得上。
    if args.optim_state and os.path.exists(args.optim_state):
        sd = torch.load(args.optim_state, map_location="cpu")
        optimizer.load_state_dict(sd)
        n = sum(1 for v in optimizer.state.values() if v)
        accelerator.print(f"已恢复 optimizer 状态 ← {args.optim_state}"
                          f"（{n} 个参数带动量）")
    elif args.optim_state:
        accelerator.print(f"optimizer 状态文件不存在，本块从零开始: {args.optim_state}")

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
                accelerator.print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}"
                                  + _mem_note(gstep))
            gstep += 1
            if args.save_every > 0 and gstep % args.save_every == 0:
                _save(accelerator, model, args, MODEL_PATH, f"step{gstep}")
            if args.max_steps > 0 and gstep >= args.max_steps:
                accelerator.print(f"到达 --max_steps {args.max_steps}，提前结束")
                stop = True
                break

        # 无论是正常跑完一个 epoch 还是被 --max_steps 截断，都要落盘，
        # 否则调试跑（--max_steps 50）训完磁盘上什么都没有
        _save(accelerator, model, args, MODEL_PATH, epoch)
        # 块末写 optimizer 状态：分块训练靠它把 AdamW 动量接上（见 --optim-state）
        if args.optim_state and accelerator.is_main_process:
            os.makedirs(os.path.dirname(os.path.abspath(args.optim_state)) or ".", exist_ok=True)
            tmp = args.optim_state + ".tmp"
            torch.save(optimizer.state_dict(), tmp)
            os.replace(tmp, args.optim_state)   # 原子替换，崩在写一半不会留下坏文件
            accelerator.print(f"已保存 optimizer 状态 → {args.optim_state}")
        if stop:
            break


def _save(accelerator, model, args, MODEL_PATH, tag):
    """tag 为 epoch 序号或 'stepN'。"""
    if accelerator.is_main_process:
        name = f"checkpoint-epoch-{tag}" if isinstance(tag, int) else f"checkpoint-{tag}"
        out_dir = os.path.join(args.output_model_path, name)
        # 跳过**顶层**的权重文件（下面会用训练后的权重重写），但子目录里的必须留：
        # speech_tokenizer/model.safetensors 有 682MB，且 from_pretrained 强制从
        # <ckpt>/speech_tokenizer 加载（modeling_qwen3_tts.py:1947-1963）。
        # 用 ignore_patterns 会**递归**作用于每层子目录，把它一起漏掉 —— 产出的
        # ckpt 加载即 OSError，--save-every 存的盘全是废的、中途崩了也热启不了。
        def _skip_top_weights(d, names):
            if os.path.abspath(d) != os.path.abspath(MODEL_PATH):
                return set()          # 子目录一律照拷
            return {n for n in names
                    if n.endswith((".safetensors", ".bin", ".pt"))
                    and os.path.isfile(os.path.join(d, n))}

        # 热启（MODEL_PATH 指向上一轮的 checkpoint）时若 gstep 撞回同一个序号，
        # out_dir 就等于 MODEL_PATH，copytree 会边走边往自己里面拷 → SameFileError
        # 在第 N 小时把训练崩掉。此时源文件本来就在原地，直接跳过。
        if os.path.abspath(out_dir) != os.path.abspath(MODEL_PATH):
            shutil.copytree(MODEL_PATH, out_dir, dirs_exist_ok=True,
                            ignore=_skip_top_weights)

        # 保持 voice_design —— 官方脚本这里硬写 custom_voice，产出的 ckpt 会被
        # qwen3_tts_model.py:686 的 != "voice_design" 直接拒掉
        with open(os.path.join(MODEL_PATH, "config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
        cfg["tts_model_type"] = "voice_design"
        # 训练用的 language 决定 codec 前缀（Chinese→think+lang 5 格 /
        # Auto→nothink 4 格）。generate_voice_design 的 language 默认是 Auto，
        # 与此不一致就是静默失配，故记进 config 供推理侧对齐。
        cfg["v2_train_language"] = args.language
        # 训练时有没有 speaker 那一格，决定推理侧必须传什么。drop_prob=0 时
        # 模型从没见过「没有 spk」的输入，推理不给就是静默失配（说不清哪错了，
        # 只表现为音质/音色乱）。记进 config 供推理侧对齐。
        cfg["v2_train_spk"] = bool(args.spk_file)
        cfg["v2_train_spk_drop_prob"] = args.spk_drop_prob
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

        # 滚动清理旧的 step checkpoint（epoch checkpoint 保留）
        if not isinstance(tag, int) and args.save_total_limit > 0:
            root = args.output_model_path
            # 用正则严格匹配纯数字后缀：有人手工留个 checkpoint-step500.bak 目录，
            # int("500.bak") 会在第 N 小时抛异常直接搞死训练
            # 按**落盘时间**而不是 step 序号排：热启是权重级的，gstep 从 0 重数，
            # 而目录里还留着上一轮的 checkpoint-step54000。按序号排的话新存的
            # step500 永远最小 —— 存完当场被自己删掉，热启后的十几个小时里
            # --save-every 完全失效，再崩一次又是全丢。
            steps = sorted(
                (d for d in os.listdir(root)
                 if re.fullmatch(r"checkpoint-step\d+", d)
                 and os.path.isdir(os.path.join(root, d))),
                key=lambda d: os.path.getmtime(os.path.join(root, d)),
            )
            for old in steps[: -args.save_total_limit]:
                shutil.rmtree(os.path.join(root, old), ignore_errors=True)
                accelerator.print(f"清理旧 ckpt {old}")
        accelerator.print(f"已保存 {out_dir}")


if __name__ == "__main__":
    train()
