#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证多卡训练时梯度是否真的在同步。

**为什么需要单独验**：accelerate 的 DDP allreduce 由 `DDP.forward` 里的
`Reducer::prepare_for_backward()` 触发。若训练代码绕过被 `prepare` 的模块的
forward（直接调子模块），allreduce **完全不发生**——不报错、loss 照降，
每张卡各训各的，最后只存 rank0。这个坑本项目踩过一次（见 commit 14c3e42）。

原理：各 rank 喂**不同**数据跑一步。梯度同步正常时，同一权重的 grad 在所有
rank 上应完全相同（allreduce 平均过）；不同步则各不相同。

用法（在 NPU 上，卡数按实际改）：
    torchrun --nproc_per_node 2 check_ddp_sync.py --model_path <VoiceDesign> \\
        --train_jsonl <train_codes.jsonl>

预期输出：`✅ 梯度已同步`。若打 ❌，说明训练循环又绕过 forward 了，
多卡训练等于白跑，**不要继续跑正式训练**。
"""
import argparse
import json
import os
import sys

import torch

try:
    import torch_npu  # noqa: F401
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--language", default="Chinese")
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="fp32")
    args = ap.parse_args()

    from accelerate import Accelerator
    from dataset_voicedesign import VoiceDesignTTSDataset
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
    from sft_12hz_voicedesign import V2TrainStep
    from transformers import AutoConfig

    acc = Accelerator(mixed_precision="bf16")
    if acc.num_processes < 2:
        print("❌ 只有 1 个进程，这个检查需要至少 2 卡：\n"
              "   torchrun --nproc_per_node 2 check_ddp_sync.py ...")
        sys.exit(1)

    dt = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    q = Qwen3TTSModel.from_pretrained(args.model_path, torch_dtype=dt,
                                      attn_implementation=args.attn)
    cfg = AutoConfig.from_pretrained(args.model_path)

    with open(args.train_jsonl) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    if len(rows) < acc.num_processes:
        rows = (rows * acc.num_processes)[: acc.num_processes]
    ds = VoiceDesignTTSDataset(rows, q.processor, cfg, language=args.language)

    model = acc.prepare(V2TrainStep(q.model, 0.3))

    # 关键：每个 rank 喂**不同**的样本。同步正常则梯度仍应一致。
    idx = acc.process_index % len(ds)
    batch = ds.collate_fn([ds[idx]])
    batch = {k: v.to(acc.device) for k, v in batch.items()}
    acc.print(f"共 {acc.num_processes} 个进程，各喂不同样本")

    loss = model(**batch)
    acc.backward(loss)

    # 取一个确定存在且参与前向的权重
    name = "model.talker.model.text_embedding.weight"
    real = model.module if hasattr(model, "module") else model
    params = dict(real.named_parameters())
    if name not in params:
        cand = [k for k in params if k.endswith("text_embedding.weight")]
        acc.print(f"找不到 {name}，改用 {cand[:1]}")
        name = cand[0]
    grad = params[name].grad
    if grad is None:
        acc.print(f"❌ {name} 没有梯度，检查该权重是否参与前向")
        sys.exit(1)

    stat = grad.detach().float().sum().reshape(1)
    gathered = acc.gather(stat.to(acc.device)).cpu()
    if acc.is_main_process:
        vals = [f"{v:.6e}" for v in gathered.tolist()]
        print(f"\n各 rank 的 grad.sum(): {vals}")
        spread = (gathered.max() - gathered.min()).abs().item()
        scale = max(gathered.abs().max().item(), 1e-12)
        if spread / scale < 1e-5:
            print("✅ 梯度已同步（各 rank 一致）—— DDP allreduce 正常")
        else:
            print(f"❌ 梯度未同步！相对差 {spread / scale:.3e}\n"
                  "   说明训练循环绕过了被 prepare 模块的 forward，多卡等于白跑。\n"
                  "   检查 sft_12hz_voicedesign.py 是否仍在调 model(**batch) 而非子模块。")
            sys.exit(1)


if __name__ == "__main__":
    main()
