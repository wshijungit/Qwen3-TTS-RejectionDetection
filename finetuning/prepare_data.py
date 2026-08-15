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

import argparse
import json
import os

from qwen_tts import Qwen3TTSTokenizer

BATCH_INFER_NUM = 32

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="NPU 上传 npu:0")
    parser.add_argument("--tokenizer_model_path", type=str, default="Qwen/Qwen3-TTS-Tokenizer-12Hz")
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    args = parser.parse_args()

    # 不用 device_map：torch 2.1 + torch_npu 上 meta 快速加载路径会炸
    # （param[...] → torch.cuda._lazy_init → "Torch not compiled with CUDA enabled"），
    # 先 CPU 加载，非 CPU 设备再整体搬过去
    tokenizer_12hz = Qwen3TTSTokenizer.from_pretrained(args.tokenizer_model_path)
    if not str(args.device).startswith("cpu"):
        import torch

        tokenizer_12hz.model.to(args.device)
        tokenizer_12hz.device = torch.device(args.device)

    # encoding 必须显式给：locale 为 C 的容器里读写中文会崩
    with open(args.input_jsonl, encoding="utf-8") as f:
        total_lines = [json.loads(l) for l in f if l.strip()]

    # 断点续跑：43.5w 条抽码要跑数小时，原实现全攒内存、跑完才落盘，
    # 中途崩一次全部重来。改为逐 batch 追加写出，重启时按已写行数跳过。
    #
    # 注意：续跑会改变 batch 的分组方式，而 batch encode **不具备 padding 不变性**
    # ——同一条音频与不同长度的样本同 batch，个别处在 VQ 边界上的帧会翻到邻近码字
    # （实测 0.8s 音频单独 encode vs 与 5s 同 batch，10 帧中 2 帧不同）。
    # 对音质无影响（邻近码字），但**不要指望 codes 逐位可复现**。
    done = 0
    if os.path.exists(args.output_jsonl):
        with open(args.output_jsonl, encoding="utf-8") as f:
            done = sum(1 for l in f if l.strip())
        if done:
            print(f"续跑：已有 {done} 行，跳过", flush=True)
    if done >= len(total_lines):
        print("已全部完成"); return
    total_lines = total_lines[done:]

    def flush(fh, lines, audios):
        enc = tokenizer_12hz.encode(audios)
        for code, line in zip(enc.audio_codes, lines):
            line["audio_codes"] = code.cpu().tolist()
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")
        fh.flush()
        lines.clear()
        audios.clear()

    batch_lines, batch_audios = [], []
    n = done
    with open(args.output_jsonl, "a", encoding="utf-8") as fh:
        for line in total_lines:
            batch_lines.append(line)
            batch_audios.append(line["audio"])
            if len(batch_lines) >= BATCH_INFER_NUM:
                flush(fh, batch_lines, batch_audios)
                n += BATCH_INFER_NUM
                if n % (BATCH_INFER_NUM * 50) == 0:
                    print(f"  {n}/{done + len(total_lines)}", flush=True)
        if batch_audios:
            n += len(batch_audios)
            flush(fh, batch_lines, batch_audios)
    print(f"完成 {n} 行 → {args.output_jsonl}")


if __name__ == "__main__":
    main()
