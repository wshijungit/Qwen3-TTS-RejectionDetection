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
        # 崩在"写了半行"时，末尾会留下一条没有换行的残行。直接按行数计会把它
        # 当成完整一行跳过，下一条 JSON 追加在它后面粘成非法行 —— 实测 6 条输入
        # 只剩 5 行且 1 行损坏，且要等数小时后训练启动 json.loads 时才炸。
        # 故先把文件截断到最后一个完整换行处。
        with open(args.output_jsonl, "rb+") as f:
            raw = f.read()
            if raw and not raw.endswith(b"\n"):
                cut = raw.rfind(b"\n")
                f.seek(0)
                f.truncate(cut + 1 if cut >= 0 else 0)
                print(f"续跑：丢弃尾部 {len(raw) - (cut + 1)} 字节的残行", flush=True)
        with open(args.output_jsonl, encoding="utf-8") as f:
            done = sum(1 for l in f if l.strip())
        if done:
            print(f"续跑：已有 {done} 行，跳过", flush=True)
    if done > len(total_lines):
        raise SystemExit(
            f"输出已有 {done} 行 > 输入 {len(total_lines)} 行 —— 输入文件很可能换过，"
            f"续跑会错配。删掉 {args.output_jsonl} 重跑，或确认输入无误")
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
    n, nb = done, 0
    with open(args.output_jsonl, "a", encoding="utf-8") as fh:
        for line in total_lines:
            batch_lines.append(line)
            batch_audios.append(line["audio"])
            if len(batch_lines) >= BATCH_INFER_NUM:
                flush(fh, batch_lines, batch_audios)
                n += BATCH_INFER_NUM
                nb += 1
                if nb % 50 == 0:          # 按 batch 计数，不依赖 n 是否为 32 的倍数
                    print(f"  {n}/{done + len(total_lines)}", flush=True)
        if batch_audios:
            n += len(batch_audios)
            flush(fh, batch_lines, batch_audios)
    print(f"完成 {n} 行 → {args.output_jsonl}")


if __name__ == "__main__":
    main()
