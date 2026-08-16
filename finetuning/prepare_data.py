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
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # 同目录的 spk_encoder

from qwen_tts import Qwen3TTSTokenizer

# batch 32 在全量真实数据上会 NPU OOM：VAD 后最长 24s（p99=10.5s，>8s 占 3.5%），
# batch 内补零到最长样本时 tokenizer 的 torch.cdist 中间张量按 B×T 放大，
# 实测 32 条含一条长样本时一次要 25.3GiB（卡 61GiB，已占 26GiB）。
# 降到 8：峰值 ∝ max_T×B 降 4 倍，代价是抽码批次数 4 倍（约 8-15h，断点续跑）。
BATCH_INFER_NUM = 8

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="NPU 上传 npu:0")
    parser.add_argument("--tokenizer_model_path", type=str, default="Qwen/Qwen3-TTS-Tokenizer-12Hz")
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, default=None)
    # ---- speaker embedding（可选；给了 --spk_out 才抽）----
    # 与 audio_codes 同一遍抽，**续跑时两者必须同步截断**，否则第 i 行的 codes
    # 会配上第 j 条的音色，且训练时看不出任何异常。
    parser.add_argument("--spk_out", type=str, default=None,
                        help="speaker 向量落盘路径（裸 fp16，每行 2048 维）。不给则不抽")
    parser.add_argument("--spk_model_path", type=str, default=None,
                        help="Base 权重目录 —— speaker_encoder 只在 tts_model_type=base 里有")
    parser.add_argument("--spk-only", action="store_true",
                        help="只抽 speaker 向量，不抽 audio_codes。此时 --input_jsonl "
                             "应直接指向**训练要用的那份 jsonl**（通常是已抽好的 "
                             "codes.jsonl），逐行对应地产出 --spk_out，不需要 "
                             "--output_jsonl / --tokenizer_model_path")
    args = parser.parse_args()
    if args.spk_out and not args.spk_model_path:
        raise SystemExit("--spk_out 需要同时给 --spk_model_path（指向 Base 权重）")
    if args.spk_only:
        if not args.spk_out:
            raise SystemExit("--spk-only 需要 --spk_out")
    elif not args.output_jsonl:
        raise SystemExit("需要 --output_jsonl（或用 --spk-only 只抽 speaker 向量）")

    # ---- --spk-only：只抽 speaker 向量 ----
    # 与 codes 同一遍抽（coupled 模式）省一次 wav 读取，但把两件事绑死了：
    # codes 已经抽完就没法再补 spk。这里提供独立模式 —— 对齐由 --input_jsonl
    # **自身的行顺序**保证，第 i 行的 audio 抽出第 i 行向量，天然不可能错位。
    if args.spk_only:
        import hashlib

        import librosa

        from spk_encoder import SPK_DIM, SPK_ITEMSIZE, SPK_SR, extract_spk, load_speaker_encoder

        # 只留 audio 字段：codes.jsonl 里的 audio_codes 是 Python list of int，
        # 43.5w 条整份读进来约 10.5GB（sft 侧为此专门压成了 numpy int16）。
        # 这条路径只需要每行的路径，约 100MB。
        audios, first = [], None
        with open(args.input_jsonl, encoding="utf-8") as f:
            for ln, l in enumerate(f, 1):
                if not l.strip():
                    continue
                d = json.loads(l)
                if first is None:
                    first = d
                if "audio" not in d:
                    raise SystemExit(
                        f"{args.input_jsonl} 第 {ln} 行没有 audio 字段，"
                        f"有的是 {sorted(d)[:6]}")
                audios.append(d["audio"])
        if first is not None and "audio_codes" not in first:
            print("提示：输入不含 audio_codes，看起来不是 codes.jsonl。"
                  "spk 必须与**训练实际用的那份 jsonl** 逐行对应，"
                  "顺序不同就会整体错位。", flush=True)

        # 输入指纹：行数相同但内容/顺序变了的话，单看行数完全发现不了 ——
        # 续跑会把旧输入抽的前 done 行留下，与新输入整体错位。训练侧
        # open_spk_memmap 只查总行数，也拦不住。这类错训练不报错，
        # 只是每条音频配了别人的音色。
        fp = hashlib.md5()
        for a in audios:
            fp.update(a.encode("utf-8")); fp.update(b"\0")
        fp = fp.hexdigest()
        meta_path = args.spk_out + ".meta"

        done = 0
        if os.path.exists(args.spk_out):
            size = os.path.getsize(args.spk_out)
            done = size // SPK_ITEMSIZE
            if size % SPK_ITEMSIZE:      # 崩在写一行的中间
                with open(args.spk_out, "rb+") as f:
                    f.truncate(done * SPK_ITEMSIZE)
                print(f"续跑：丢弃尾部 {size % SPK_ITEMSIZE} 字节的残行", flush=True)
            if done:
                if not os.path.exists(meta_path):
                    raise SystemExit(
                        f"{args.spk_out} 已有 {done} 行，但找不到 {meta_path} —— "
                        f"无法确认它是不是用同一份输入抽的。\n"
                        f"（旧版本产出的文件没有 .meta）删掉 {args.spk_out} 重抽。")
                with open(meta_path, encoding="utf-8") as f:
                    meta = json.load(f)
                if meta.get("md5") != fp or meta.get("n") != len(audios):
                    raise SystemExit(
                        f"{args.spk_out} 是用**另一份输入**抽的，续跑会整体错位：\n"
                        f"  已有: {meta.get('n')} 行, md5 {str(meta.get('md5'))[:12]}, "
                        f"输入 {meta.get('input')}\n"
                        f"  现在: {len(audios)} 行, md5 {fp[:12]}, 输入 {args.input_jsonl}\n"
                        f"删掉 {args.spk_out} 和 {meta_path} 重抽。")
            if done > len(audios):
                raise SystemExit(
                    f"{args.spk_out} 已有 {done} 行 > 输入 {len(audios)} 行 —— "
                    f"输入很可能换过。删掉 {args.spk_out} 重抽。")
            if done == len(audios):
                print(f"已全部完成（{done} 行）"); return
            if done:
                print(f"续跑：已有 {done} 行，跳过", flush=True)

        enc = load_speaker_encoder(args.spk_model_path, device=args.device)
        print(f"已加载 speaker_encoder（{args.spk_model_path}）", flush=True)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"n": len(audios), "md5": fp,
                       "input": os.path.abspath(args.input_jsonl),
                       "spk_model": os.path.abspath(args.spk_model_path)}, f)

        t0, n = time.time(), 0
        with open(args.spk_out, "ab") as fh:
            for i, p in enumerate(audios[done:], start=done):
                try:
                    wav, _ = librosa.load(p, sr=SPK_SR, mono=True)
                    v = extract_spk(enc, wav, SPK_SR, device=args.device)
                except Exception as e:
                    raise RuntimeError(f"抽 speaker 向量失败（第 {i} 行）: {p} "
                                       f"—— {type(e).__name__}: {e}")
                # 第一条就查维度：0.6B-Base 的 enc_dim 是 1024，而它
                # speaker_encoder_config 存在、权重也自洽，load 阶段查不出来。
                # 不在这里拦，43.5w 条会**完整跑完数小时**才在末尾行数对不上时报错，
                # 且续跑接上正确模型后文件是 1024/2048 维交错的废数据。
                if v.shape[0] != SPK_DIM:
                    raise SystemExit(
                        f"抽出的向量是 {v.shape[0]} 维，不是 {SPK_DIM} 维 —— "
                        f"{args.spk_model_path} 多半是 0.6B-Base（enc_dim=1024）。"
                        f"V2 用的 VoiceDesign 是 1.7B，必须配 1.7B-Base。")
                fh.write(v.numpy().astype(np.float16).tobytes())
                n += 1
                if n % 2000 == 0:
                    fh.flush()
                    el = time.time() - t0
                    print(f"  {i + 1}/{len(audios)}  {el / n * 1000:.1f} ms/条，"
                          f"剩约 {(len(audios) - i - 1) * el / n / 3600:.2f}h", flush=True)
        got = os.path.getsize(args.spk_out) // SPK_ITEMSIZE
        if got != len(audios):
            raise SystemExit(f"产出 {got} 行 != 输入 {len(audios)} 行")
        print(f"完成 {got} 行 → {args.spk_out}"
              f"（{(time.time() - t0) / max(n, 1) * 1000:.1f} ms/条）")
        return

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
        # 只回扫尾部：43.5w 条的输出有数 GB，f.read() 会造成等量内存尖峰。
        with open(args.output_jsonl, "rb+") as f:
            size = f.seek(0, os.SEEK_END)
            if size:
                f.seek(size - 1)
                if f.read(1) != b"\n":
                    cut, blk = -1, 1 << 20
                    pos = size
                    while pos > 0 and cut < 0:
                        pos = max(0, pos - blk)
                        f.seek(pos)
                        chunk = f.read(min(blk, size - pos))
                        i = chunk.rfind(b"\n")
                        if i >= 0:
                            cut = pos + i
                    f.truncate(cut + 1 if cut >= 0 else 0)
                    print(f"续跑：丢弃尾部 {size - (cut + 1)} 字节的残行", flush=True)
        with open(args.output_jsonl, encoding="utf-8") as f:
            done = sum(1 for l in f if l.strip())
        if done:
            print(f"续跑：已有 {done} 行，跳过", flush=True)
    # spk 文件必须跟着 jsonl 一起截断到 done 行。少截/多截都会让 codes 与音色
    # 整体错位 —— 训练不报错，只是每条音频配了别人的音色。
    # 注意这里**不能**写成 `if os.path.exists(spk_out)`：文件不存在时 cur=0，
    # 若 jsonl 已有 done 行（上一轮没开 --spk_out 抽的），跳过检查就会从第 done 条
    # 开始只补 spk 的后半段 —— 产出 jsonl N 行、spk N-done 行，零告警。
    # 训练侧 open_spk_memmap 会拦下，但那是数小时之后的事了。
    if args.spk_out:
        from spk_encoder import SPK_ITEMSIZE

        want = done * SPK_ITEMSIZE
        cur = os.path.getsize(args.spk_out) if os.path.exists(args.spk_out) else 0
        if cur < want:
            # truncate 到更大的值会**用 0 补齐**，静默产出一批全零音色向量。
            # 正常流程里 spk 先写、jsonl 后写，spk 只可能多不可能少。
            raise SystemExit(
                f"{args.spk_out} 只有 {cur / SPK_ITEMSIZE:.1f} 行，少于 jsonl 的 {done} 行。\n"
                f"常见原因：上一轮没带 --spk_out 就抽了 codes，这轮才加上。\n"
                f"**不要删 {args.output_jsonl}**（那是数小时的产出），"
                f"用独立模式补抽即可：\n"
                f"  python prepare_data.py --spk-only --device {args.device} \\\n"
                f"    --input_jsonl {args.output_jsonl} \\\n"
                f"    --spk_out {args.spk_out} --spk_model_path {args.spk_model_path}\n"
                f"（先删掉已有的 {args.spk_out}）")
        if cur > want:
            with open(args.spk_out, "rb+") as f:
                f.truncate(want)
            print(f"续跑：spk 文件 {cur // SPK_ITEMSIZE} 行 → 截到 {done} 行", flush=True)
    if done > len(total_lines):
        raise SystemExit(
            f"输出已有 {done} 行 > 输入 {len(total_lines)} 行 —— 输入文件很可能换过，"
            f"续跑会错配。删掉 {args.output_jsonl} 重跑，或确认输入无误")
    if done >= len(total_lines):
        print("已全部完成"); return
    total_lines = total_lines[done:]

    spk_enc = None
    if args.spk_out:
        from spk_encoder import SPK_SR, extract_spk, load_speaker_encoder

        spk_enc = load_speaker_encoder(args.spk_model_path, device=args.device)
        print(f"已加载 speaker_encoder（{args.spk_model_path}）", flush=True)

    def flush(fh, lines, audios, sfh=None):
        # spk 先算：它比 tokenizer.encode 更容易在坏音频上崩，先崩掉就不会留下
        # 「codes 写了、spk 没写」的错位状态。写盘顺序同理 —— spk 先落、jsonl 后落，
        # 崩在中间时 spk 只会多不会少，续跑截断逻辑才成立。
        if spk_enc is not None:
            import librosa

            vecs = []
            for p in audios:
                # 报错必须带上是哪个 wav：43.5w 条跑到一半崩，没有文件名没法定位，
                # 而这类失败是确定性的（同一条每次都崩），续跑会卡死在同一批。
                try:
                    wav, _ = librosa.load(p, sr=SPK_SR, mono=True)
                    vecs.append(extract_spk(spk_enc, wav, SPK_SR, device=args.device))
                except Exception as e:
                    raise RuntimeError(f"抽 speaker 向量失败: {p} —— {type(e).__name__}: {e}")
            v = torch.stack(vecs).to(torch.float16).numpy()
            if v.shape[1] != 2048:
                raise RuntimeError(f"speaker 向量维度 {v.shape[1]} != 2048")
            sfh.write(v.tobytes())
            sfh.flush()

        enc = tokenizer_12hz.encode(audios)
        # py3.9 没有 zip(strict=True)。若 encode 返回条数少于输入（异常音频等），
        # 那些行会静默消失，而续跑是按行数跳过的 —— 一旦少一行，之后的
        # audio/text 与 codes 就整体错位且不报错。宁可当场崩。
        #
        # 不用 assert：python -O 会把它整个剥掉，这是 43.5w 条的对齐保命线，
        # 不该寄托在解释器开关上。且这类失败是**确定性**的（同一个 wav 每次都崩），
        # 续跑会卡死在同一批，所以报错必须带上是哪几个文件，好让人删掉再续。
        if len(enc.audio_codes) != len(lines):
            raise RuntimeError(
                f"encode 返回 {len(enc.audio_codes)} 条但输入 {len(lines)} 条，"
                f"本批文件：{[l.get('audio') for l in lines]}")
        for code, line in zip(enc.audio_codes, lines):
            # 不要写成 line["audio_codes"] = ...：line 是 total_lines 里的 dict，
            # 原地写入后引用仍在，codes 不会被回收，43.5w 条跑到尾部会积累约 13GB
            fh.write(json.dumps({**line, "audio_codes": code.cpu().tolist()},
                                ensure_ascii=False) + "\n")
        fh.flush()
        lines.clear()
        audios.clear()

    import contextlib

    batch_lines, batch_audios = [], []
    n, nb = done, 0
    with contextlib.ExitStack() as st:
        fh = st.enter_context(open(args.output_jsonl, "a", encoding="utf-8"))
        sfh = st.enter_context(open(args.spk_out, "ab")) if args.spk_out else None
        for line in total_lines:
            batch_lines.append(line)
            batch_audios.append(line["audio"])
            if len(batch_lines) >= BATCH_INFER_NUM:
                flush(fh, batch_lines, batch_audios, sfh)
                n += BATCH_INFER_NUM
                nb += 1
                if nb % 50 == 0:          # 按 batch 计数，不依赖 n 是否为 32 的倍数
                    print(f"  {n}/{done + len(total_lines)}", flush=True)
        if batch_audios:
            n += len(batch_audios)
            flush(fh, batch_lines, batch_audios, sfh)
    print(f"完成 {n} 行 → {args.output_jsonl}"
          + (f"，speaker 向量 → {args.spk_out}" if args.spk_out else ""))


if __name__ == "__main__":
    main()
