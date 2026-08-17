# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""V2 效果评测：同一句话 × 对机腔/对人腔 × 固定音色。

**V2 存在的意义就是这张表**：真实数据里「文本」和「拒识标签」是强相关的
（说"打开空调"基本都是对机的），所以拒识模型很容易退化成背文本。V2 要做的是
**同一句文本，两种腔调都合成一遍**，把这个相关性打断。

这个脚本就是验它到底成没成：

  - 同一条 text，配 addressee=机器 / addressee=人 两种 instruct
  - 两者用**同一个 spk 向量**，把音色变量摁住 —— 只剩腔调在变
  - 输出时长/能量对比 + wav，人耳复核

speaker 向量的来源（--spk-source）：
  file:<路径>[:<行号>]  从 prepare_data.py --spk-only 产出的 spk.f16 里取一行
  wav:<音频路径>        现抽（需要 --spk_model_path 指向 1.7B-Base）
  zero                  全零向量 = 「只要 instruct 控制、不指定音色」
                        注意这**不等于不传**：不传是 5 格块，训练时是 6 格块

用法：
    python eval_v2_spk.py --ckpt <微调后的 ckpt> \\
        --spk-source file:/path/spk.f16:0 --device npu:0 --out-dir ./eval_out
"""
import argparse
import json
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 默认评测句：都是「文本本身看不出对谁说」的，这类最能体现 instruct 的作用。
# 若文本本身就强烈指向某一方（"小P小P打开空调"），两种腔调的差异会被文本盖住。
DEFAULT_TEXTS = [
    "打开空调",
    "有点冷",
    "算了不用了",
    "音量调小一点",
    "前面路口左转",
    "帮我看看还有多远",
]

# instruct 的写法要与训练时一致 —— 训练用的是 gemini 的 evidence + reason
# 拼成的自然语言（见 prepare_v2_data.py 的 build_instruct），不是标签词。
# 这里模仿那个风格，而不是写"对机腔"三个字。
INSTRUCT = {
    "机器": "说话人正在与车机语音助手交互；语气明确、音量偏大、吐字清晰，"
            "像是在下达指令。应当拒识",
    "人":   "说话人正在与车内其他乘客闲聊；语气随意、音量偏低、语速自然，"
            "带有对话中的迟疑。不应拒识",
}


def load_spk(source, spk_model_path, device):
    """返回 (向量 np.float32[2048], 描述字符串)。"""
    if source == "zero":
        return np.zeros(2048, dtype=np.float32), "全零向量"
    kind, _, rest = source.partition(":")
    if kind == "file":
        path, _, idx = rest.rpartition(":")
        if not path:                       # 没写行号
            path, idx = rest, "0"
        idx = int(idx)
        from spk_encoder import SPK_DIM, SPK_ITEMSIZE

        n = os.path.getsize(path) // SPK_ITEMSIZE
        if not 0 <= idx < n:
            raise SystemExit(f"{path} 只有 {n} 行，取不到第 {idx} 行")
        v = np.memmap(path, dtype=np.float16, mode="r",
                      shape=(n, SPK_DIM))[idx]
        return np.asarray(v, dtype=np.float32), f"{os.path.basename(path)} 第 {idx} 行"
    if kind == "wav":
        if not spk_model_path:
            raise SystemExit("--spk-source wav:... 需要 --spk_model_path（1.7B-Base）")
        import librosa

        from spk_encoder import SPK_SR, extract_spk, load_speaker_encoder

        enc = load_speaker_encoder(spk_model_path, device=device)
        wav, _ = librosa.load(rest, sr=SPK_SR, mono=True)
        return (extract_spk(enc, wav, SPK_SR, device=device).numpy().astype(np.float32),
                f"现抽自 {os.path.basename(rest)}")
    raise SystemExit(f"--spk-source 不认识: {source}（应为 file:/wav:/zero）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="微调后的 VoiceDesign checkpoint")
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--spk-source", default="zero",
                    help="file:<spk.f16>[:行号] / wav:<音频> / zero")
    ap.add_argument("--spk_model_path", default=None, help="--spk-source wav: 时需要")
    ap.add_argument("--texts", default=None, help="每行一句的文本文件；不给用内置 6 句")
    ap.add_argument("--out-dir", default="./eval_v2_out")
    ap.add_argument("--language", default=None,
                    help="不给则读 ckpt 里的 v2_train_language —— 训练与推理的"
                         "language 必须一致，否则 codec 前缀格数就对不上")
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--temperature", type=float, default=0.9)
    args = ap.parse_args()

    import torch

    try:
        import torch_npu  # noqa: F401
    except Exception:
        pass
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.ckpt, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    lang = args.language or cfg.get("v2_train_language", "Chinese")
    trained_spk = bool(cfg.get("v2_train_spk", False))
    print(f"ckpt: {args.ckpt}\n  v2_train_spk={trained_spk} "
          f"drop_prob={cfg.get('v2_train_spk_drop_prob', 0)} language={lang}")

    spk, spk_desc = (None, "不传")
    if trained_spk:
        spk, spk_desc = load_spk(args.spk_source, args.spk_model_path, args.device)
    elif args.spk_source != "zero":
        print("⚠️  ckpt 没有 v2_train_spk，忽略 --spk-source")
    print(f"  speaker 向量: {spk_desc}")

    texts = DEFAULT_TEXTS
    if args.texts:
        with open(args.texts, encoding="utf-8") as f:
            texts = [l.strip() for l in f if l.strip()]

    m = Qwen3TTSModel.from_pretrained(args.ckpt, torch_dtype=torch.bfloat16,
                                      attn_implementation=args.attn)
    m.model.to(args.device)
    m.device = torch.device(args.device)

    rows = []
    for i, t in enumerate(texts):
        r = {"text": t}
        for tag, ins in INSTRUCT.items():
            wavs, sr = m.generate_voice_design(
                text=t, instruct=ins, language=lang,
                spk_embedding=spk, temperature=args.temperature)
            w = wavs[0]
            p = os.path.join(args.out_dir, f"{i:02d}_{tag}.wav")
            sf.write(p, w, sr)
            r[tag] = {"dur": len(w) / sr, "rms": float(np.sqrt((w.astype(np.float64) ** 2).mean())),
                      "wav": p}
        rows.append(r)
        a, b = r["机器"], r["人"]
        print(f"  [{i}] {t!r:22} 机器 {a['dur']:.2f}s / 人 {b['dur']:.2f}s  "
              f"时长差 {abs(a['dur'] - b['dur']) / max(a['dur'], b['dur']) * 100:.1f}%")

    # 汇总：时长差是最容易自动看的代理指标 —— 对机腔通常更短更利落。
    # **它只是代理，不是结论**，最终还得听。
    difs = [abs(r["机器"]["dur"] - r["人"]["dur"]) / max(r["机器"]["dur"], r["人"]["dur"])
            for r in rows]
    mean_dif = sum(difs) / len(difs) * 100
    print(f"\n平均时长差 {mean_dif:.1f}%（{len(rows)} 句）")
    if mean_dif < 10:
        print("⚠️  差异偏小 —— instruct 可能没起作用，或被 speaker 那一格盖过去了。\n"
              "    对照：换 --spk-source zero 再跑一次，若差异明显变大，"
              "说明是 spk 压制了 instruct（ref==target 的 shortcut）")
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"ckpt": args.ckpt, "spk": spk_desc, "language": lang,
                   "mean_dur_diff_pct": mean_dif, "rows": rows}, f,
                  ensure_ascii=False, indent=2)
    print(f"wav 与 summary.json 都在 {args.out_dir}")


if __name__ == "__main__":
    main()
