"""对 ref 音频做 VAD 裁剪，供语音克隆使用。

处理三件事：切掉首尾静音、把过长的句内静音压到上限、丢弃裁剪后过短的段。
输出 ref_wavs_vad/<uttid>.wav（16 kHz 单声道）+ vad_report.json。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "ref_wavs"
DST_DIR = ROOT / "ref_wavs_vad"

TOP_DB = 30          # 低于峰值 30 dB 视为静音
MARGIN = 0.08        # 每段前后保留的余量（秒），避免削掉爆破音/尾音
MAX_GAP = 0.20       # 句内静音上限（秒）
EDGE_PAD = 0.05      # 裁剪后首尾各留一点，避免起音过突
SR = 16000


def trim(y: np.ndarray, sr: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    intervals = librosa.effects.split(y, top_db=TOP_DB)
    if len(intervals) == 0:
        return y, []

    margin = int(MARGIN * sr)
    max_gap = int(MAX_GAP * sr)
    pad = np.zeros(int(EDGE_PAD * sr), dtype=y.dtype)

    pieces: list[np.ndarray] = [pad]
    prev_end: int | None = None
    for start, end in intervals:
        s = max(0, start - margin)
        e = min(len(y), end + margin)
        if prev_end is not None:
            gap = s - prev_end
            if gap > 0:
                # 句内静音保留真实内容但截断到上限，节奏不至于被压平
                keep = min(gap, max_gap)
                pieces.append(y[prev_end : prev_end + keep])
        pieces.append(y[s:e])
        prev_end = e
    pieces.append(pad)
    return np.concatenate(pieces), [(int(a), int(b)) for a, b in intervals]


def main(argv: list[str]) -> int:
    src_files = sorted(SRC_DIR.glob("*.mp3")) + sorted(SRC_DIR.glob("*.wav"))
    if not src_files:
        print(f"{SRC_DIR} 里没有音频", file=sys.stderr)
        return 1
    DST_DIR.mkdir(exist_ok=True)

    report = []
    for p in src_files:
        y, sr = librosa.load(p, sr=SR, mono=True)
        y2, intervals = trim(y, sr)
        out = DST_DIR / (p.stem + ".wav")
        sf.write(out, y2, sr)
        rec = {
            "src": p.name,
            "out": out.name,
            "sec_before": round(len(y) / sr, 2),
            "sec_after": round(len(y2) / sr, 2),
            "segments": len(intervals),
            "voiced_sec": round(sum(b - a for a, b in intervals) / sr, 2),
        }
        rec["cut_sec"] = round(rec["sec_before"] - rec["sec_after"], 2)
        report.append(rec)

    (DST_DIR / "vad_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    tot_b = sum(r["sec_before"] for r in report)
    tot_a = sum(r["sec_after"] for r in report)
    print(f"处理 {len(report)} 个 ref：{tot_b:.1f}s -> {tot_a:.1f}s（削掉 {tot_b - tot_a:.1f}s）")
    for r in sorted(report, key=lambda x: -x["cut_sec"])[:10]:
        print(f"  {r['src'][:12]} {r['sec_before']:6.2f}s -> {r['sec_after']:5.2f}s"
              f"  段数{r['segments']:2d}  削 {r['cut_sec']:5.2f}s")
    short = [r for r in report if r["sec_after"] < 1.0]
    if short:
        print(f"警告：{len(short)} 个 ref 裁剪后不足 1s，克隆效果可能不稳："
              f"{[r['src'][:12] for r in short]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
