"""按 voice_clone_testset.md 逐轮做语音克隆合成。

ref 音频取自 ref_wavs_vad/<uttid>.wav（原始 mp3 由 error_sessions_model.html 内嵌
base64 导出，再经 vad_trim_refs.py 切掉首尾静音、压缩过长句内静音）。
助手轮（ref 列为 —）不合成。输出 out_testset/<对话>/<轮次>.wav。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent
MODEL = "/mnt/data6/swx/model/dots.tts-soar"
REF_DIR = ROOT / "ref_wavs_vad"
REF_EXT = ".wav"
OUT_DIR = ROOT / "out_testset"


def parse_markdown(path: Path) -> list[dict]:
    scenes: list[dict] = []
    cur_s = cur_d = None
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^## (场景 \d+.*)$", line)
        if m:
            cur_s = {"title": m.group(1).strip(), "desc": "", "dialogs": []}
            scenes.append(cur_s)
            cur_d = None
            continue
        m = re.match(r"^### (对话 [\d.]+.*)$", line)
        if m and cur_s is not None:
            cur_d = {"title": m.group(1).strip(), "note": "", "turns": []}
            cur_s["dialogs"].append(cur_d)
            continue
        if line.startswith("> ") and cur_d is not None and not cur_d["note"]:
            cur_d["note"] = line[2:].strip()
            continue
        if cur_s is not None and cur_d is None and line.strip().startswith("**覆盖**"):
            cur_s["desc"] = line.strip()
        if line.strip().startswith("|") and cur_d is not None:
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            if len(c) >= 5 and re.match(r"^\d+$", c[0]):
                cur_d["turns"].append(
                    {
                        "idx": int(c[0]),
                        "spk": c[1],
                        "ref": c[2],
                        "ref_text": c[3],
                        "text": c[4],
                    }
                )
    return scenes


def dialog_slug(title: str) -> str:
    m = re.match(r"对话 ([\d.]+)", title)
    return m.group(1) if m else re.sub(r"\W+", "_", title)


def main() -> int:
    from dots_tts.runtime import DotsTtsRuntime
    from dots_tts.utils.util import seed_everything

    scenes = parse_markdown(ROOT / "voice_clone_testset.md")
    todo = [
        (s, d, t)
        for s in scenes
        for d in s["dialogs"]
        for t in d["turns"]
        if t["ref"].endswith(".wav")
    ]
    print(f"待合成 {len(todo)} 轮", flush=True)

    runtime = DotsTtsRuntime.from_pretrained(MODEL, precision="bfloat16")

    manifest = []
    for n, (scene, dialog, turn) in enumerate(todo, 1):
        ref = REF_DIR / (Path(turn["ref"]).stem + REF_EXT)
        if not ref.exists():
            print(f"[{n}/{len(todo)}] 跳过，ref 缺失: {ref.name}", flush=True)
            continue
        slug = dialog_slug(dialog["title"])
        out = OUT_DIR / slug / f"turn{turn['idx']:02d}.wav"
        out.parent.mkdir(parents=True, exist_ok=True)

        # 每轮固定 seed，保证可复现
        seed_everything(42)
        result = runtime.generate(
            text=turn["text"],
            prompt_audio_path=str(ref),
            prompt_text=turn["ref_text"],
            language="ZH",
            normalize_text=True,
        )
        audio = result["audio"].float().cpu().squeeze().numpy()
        sr = result["sample_rate"]
        sf.write(out, audio, sr)
        dur = len(audio) / sr
        manifest.append(
            {
                "scene": scene["title"],
                "dialog": dialog["title"],
                "slug": slug,
                "turn": turn["idx"],
                "speaker": turn["spk"],
                "ref": turn["ref"],
                "ref_file": ref.name,
                "ref_text": turn["ref_text"],
                "text": turn["text"],
                "wav": str(out.relative_to(ROOT)),
                "seconds": round(dur, 2),
            }
        )
        print(f"[{n}/{len(todo)}] {slug} 轮{turn['idx']} {dur:5.2f}s  {turn['text'][:24]}", flush=True)

        if n % 10 == 0:
            torch.cuda.empty_cache()

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    (OUT_DIR / "testset.json").write_text(
        json.dumps(scenes, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"完成 {len(manifest)} 条 -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
