"""把合成结果打包成单文件听测页（mp3 全部 base64 内嵌）。

用法: python build_clone_html.py [输出路径]
依赖: ffmpeg（wav -> mp3 转码）
"""

from __future__ import annotations

import base64
import html
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "out_testset"
REF_DIR = ROOT / "ref_wavs_vad"      # VAD 裁剪后，实际用于克隆
REF_RAW_DIR = ROOT / "ref_wavs"      # 原始，仅供对比
DEFAULT_HTML = ROOT / "assets" / "voice_clone_testset.html"

MP3_BITRATE = "64k"      # 单声道语音，64k 足够听测
MP3_RATE = "24000"       # 采样率降到 24k，页面体积减半，音色判断不受影响


def to_mp3_b64(src: Path, cache: dict[Path, str]) -> str:
    if src in cache:
        return cache[src]
    if src.suffix == ".mp3":
        data = src.read_bytes()
    else:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            dst = Path(tmp.name)
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
             "-ac", "1", "-ar", MP3_RATE, "-b:a", MP3_BITRATE, str(dst)],
            check=True,
        )
        data = dst.read_bytes()
        dst.unlink()
    b64 = base64.b64encode(data).decode()
    cache[src] = b64
    return b64


def audio_tag(b64: str, cls: str = "") -> str:
    c = f' class="{cls}"' if cls else ""
    return f'<audio{c} controls preload="none" src="data:audio/mpeg;base64,{b64}"></audio>'


CSS = """
:root{--bg:#12151c;--panel:#1b202b;--panel2:#222838;--line:#2c3344;
      --txt:#e6e9f0;--dim:#98a1b3;--acc:#4ea1ff;--drv:#ff9f43;--pas:#5ad1a8;--bot:#7c8496;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--txt);
     font:15px/1.65 "PingFang SC","Microsoft YaHei","Noto Sans CJK SC",sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:28px 20px 80px}
h1{font-size:26px;margin:0 0 6px}
.sub{color:var(--dim);margin:0 0 18px}
.legend{display:flex;gap:22px;flex-wrap:wrap;background:var(--panel);border:1px solid var(--line);
        border-radius:10px;padding:12px 18px;margin:14px 0 26px;font-size:14px}
.badge{display:inline-block;padding:1px 10px;border-radius:20px;font-size:13px;font-weight:600}
.badge.drv{background:#4a2c10;color:var(--drv);border:1px solid var(--drv)}
.badge.pas{background:#123528;color:var(--pas);border:1px solid var(--pas)}
.badge.bot{background:#2a3140;color:var(--dim);border:1px solid var(--line)}
section{margin:38px 0 10px}
h2{font-size:20px;border-left:4px solid var(--acc);padding-left:12px;margin:0 0 6px}
.sec-desc{color:var(--dim);font-size:14px;margin:0 0 4px;padding-left:16px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;
      padding:16px 18px;margin:18px 0}
.card-title{font-size:16px;font-weight:600;margin-bottom:4px}
.card-note{color:var(--dim);font-size:13px;margin-bottom:10px}
.card-note code{background:var(--panel2);padding:1px 5px;border-radius:4px;font-size:12px}
table{width:100%;border-collapse:collapse;font-size:14px}
th{background:var(--panel2);color:var(--dim);font-weight:600;text-align:left;
   padding:8px 10px;border:1px solid var(--line);white-space:nowrap}
td{padding:8px 10px;border:1px solid var(--line);vertical-align:top}
tr.bot-row td{color:var(--dim);background:rgba(255,255,255,.02)}
.c-idx{width:44px;text-align:center;color:var(--dim)}
.c-spk{width:78px;white-space:nowrap}
.c-text{min-width:220px}
.c-ref{min-width:200px;color:var(--dim);font-size:13px}
.uttid{display:block;font-size:11px;color:#69718a;font-family:ui-monospace,monospace;
       word-break:break-all;margin-bottom:3px}
audio{height:32px;width:210px;vertical-align:middle}
audio.ref{width:190px;height:28px}
.dur{color:var(--dim);font-size:12px;margin-left:6px}
.tag{display:inline-block;min-width:34px;font-size:11px;color:var(--acc);
     border:1px solid var(--line);border-radius:4px;padding:0 4px;margin-right:5px;text-align:center}
.tag.raw{color:var(--dim)}
.miss{color:#ff5f56;font-size:13px}
"""


def main(argv: list[str]) -> int:
    if not shutil.which("ffmpeg"):
        print("需要 ffmpeg", file=sys.stderr)
        return 1
    out_html = Path(argv[1]) if len(argv) > 1 else DEFAULT_HTML
    out_html.parent.mkdir(parents=True, exist_ok=True)

    scenes = json.loads((OUT_DIR / "testset.json").read_text(encoding="utf-8"))
    manifest = json.loads((OUT_DIR / "manifest.json").read_text(encoding="utf-8"))
    by_key = {(m["slug"], m["turn"]): m for m in manifest}

    vad_dur: dict[str, str] = {}
    vad_report = REF_DIR / "vad_report.json"
    if vad_report.exists():
        for r in json.loads(vad_report.read_text(encoding="utf-8")):
            vad_dur[Path(r["src"]).stem] = f'{r["sec_before"]}s→{r["sec_after"]}s'

    cache: dict[Path, str] = {}
    parts: list[str] = []
    n_syn = n_ref = 0

    for scene in scenes:
        parts.append('<section class="scene">')
        parts.append(f'<h2>{html.escape(scene["title"])}</h2>')
        if scene["desc"]:
            desc = scene["desc"].replace("**", "")
            parts.append(f'<p class="sec-desc">{html.escape(desc)}</p>')

        for dialog in scene["dialogs"]:
            slug = dialog["title"].split()[1] if " " in dialog["title"] else dialog["title"]
            note = dialog["note"].replace("**", "").replace("`", "")
            parts.append('<div class="card">')
            parts.append(f'<div class="card-title">{html.escape(dialog["title"])}</div>')
            if note:
                parts.append(f'<div class="card-note">{html.escape(note)}</div>')
            parts.append("<table><thead><tr>"
                         "<th>轮</th><th>说话人</th><th>合成文本</th>"
                         "<th>合成音频</th><th>ref（原始音频 + transcript）</th>"
                         "</tr></thead><tbody>")

            for turn in dialog["turns"]:
                is_bot = not turn["ref"].endswith(".wav")
                spk = turn["spk"]
                cls = "drv" if "主驾" in spk else ("pas" if "副驾" in spk else "bot")
                row_cls = ' class="bot-row"' if is_bot else ""
                parts.append(f"<tr{row_cls}>")
                parts.append(f'<td class="c-idx">{turn["idx"]}</td>')
                parts.append(f'<td class="c-spk"><span class="badge {cls}">{html.escape(spk)}</span></td>')
                parts.append(f'<td class="c-text">{html.escape(turn["text"])}</td>')

                if is_bot:
                    parts.append('<td class="c-ref" colspan="2">助手轮（机器），不合成</td>')
                else:
                    rec = by_key.get((slug, turn["idx"]))
                    if rec and (ROOT / rec["wav"]).exists():
                        b64 = to_mp3_b64(ROOT / rec["wav"], cache)
                        n_syn += 1
                        parts.append(f'<td>{audio_tag(b64)}'
                                     f'<span class="dur">{rec["seconds"]}s</span></td>')
                    else:
                        parts.append('<td class="miss">未合成</td>')

                    stem = Path(turn["ref"]).stem
                    ref_vad = REF_DIR / f"{stem}.wav"
                    ref_raw = REF_RAW_DIR / f"{stem}.mp3"
                    cell = f'<span class="uttid">{html.escape(turn["ref"])}</span>'
                    if ref_vad.exists():
                        cell += (f'<div><span class="tag">VAD</span>'
                                 f'{audio_tag(to_mp3_b64(ref_vad, cache), "ref")}'
                                 f'<span class="dur">{vad_dur.get(stem, "")}</span></div>')
                        n_ref += 1
                    else:
                        cell += '<span class="miss">ref 缺失</span>'
                    if ref_raw.exists():
                        cell += (f'<div><span class="tag raw">原始</span>'
                                 f'{audio_tag(to_mp3_b64(ref_raw, cache), "ref")}</div>')
                    cell += f'<div>{html.escape(turn["ref_text"])}</div>'
                    parts.append(f'<td class="c-ref">{cell}</td>')
                parts.append("</tr>")

            parts.append("</tbody></table></div>")
        parts.append("</section>")

    n_dialog = sum(len(s["dialogs"]) for s in scenes)
    n_turn = sum(len(d["turns"]) for s in scenes for d in s["dialogs"])
    doc = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>语音克隆测试集听测（dots.tts-soar）</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
  <h1>语音克隆测试集听测（dots.tts-soar）</h1>
  <p class="sub">按 <code>voice_clone_testset.md</code> 逐轮克隆合成：{len(scenes)} 场景 / {n_dialog} 组对话 /
     {n_turn} 轮（其中 {n_syn} 轮合成，助手轮不合成）。
     ref 音频取自 <code>error_sessions_model.html</code> 内嵌 base64，按 ref text 唯一匹配导出，
     并经 VAD 切掉首尾静音、把句内静音压到 0.2s 上限后再送进克隆。</p>
  <div class="legend">
    <span><b>图例</b></span>
    <span><span class="badge drv">主驾</span> 人类说话人 1</span>
    <span><span class="badge pas">副驾</span> 人类说话人 2</span>
    <span><span class="badge bot">助手</span> 车机（机器，仅作上下文）</span>
    <span>右侧 <span class="tag">VAD</span> 为实际克隆源，<span class="tag raw">原始</span> 为裁剪前，可与左侧合成音频对比</span>
  </div>
{"".join(parts)}
</div>
</body>
</html>
"""
    out_html.write_text(doc, encoding="utf-8")
    print(f"写出 {out_html}  ({out_html.stat().st_size/1e6:.1f} MB)")
    print(f"内嵌：合成 {n_syn} 条 / ref {n_ref} 条（去重后 {len(cache)} 个音频）")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
