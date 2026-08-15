#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V2 训练数据准备：拒识打标 jsonl → Qwen3-TTS VoiceDesign 微调 JSONL

流程
    读 jsonl → uttid 去重 → gemini/真值一致性筛选 → 定位 wav
      → VAD 切首尾静音 → instruct 转写 → 输出 train/validate jsonl

音频预处理只做 VAD（切首尾静音），不做任何其它过滤。

instruct 转写（只用 gemini 的 reason + evidence，不引入外部信息）
    "{evidence 逐条以"；"连接}。{reason}"
    句尾的「应予拒识/应予响应」会被剥掉——那是打标结论，不是说话方式，
    喂进 instruct 槽位是噪声。

输出格式（Qwen3-TTS finetuning 契约，喂给 prepare_data.py 抽 audio_codes）
    {"audio": "/abs/path.wav", "text": "用户原话", "instruct": "…"}
  注意：**没有 ref_audio**。VoiceDesign 走 instruct 控风格，无参考音频
  （generate_voice_design 的签名里也没有 ref_audio），这与 V2「风格的唯一
  信息源是指令」的设计一致。官方 finetuning/README 那套 ref_audio 是给
  Base 模型做音色克隆的，不是这条路。

用法
    python prepare_v2_data.py \
        --dataset cc_new_0601 /path/CC_new_0601_0630.jsonl /path/turn_audio \
        --dataset cc_new_0701 /path/CC_new_0701_0730.jsonl /path/turn_audio \
        --dataset car_all_0415 /path/car_all_0415_0424.jsonl /path/car_all_0415_0424_wavs \
        --out-dir ./data_v2 --wav-out-dir /data/v2_wav_trimmed --workers 32

    # 100 条 overfit 自检
    python prepare_v2_data.py --dataset ... --out-dir ./data_v2_debug --limit 100
"""

import argparse
import json
import multiprocessing
import os
import random
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import soundfile as sf

# 必须 spawn：fork 的 worker 里首次触发 numba JIT（librosa.effects.trim）会在本机
# aarch64 上崩 —— llvmlite 的 RuntimeDyldELF 报 relocation overflow
# （`Assertion 'isInt<33>(Result)' failed`，进程 SIGABRT，整个池 BrokenProcessPool）。
# 主进程 JIT 正常，spawn 后各 worker 是全新解释器，不受 fork 继承的内存布局影响。
# 代价是每建一次池要重新 import（16 worker × ~2s），全量 22 个分块约多 12 分钟。
_MP_CTX = multiprocessing.get_context("spawn")

# ---------------------------------------------------------------- 常量

# 文本阶段的丢弃原因（不碰音频）
TEXT_DROPS = [
    "json_parse_fail",
    "partial_recovery",
    "dup_uttid",
    "missing_field",
    "missing_true_label",
    "label_mismatch",
    "empty_evidence",
    "empty_reason",
    "empty_text",
]
# 音频阶段的丢弃原因（全部按 accept/reject 分桶统计）
AUDIO_DROPS = [
    "wav_not_found",
    "wav_read_fail",
    "vad_error",
    "wav_write_fail",
    "vad_too_short",
    "worker_crash",
]
DROP_REASONS = TEXT_DROPS + AUDIO_DROPS

# 这些原因发生在"进入 VAD 之前"，算丢弃率时分母应为 text_pass；
# 其余音频原因的分母是真正进入 VAD 阶段的条数
PRE_VAD_DROPS = {"wav_not_found", "wav_read_fail"}

# 一次提交给进程池的任务数，避免 43.5w 个 future 同时驻留内存
CHUNK = 20000

PUNCT = "。；;，,、 \t\r\n"


# ---------------------------------------------------------------- instruct 转写


def norm_evidence(evidence):
    """统一的 evidence 规整：去首尾空白和尾部标点。筛选与转写共用，保证口径一致。"""
    out = []
    for e in evidence:
        if not isinstance(e, str):
            continue
        e = e.strip().strip(PUNCT)
        if e:
            out.append(e)
    return out


# 打标结论的收尾语。它描述的是"该不该拒识"这个判定，不是"该怎么说话"，
# 留在 instruct 里只会稀释风格信号。
#
# 用正则而非后缀元组：真实数据里结论词有 应予拒识 / 应拒识 / 不应拒识 / 应予响应 …
# 多种变体，按元组逐个 endswith 会出现"短的挡住长的"——3 字的「应拒识」先命中
# 「不应拒识」，剥完剩一个反义的「不」字（data_report.md:113 那条真实样本
# "用户在多轮交互中明确回答了助手的提问，不应拒识。" 即触发，且它全落在 accept 侧）。
_VERDICT_RE = re.compile(
    r"[，,、;；\s]*(?:因此|所以|故|则|应当|故而)?\s*"
    r"不?应(?:予|当|该)?(?:拒识|响应|识别|接受)"
    r"[。.！!？?…\s]*$"
)
_TAIL_PUNCT = "。.！!？?…，,、；;： \t"


def strip_verdict(reason):
    r = _VERDICT_RE.sub("", reason.strip()).rstrip(_TAIL_PUNCT)
    return r + "。" if r else ""


def build_instruct(evidence, reason):
    """gemini 的 evidence + reason → VoiceDesign 的 instruct 串。纯机械转写。"""
    head = "；".join(evidence) + "。"
    tail = strip_verdict(reason)      # 整句都是结论词时返回空串
    return head + tail if tail else head


def build_row(text, wav_path, instruct):
    """Qwen3-TTS finetuning 的一行。无 ref_audio —— VoiceDesign 不用参考音频。"""
    return {"audio": wav_path, "text": text, "instruct": instruct}


# ---------------------------------------------------------------- VAD


_SILERO = {}


def _get_silero(model_path):
    """惰性加载 silero-vad（每个 worker 进程各持一份）。"""
    if "m" not in _SILERO:
        import torch

        torch.set_num_threads(1)
        model, utils = torch.hub.load(
            repo_or_dir=model_path, model="silero_vad", source="local", onnx=False
        )
        _SILERO["m"] = model
        _SILERO["get_ts"] = utils[0]  # get_speech_timestamps
        _SILERO["torch"] = torch
    return _SILERO


def trim_silence(wav, sr, mode, top_db, silero_path):
    """
    只切首尾静音，不动句中停顿——句中停顿是腔调的一部分。
    返回 (trimmed_wav, 是否检测到人声)
    """
    if mode == "none":
        return wav, True

    if mode == "silero":
        S = _get_silero(silero_path)
        torch = S["torch"]
        ts = S["get_ts"](torch.from_numpy(wav), S["m"], sampling_rate=sr)
        if not ts:
            return wav[:0], False
        return wav[ts[0]["start"] : ts[-1]["end"]], True

    # energy（默认）：纯 numpy 实现，**不要换回 librosa**。
    # 实测本机 aarch64 上「import soundfile（加载 libsndfile .so）+ numba JIT
    # （librosa.effects.trim → util.frame）」组合必崩：llvmlite RuntimeDyldELF 报
    # relocation overflow（`Assertion 'isInt<33>(Result)' failed`，SIGABRT）——
    # libsndfile 的加载改变了进程地址空间布局，LLVM JIT 的代码段与全局符号距离
    # 超过 4GB。主进程/worker、fork/spawn 都不行，与数据无关。
    # 语义与 librosa.effects.trim 一致（frame 2048 / hop 512）：分帧 RMS → dB →
    # 掐掉首尾低于 (max_db - top_db) 的帧。
    FRAME, HOP = 2048, 512
    if wav.size < FRAME:
        return wav[:0], False
    n = 1 + (wav.size - FRAME) // HOP
    frames = np.lib.stride_tricks.as_strided(
        wav,
        shape=(n, FRAME),
        strides=(wav.strides[0] * HOP, wav.strides[0]),
        writeable=False,
    )
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    ref_db = 20.0 * np.log10(max(float(rms.max()), 1e-10)) - top_db
    idx = np.nonzero(20.0 * np.log10(np.maximum(rms, 1e-10)) > ref_db)[0]
    if idx.size == 0:
        return wav[:0], False
    start = int(idx[0]) * HOP
    end = min(wav.size, (int(idx[-1]) + 1) * HOP + FRAME)
    return wav[start:end], True


# ---------------------------------------------------------------- 单条处理（子进程）


def process_one(job):
    """
    统一返回 dict，便于父进程按 label 分桶统计：
        {status, uttid, label, rec?}
    status == "ok" 时 rec 携带 _wav / _dur_before / _dur_after
    """
    rec = job["rec"]
    cfg = job["cfg"]
    uttid = rec["uttid"]
    label = rec["label"]

    def fail(status):
        return {"status": status, "uttid": uttid, "label": label}

    src = os.path.join(cfg["wav_dir"], f"{uttid}.wav")
    if not os.path.isfile(src):
        return fail("wav_not_found")

    dst = os.path.join(cfg["wav_out_dir"], job["ds_name"], f"{uttid}.wav")

    dur_before = None
    dur_after = None

    # 断点续跑：已有产物则复用（VAD 参数指纹已在父进程校验过一致）
    if cfg["skip_existing"] and os.path.isfile(dst):
        try:
            info = sf.info(dst)
            cand = info.frames / info.samplerate
            if cand >= cfg["min_dur"]:
                dur_after = cand
                try:
                    si = sf.info(src)
                    dur_before = si.frames / si.samplerate
                except Exception:
                    dur_before = cand
        except Exception:
            dur_after = None

    if dur_after is None:
        try:
            wav, sr = sf.read(src, dtype="float32", always_2d=False)
        except Exception:
            return fail("wav_read_fail")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        dur_before = len(wav) / sr

        try:
            wav, has_speech = trim_silence(
                wav, sr, cfg["vad"], cfg["top_db"], cfg["silero_path"]
            )
        except Exception:
            return fail("vad_error")

        dur_after = len(wav) / sr
        if not has_speech or dur_after < cfg["min_dur"]:
            return fail("vad_too_short")

        os.makedirs(os.path.dirname(dst), exist_ok=True)
        # 先写临时文件再原子替换：进程被 kill 时不会在 dst 留下截断的 wav，
        # 否则 --skip-existing 下一轮会把残缺音频当成有效产物复用
        # 临时名不带 .wav 后缀，soundfile 无法从扩展名推断格式，必须显式给 format
        tmp = f"{dst}.tmp{os.getpid()}"
        try:
            sf.write(tmp, wav, sr, subtype="PCM_16", format="WAV")
            os.replace(tmp, dst)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            return fail("wav_write_fail")

    rec["_wav"] = dst
    rec["_dur_before"] = dur_before
    rec["_dur_after"] = dur_after
    return {"status": "ok", "uttid": uttid, "label": label, "rec": rec}


# ---------------------------------------------------------------- jsonl 读取与筛选


def iter_jsonl(path, stats, seen_uttids, keep_partial):
    """流式读 jsonl，做纯文本层面的筛选（不碰音频）。"""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            stats["total"] += 1
            try:
                rec = json.loads(line)
            except Exception:
                stats["json_parse_fail"] += 1
                continue

            if not keep_partial and rec.get("_partial_json_recovery"):
                stats["partial_recovery"] += 1
                continue

            uttid = rec.get("uttid")
            text = rec.get("text")
            if not uttid or text is None:
                stats["missing_field"] += 1
                continue

            if uttid in seen_uttids:
                stats["dup_uttid"] += 1
                continue

            text = str(text).strip()
            if not text:
                stats["empty_text"] += 1
                continue

            # gemini 判定：rej_label 存在但为 null 时也要回退到 decision
            pred = rec.get("rej_label")
            if pred is None:
                pred = rec.get("decision")
            if pred is None:
                stats["missing_field"] += 1
                continue
            try:
                pred = int(pred)
            except Exception:
                stats["missing_field"] += 1
                continue

            # 真值
            truth = rec.get("true_rej_label")
            if truth is None:
                tc = rec.get("true_class")
                if tc in ("accept", "reject"):
                    truth = 1 if tc == "reject" else 0
            if truth is None:
                stats["missing_true_label"] += 1
                continue
            try:
                truth = int(truth)
            except Exception:
                stats["missing_true_label"] += 1
                continue

            # 值域校验：非 0/1 会让统计口径失真（entered_vad 只遍历 0/1），
            # 且报告的 accept+reject 不等于 kept，完全看不出来
            if pred not in (0, 1) or truth not in (0, 1):
                stats["missing_field"] += 1
                continue

            # 到达一致性比较的全部记录，按 gemini 预测标签计数。
            # 用它当"筛选前"的口径，避免与 text_pass（在 evidence/reason 过滤之后）人群不一致
            stats["pred_seen_by_label"][pred] += 1

            # 一致性筛选：gemini 与真值不一致的直接丢
            if pred != truth:
                stats["label_mismatch"] += 1
                stats["mismatch_by_label"][pred] += 1
                continue

            evidence = rec.get("evidence") or []
            if isinstance(evidence, str):
                evidence = [evidence]
            evidence = norm_evidence(evidence)
            if not evidence:
                stats["empty_evidence"] += 1
                continue

            reason = rec.get("reason")
            if not isinstance(reason, str) or not reason.strip():
                stats["empty_reason"] += 1
                continue

            seen_uttids.add(uttid)
            yield {
                "uttid": uttid,
                "text": text,
                "label": pred,
                "evidence": evidence,
                "reason": reason,
            }


# ---------------------------------------------------------------- VAD 参数指纹


def vad_fingerprint(args):
    return {
        "vad": args.vad,
        "top_db": args.top_db,
        "min_dur": args.min_dur,
        "silero_path": args.silero_path,
    }


def _fp_path(wav_out_dir, ds_name):
    d = os.path.join(wav_out_dir, ds_name)
    os.makedirs(d, exist_ok=True)
    return d, os.path.join(d, ".vad_params.json")


def _atomic_json_dump(path, obj):
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def check_fingerprint(wav_out_dir, ds_name, fp, skip_existing, clean_stale=False):
    """
    决定本数据集能否复用已有产物。

    只有上一次是「同参数」且「完整跑完」才允许 skip，覆盖这个静默污染场景：
      运行A(top_db=30) 跑一半被 kill → 运行B(top_db=20) 重算又跑一半崩了
      → 运行C(top_db=20) 若只比参数就会复用运行A 留下的 top_db=30 产物。

    **不做删除**：allow=False 时 process_one 会对每条候选重算并覆盖，旧参数产物
    永远不会被使用，删除纯属磁盘卫生。而无条件删除的代价极大——一次 `--limit 100`
    的调试跑就会抹掉全量 43.5w 条、50-60GB 的产物，且旧 jsonl 仍指向这些路径，
    下游 prepare_data.py 读不到 wav 会直接抛异常，整个 job 挂掉。
    需要清理时用 --clean-stale 显式开启。
    """
    d, path = _fp_path(wav_out_dir, ds_name)
    old = None
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                old = json.load(f)
        except Exception:
            old = None  # 指纹损坏：保守处理，下面会禁用 skip

    allow = skip_existing
    if old is None:
        if skip_existing and os.path.isdir(d) and any(
            fn.endswith(".wav") for fn in os.listdir(d)
        ):
            print(
                f"  ⚠ [{ds_name}] 目录有产物但无有效指纹，无法确认生成参数，本次全部重算",
                flush=True,
            )
            allow = False
    elif old.get("fp") != fp:
        if skip_existing:
            print(
                f"  ⚠ [{ds_name}] VAD 参数与上次不一致，已禁用 --skip-existing，全部重算覆盖\n"
                f"      上次: {old.get('fp')}\n      本次: {fp}",
                flush=True,
            )
        allow = False
    elif not old.get("complete"):
        if skip_existing:
            print(
                f"  ⚠ [{ds_name}] 上次运行未跑完，残留产物不可信，本次全部重算",
                flush=True,
            )
        allow = False

    if clean_stale:
        n = 0
        for fn in os.listdir(d):
            if fn.endswith(".wav") or ".wav.tmp" in fn:
                try:
                    os.unlink(os.path.join(d, fn))
                    n += 1
                except OSError:
                    pass
        print(f"  --clean-stale: [{ds_name}] 已删除 {n} 个文件", flush=True)
    else:
        # 无论如何都清掉上次被 SIGKILL 留下的半截临时文件（它们不带 .wav 结尾，
        # 不会被 skip 逻辑误用，但会持续堆积）
        for fn in os.listdir(d):
            if ".wav.tmp" in fn:
                try:
                    os.unlink(os.path.join(d, fn))
                except OSError:
                    pass

    _atomic_json_dump(path, {"fp": fp, "complete": False})
    return allow


def mark_complete(wav_out_dir, ds_name, fp, trustworthy=True):
    """
    该数据集处理完毕后落 complete 标记。
    trustworthy=False（--limit 截断 或 有 worker 崩溃）时不打标记，
    否则下次 --skip-existing 会把一个残缺的产物集当成完整的复用。
    """
    if not trustworthy:
        return
    _, path = _fp_path(wav_out_dir, ds_name)
    _atomic_json_dump(path, {"fp": fp, "complete": True})


# ---------------------------------------------------------------- 统计


def new_stats():
    s = {k: 0 for k in DROP_REASONS}
    s["total"] = 0
    s["kept"] = 0
    s["mismatch_by_label"] = Counter()  # 按 gemini 预测值 pred 计数
    s["pred_seen_by_label"] = Counter()  # 到达一致性比较的全部记录，按 pred 计数
    s["kept_by_label"] = Counter()
    s["text_pass_by_label"] = Counter()
    s["drop_by_label"] = defaultdict(Counter)  # reason -> Counter{label: n}
    s["dur_before"] = []
    s["dur_after"] = []
    return s


def entered_vad(s):
    """真正进入 VAD 阶段的条数 = 文本通过 - 找不到/读不了 wav。按 label 分。"""
    e = Counter()
    for lb in (0, 1):
        e[lb] = s["text_pass_by_label"][lb] - sum(
            s["drop_by_label"][k][lb] for k in PRE_VAD_DROPS
        )
    return e


def drop_base(s, reason):
    """
    某丢弃原因的正确分母：
      wav_not_found / wav_read_fail 发生在进入 VAD 之前 → 分母是 text_pass
      其余（vad_*, worker_crash）   → 分母是实际进入 VAD 的条数
    用错分母会出现 wav_not_found/(text_pass-wav_not_found) 这种自我扣除，缺失率越高偏差越大。
    """
    if reason in PRE_VAD_DROPS:
        return s["text_pass_by_label"]
    return entered_vad(s)


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(
        description="拒识打标 jsonl → Qwen3-TTS VoiceDesign 微调 JSONL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--dataset",
        nargs=3,
        action="append",
        metavar=("NAME", "JSONL", "WAV_DIR"),
        required=True,
        help="可重复。WAV_DIR 下按 <uttid>.wav 扁平查找",
    )
    ap.add_argument("--out-dir", required=True, help="train_raw.jsonl / valid_raw.jsonl 输出目录")
    ap.add_argument(
        "--wav-out-dir",
        default=None,
        help="VAD 后 wav 的输出目录（默认 <out-dir>/wav）。全量约需 50-60GB",
    )
    ap.add_argument("--vad", choices=["energy", "silero", "none"], default="energy")
    ap.add_argument(
        "--top-db",
        type=float,
        default=30.0,
        help="energy 模式阈值。越小越激进；车内录音背景噪声大，不建议低于 25",
    )
    ap.add_argument("--silero-path", default=None, help="silero-vad 本地仓库路径")
    ap.add_argument("--min-dur", type=float, default=0.3, help="VAD 后短于此秒数则丢弃")
    ap.add_argument(
        "--val-size", type=int, default=500, help="验证集条数（至少 1，训练配了 eval_strategy）"
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="**每个**数据集最多取多少条（调试用）。三个数据集传 100 会得到约 300 条",
    )
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--skip-existing", action="store_true", help="复用已有的 trimmed wav，可断点续跑"
    )
    ap.add_argument("--keep-partial-recovery", action="store_true")
    ap.add_argument(
        "--clean-stale",
        action="store_true",
        help="删除输出目录下已有的全部 wav 再跑。破坏性操作，仅用于磁盘清理",
    )
    args = ap.parse_args()

    if args.vad == "silero" and not args.silero_path:
        ap.error("--vad silero 需要同时给 --silero-path")

    names = [d[0] for d in args.dataset]
    if len(names) != len(set(names)):
        ap.error(f"--dataset 名称重复会共用输出目录并覆盖统计: {names}")
    if args.val_size < 1:
        ap.error("--val-size 至少为 1：训练脚本配了 eval_strategy=steps，空验证集会在首次 eval 时崩")

    random.seed(args.seed)
    # 下游 prepare_data.py 的 cwd 与本脚本不同，写进 jsonl 的必须是绝对路径
    args.out_dir = os.path.abspath(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    wav_out_dir = os.path.abspath(args.wav_out_dir or os.path.join(args.out_dir, "wav"))
    os.makedirs(wav_out_dir, exist_ok=True)

    fp = vad_fingerprint(args)
    seen_uttids = set()  # 跨数据集全局去重
    per_ds_stats = {}
    kept_recs = []  # (rec, ds_name)

    for ds_name, jsonl_path, wav_dir in args.dataset:
        print(f"\n{'=' * 60}\n[{ds_name}] {jsonl_path}", flush=True)
        stats = new_stats()
        per_ds_stats[ds_name] = stats

        # 输出目录与源目录相同会原地覆盖源音频，且 --clean-stale 会把源文件删光
        ds_out = os.path.join(wav_out_dir, ds_name)
        if os.path.abspath(wav_dir) == os.path.abspath(ds_out):
            print(
                f"\n❌ [{ds_name}] 输入 wav 目录与输出目录相同（{ds_out}）——"
                f"会原地覆盖源音频。请换一个 --wav-out-dir",
                file=sys.stderr,
            )
            sys.exit(1)

        allow_skip = check_fingerprint(
            wav_out_dir, ds_name, fp, args.skip_existing, args.clean_stale
        )
        crash_before = stats["worker_crash"]
        cfg = {
            "wav_dir": os.path.abspath(wav_dir),
            "wav_out_dir": wav_out_dir,
            "vad": args.vad,
            "top_db": args.top_db,
            "silero_path": args.silero_path,
            "min_dur": args.min_dur,
            "skip_existing": allow_skip,
        }

        # 阶段一：纯文本筛选
        candidates = []
        for rec in iter_jsonl(jsonl_path, stats, seen_uttids, args.keep_partial_recovery):
            stats["text_pass_by_label"][rec["label"]] += 1
            candidates.append(rec)
            if args.limit and len(candidates) >= args.limit:
                break
        print(f"  文本层筛选后: {len(candidates)} 条", flush=True)
        if not candidates:
            continue

        # 阶段二：音频处理。
        # 进程池建在分块循环**内部**：ProcessPoolExecutor 一旦有 worker 异常终止就会进入
        # 永久 broken 状态，后续所有 submit/map 立即抛 BrokenProcessPool。若池建在循环外，
        # 第 3 万条崩掉会导致剩下 40 万条一条都处理不到，而脚本仍"正常"输出 jsonl。
        done = 0
        for start in range(0, len(candidates), CHUNK):
            chunk = candidates[start : start + CHUNK]
            chunk_done = 0  # 块内独立计数，不能用全局 done 反推
            try:
                with ProcessPoolExecutor(max_workers=args.workers, mp_context=_MP_CTX) as ex:
                    jobs = ({"rec": c, "cfg": cfg, "ds_name": ds_name} for c in chunk)
                    for r in ex.map(process_one, jobs, chunksize=64):
                        chunk_done += 1
                        done += 1
                        st = r["status"]
                        if st != "ok":
                            stats[st] += 1
                            stats["drop_by_label"][st][r["label"]] += 1
                            continue
                        rec = r["rec"]
                        kept_recs.append((rec, ds_name))
                        stats["kept"] += 1
                        stats["kept_by_label"][rec["label"]] += 1
                        stats["dur_before"].append(rec["_dur_before"])
                        stats["dur_after"].append(rec["_dur_after"])
            except Exception as e:
                tail = chunk[chunk_done:]
                stats["worker_crash"] += len(tail)
                for c in tail:
                    stats["drop_by_label"]["worker_crash"][c["label"]] += 1
                print(
                    f"  ⚠ 分块 [{start}:{start + len(chunk)}] 处理中断"
                    f"({type(e).__name__}: {e})，本块跳过 {len(tail)} 条，继续下一块",
                    flush=True,
                )
            print(f"  音频处理 {done}/{len(candidates)}", flush=True)

        mark_complete(
            wav_out_dir,
            ds_name,
            fp,
            trustworthy=(args.limit is None and stats["worker_crash"] == crash_before),
        )
        candidates.clear()

    # ---------------- 输出 ----------------
    if not kept_recs:
        print("\n❌ 没有任何样本通过筛选，检查路径和字段名", file=sys.stderr)
        sys.exit(1)

    random.shuffle(kept_recs)

    all_samples, manifest = [], []
    for rec, ds_name in kept_recs:
        instruct = build_instruct(rec["evidence"], rec["reason"])
        all_samples.append(build_row(rec["text"], rec["_wav"], instruct))
        manifest.append(
            {
                "uttid": rec["uttid"],
                "dataset": ds_name,
                "label": rec["label"],
                "text": rec["text"],
                "wav": rec["_wav"],
                "dur_before": round(rec["_dur_before"], 3),
                "dur_after": round(rec["_dur_after"], 3),
                "instruct": instruct,
            }
        )

    val_n = min(args.val_size, max(1, len(all_samples) // 10))
    val, train = all_samples[:val_n], all_samples[val_n:]

    train_path = os.path.join(args.out_dir, "train_raw.jsonl")
    val_path = os.path.join(args.out_dir, "valid_raw.jsonl")
    for path, rows in ((train_path, train), (val_path, val)):
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out_dir, "manifest.jsonl"), "w", encoding="utf-8") as f:
        for m in manifest:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    # ---------------- 统计报告 ----------------
    cfg_dump = dict(vars(args))
    cfg_dump["wav_out_dir"] = wav_out_dir  # 用默认值时 args 里是 None，记录实际路径
    report = {"config": cfg_dump, "datasets": {}, "overall": {}}
    tot_kept = Counter()
    lines = ["", "=" * 72, "数据准备统计", "=" * 72]

    for ds, s in per_ds_stats.items():
        db, da = s["dur_before"], s["dur_after"]
        entered = entered_vad(s)

        d = {
            "total_lines": s["total"],
            "kept": s["kept"],
            "kept_accept": s["kept_by_label"][0],
            "kept_reject": s["kept_by_label"][1],
            "drops": {k: s[k] for k in DROP_REASONS if s[k]},
            "drops_by_label": {
                k: dict(v) for k, v in s["drop_by_label"].items() if sum(v.values())
            },
            # 一致性筛选是全流程最大的单一丢弃源且高度不对称（gemini FP 远多于 FN），
            # 会系统性压低 reject 占比。不做平衡是产品决策，但必须显式暴露。
            "label_mismatch_by_pred": dict(s["mismatch_by_label"]),
            "entered_audio_accept": entered[0],
            "entered_audio_reject": entered[1],
        }
        if db:
            d["dur_before_mean"] = round(float(np.mean(db)), 2)
            d["dur_after_mean"] = round(float(np.mean(da)), 2)
            tot_b = float(np.sum(db))
            d["trimmed_ratio"] = (
                round(1 - float(np.sum(da)) / tot_b, 4) if tot_b > 0 else None
            )
        report["datasets"][ds] = d
        tot_kept += s["kept_by_label"]

        lines.append(f"\n[{ds}]")
        lines.append(f"  原始行数        {s['total']:>9,}")
        lines.append(
            f"  保留            {s['kept']:>9,}   (accept {d['kept_accept']:,} / reject {d['kept_reject']:,})"
        )
        for k in DROP_REASONS:
            if not s[k]:
                continue
            if k in AUDIO_DROPS:
                by = s["drop_by_label"][k]
                lines.append(
                    f"    - {k:<22} {s[k]:>7,}   (accept {by[0]:,} / reject {by[1]:,})"
                )
            elif k == "label_mismatch":
                mb = s["mismatch_by_label"]
                lines.append(
                    f"    - {k:<22} {s[k]:>7,}   "
                    f"(gemini判reject实为accept {mb[1]:,} / gemini判accept实为reject {mb[0]:,})"
                )
            else:
                lines.append(f"    - {k:<22} {s[k]:>7,}")
        # 一致性筛选前后的 reject 占比变化——不做平衡，但要看得见。
        # 两侧都用同一人群（到达一致性比较的记录）、同一口径（gemini 预测标签）
        ps = s["pred_seen_by_label"]
        mb = s["mismatch_by_label"]
        raw_tot = ps[0] + ps[1]
        post_r, post_a = ps[1] - mb[1], ps[0] - mb[0]
        if raw_tot > 0 and post_r + post_a > 0:
            lines.append(
                f"  一致性筛选前后 reject 占比（按 gemini 预测标签）  "
                f"{ps[1] / raw_tot * 100:.1f}% → {post_r / (post_r + post_a) * 100:.1f}%"
            )
        if db and d.get("trimmed_ratio") is not None:
            lines.append(
                f"  时长 VAD 前/后  {d['dur_before_mean']}s → {d['dur_after_mean']}s"
                f"  (切掉 {d['trimmed_ratio'] * 100:.1f}%)"
            )
        # VAD 丢弃率，分母是真正进入 VAD 阶段的条数
        by = s["drop_by_label"]["vad_too_short"]
        if sum(by.values()):
            for lb, name in ((0, "accept"), (1, "reject")):
                if entered[lb] > 0:
                    lines.append(
                        f"  VAD 丢弃率 {name:<7} {by[lb] / entered[lb] * 100:>5.2f}%"
                        f"  ({by[lb]:,}/{entered[lb]:,})"
                    )

    report["overall"] = {
        "train": len(train),
        "validate": len(val),
        "accept": tot_kept[0],
        "reject": tot_kept[1],
    }
    total_kept = max(1, sum(tot_kept.values()))
    lines.append(f"\n{'-' * 72}")
    lines.append(f"合计  train {len(train):,}  validate {len(val):,}")
    lines.append(
        f"      accept {tot_kept[0]:,} ({tot_kept[0] / total_kept * 100:.1f}%)  "
        f"reject {tot_kept[1]:,} ({tot_kept[1] / total_kept * 100:.1f}%)"
    )

    # 类别偏斜告警：音频阶段任一丢弃原因，两类丢弃率差 2 倍以上就报
    warn = []
    for ds, s in per_ds_stats.items():
        for k in AUDIO_DROPS:
            by = s["drop_by_label"][k]
            if not sum(by.values()):
                continue
            base = drop_base(s, k)  # 每个原因用各自正确的分母
            r0 = by[0] / max(1, base[0])
            r1 = by[1] / max(1, base[1])
            # 文件缺失/读不出跟 VAD 阈值无关，别把人往错方向引
            hint = (
                "检查 wav 目录挂载与文件完整性"
                if k in PRE_VAD_DROPS
                else "正在系统性砍掉某一类样本，检查 --top-db"
            )
            if r1 > r0 * 2 and r1 > 0.02:
                warn.append(
                    f"  ⚠ [{ds}] {k}: reject 丢弃率 {r1 * 100:.2f}% 明显高于 "
                    f"accept({r0 * 100:.2f}%)——{hint}"
                )
            elif r0 > r1 * 2 and r0 > 0.02:
                warn.append(
                    f"  ⚠ [{ds}] {k}: accept 丢弃率 {r0 * 100:.2f}% 明显高于 "
                    f"reject({r1 * 100:.2f}%)——{hint}"
                )
    if warn:
        lines.append("")
        lines.extend(warn)

    lines.append(f"\n输出: {train_path}")
    lines.append(f"      {val_path}")
    lines.append("=" * 72)

    # worker 崩溃会静默少掉最多一个 CHUNK（2w 条）的数据，必须体现在退出码上，
    # 否则调度脚本看到 exit 0 就继续往下跑训练了
    total_crash = sum(s["worker_crash"] for s in per_ds_stats.values())
    crash_ratio = total_crash / max(1, total_crash + len(all_samples))
    report["overall"]["worker_crash"] = total_crash
    if total_crash:
        lines.append("")
        lines.append(
            f"  ⚠⚠ 有 {total_crash:,} 条因 worker 崩溃未被处理"
            f"（占 {crash_ratio * 100:.2f}%）——本次产出的数据集是**不完整**的。"
        )
        lines.append("     排查 OOM / 损坏音频后，用 --skip-existing 续跑可补齐。")

    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(args.out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out_dir, "stats.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")

    if crash_ratio > 0.01:
        print(
            f"\n❌ worker 崩溃比例 {crash_ratio * 100:.2f}% 超过 1%，以非零码退出",
            file=sys.stderr,
        )
        sys.exit(2)


if __name__ == "__main__":
    main()
