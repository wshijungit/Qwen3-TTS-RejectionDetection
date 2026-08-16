#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V2 VoiceDesign 微调链路在昇腾 NPU 上的分阶段冒烟测试。

设计原则：**每阶段独立可诊断**。任一阶段失败时，输出直接指出是哪一层的问题、
下一步该怎么办，而不是丢一个 traceback 让人猜。

阶段：
  0 环境        torch_npu / transformers / accelerate / qwen_tts
  1 基础算子    bf16 matmul、显存查询
  2 注意力后端  sdpa vs eager 能否跑、相对耗时（sdpa 若静默退化会体现在耗时上）
  3 模型加载    VoiceDesign 权重能否加载到 NPU
  4 音频 tokenizer  抽 audio_codes（prepare_data.py 那一步的核心）
  5 序列布局    collate 产出与推理侧逐格核对（纯 CPU，不依赖 NPU）
  6 前反向      若干步训练，看 loss 是否真的在降
  7 保存加载    ckpt 完整性 + 能否被 from_pretrained 读回
  8 推理        用微调后的 ckpt 合成，验证 instruct 通路仍然生效

用法：
    python npu_smoke_test.py --model_path /path/to/VoiceDesign
    python npu_smoke_test.py --model_path ... --stages 0-6      # 只跑前 7 个
    python npu_smoke_test.py --model_path ... --real_jsonl x.jsonl  # 用真实数据

产出 npu_smoke_report.txt —— 直接把它贴回来即可定位问题。
"""
import argparse
import json
import os
import shlex
import shutil
import sys
import tempfile
import time
import traceback

# torch_npu 必须在模块级 import：阶段 1/2/3/4/8 都用 torch.npu / "npu:0"，
# 若只在阶段 0 里 import，单跑 --stages 2 或 4-6 时会抛 AttributeError，
# 而各阶段的兜底诊断会把它误报成"sdpa 不可用"或"tokenizer 算子不支持"。
try:
    import torch_npu  # noqa: F401
    _NPU_IMPORT_ERR = None
except Exception as _e:  # noqa: BLE001
    _NPU_IMPORT_ERR = _e

REPORT = []


def log(msg=""):
    print(msg, flush=True)
    REPORT.append(str(msg))


def stage(n, title):
    log(f"\n{'=' * 66}\n[阶段 {n}] {title}\n{'=' * 66}")


def ok(msg):
    log(f"  ✅ {msg}")


def warn(msg):
    log(f"  ⚠️  {msg}")


def fail(msg, howto=""):
    log(f"  ❌ {msg}")
    if howto:
        for line in howto.strip().split("\n"):
            log(f"     → {line}")
    return False


# ---------------------------------------------------------------- 0 环境


def s0_env():
    stage(0, "环境")
    import torch

    if _NPU_IMPORT_ERR is not None:
        e = _NPU_IMPORT_ERR
        return fail(
            f"torch_npu 不可用: {type(e).__name__}: {e}",
            "先 source CANN 环境（scripts/npu_env.sh 会做）。\n"
            "若报找不到 .so，是 CANN 与 torch_npu 版本不配套。",
        )
    log(f"  torch {torch.__version__} / torch_npu {torch_npu.__version__}")
    n = torch.npu.device_count()
    if n == 0:
        return fail("看不到 NPU 设备", "npu-smi info 确认卡在位且未被占用")
    ok(f"{n} 张卡，device0 = {torch.npu.get_device_name(0)}")

    import transformers

    v = tuple(int(x) for x in transformers.__version__.split(".")[:2])
    if v < (4, 41):
        return fail(
            f"transformers {transformers.__version__} 过低",
            "需 >= 4.41（4.40 没有 eval_strategy，且 fsdp_config 的 bool 会炸）。\n"
            "pyproject 要求 4.57.3，其 requires-python >=3.9.0，py3.9 可装。",
        )
    ok(f"transformers {transformers.__version__}")

    import accelerate

    ok(f"accelerate {accelerate.__version__}")
    if tuple(int(x) for x in accelerate.__version__.split(".")[:2]) >= (1, 12):
        warn("accelerate >= 1.12 要求 py>=3.10；若本机是 py3.9 而它能 import，说明装的是别的版本")
    from accelerate import Accelerator

    missing = [a for a in ("accumulate", "backward", "clip_grad_norm_", "prepare",
                           "unwrap_model", "is_main_process") if not hasattr(Accelerator, a)]
    if missing:
        return fail(f"accelerate 缺少接口 {missing}", "版本过低，升到 1.0.0 以上")
    ok("Accelerator 具备本链路用到的全部接口")

    import qwen_tts  # noqa: F401

    ok(f"qwen_tts 可 import（python {sys.version.split()[0]}）")
    return True


# ---------------------------------------------------------------- 1 基础算子


def s1_ops():
    stage(1, "基础算子")
    import torch

    dev = "npu:0"
    try:
        a = torch.randn(2048, 2048, dtype=torch.bfloat16, device=dev)
        t0 = time.time()
        for _ in range(20):
            a = a @ a.T / 100
        torch.npu.synchronize()
        ok(f"bf16 matmul 2048² ×20 耗时 {time.time() - t0:.3f}s")
    except Exception as e:
        return fail(f"bf16 matmul 失败: {type(e).__name__}: {e}", "CANN 与 torch_npu 不配套")

    try:
        free, total = torch.npu.mem_get_info(0)
        ok(f"显存 {free / 2**30:.1f} / {total / 2**30:.1f} GiB 可用")
        if total / 2**30 < 32:
            warn("单卡显存 < 32GB：--dtype fp32 需约 25GiB + 激活，可能偏紧；"
                 "可改 --dtype bf16（约 12.7GiB），但 Adam 动量也会是 bf16，注意收敛")
    except Exception:
        warn("mem_get_info 不可用，跳过显存检查")
    return True


# ---------------------------------------------------------------- 2 注意力后端


def s2_attn(model_path):
    stage(2, "注意力后端（本链路在 NPU 上最大的性能未知项）")
    import torch
    from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig

    cfg = Qwen3TTSConfig.from_pretrained(model_path)
    h = cfg.talker_config.hidden_size
    nh = cfg.talker_config.num_attention_heads
    results = {}
    for impl in ("sdpa", "eager"):
        try:
            q = torch.randn(2, nh, 512, h // nh, dtype=torch.bfloat16, device="npu:0")
            k, v = q.clone(), q.clone()
            causal = torch.triu(torch.ones(512, 512, device=q.device, dtype=torch.bool), 1)
            # warm-up：首次 aclnn 编译开销可达数百毫秒，eager 有 5 个算子要编、
            # sdpa 只有 1 个，不预热会系统性地把 eager 拉慢
            for _ in range(3):
                if impl == "sdpa":
                    torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
                else:
                    w = (q @ k.transpose(-1, -2)) / (q.shape[-1] ** 0.5)
                    torch.softmax(w.masked_fill(causal, float("-inf")), -1) @ v
            torch.npu.synchronize()
            t0 = time.time()
            for _ in range(10):
                if impl == "sdpa":
                    torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
                else:
                    w = (q @ k.transpose(-1, -2)) / (q.shape[-1] ** 0.5)
                    torch.softmax(w.masked_fill(causal, float("-inf")), -1) @ v
            torch.npu.synchronize()
            results[impl] = time.time() - t0
            ok(f"{impl:6} 可用，10 次耗时 {results[impl]:.3f}s")
        except Exception as e:
            warn(f"{impl:6} 失败: {type(e).__name__}: {e}")

    if "sdpa" not in results:
        return fail("sdpa 不可用", "启动脚本传 ATTN=eager")
    if "eager" not in results:
        warn("eager 未跑成，无对照数据，无法判断 sdpa 是否真的走了融合 kernel")
        return True
    if results["sdpa"] > results["eager"] * 0.9:
        warn("sdpa 并不比 eager 快 —— 可能静默退化到 math 分支。\n"
             "     若训练显存/速度不可接受，考虑改用 torch_npu.npu_fusion_attention（需改模型代码）")
    else:
        ok("sdpa 有加速效果，按默认用 sdpa")
    return True


# ---------------------------------------------------------------- 3 模型加载


def s3_load(model_path, attn):
    stage(3, "模型加载")
    import torch
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

    t0 = time.time()
    try:
        # 不用 device_map：torch 2.1 + torch_npu 上 meta 快速加载路径会炸
        # （param[...] → torch.cuda._lazy_init → "Torch not compiled with CUDA enabled"），
        # 与训练脚本同路线：先 CPU 加载再整体 .to(npu)
        m = Qwen3TTSModel.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                                          attn_implementation=attn)
        m.model.to("npu:0")
        m.device = torch.device("npu:0")
    except Exception as e:
        return fail(f"加载失败: {type(e).__name__}: {e}",
                    "若是 flash_attention_2 相关，改 --attn sdpa；\n"
                    "若是 model_type 不认识，确认 qwen_tts 已 pip install -e .")
    n = sum(p.numel() for p in m.model.parameters())
    ok(f"加载成功 {time.time() - t0:.1f}s，参数 {n / 1e9:.2f}B，attn={attn}")
    mt = getattr(m.model.config, "tts_model_type", "?")
    if mt != "voice_design":
        return fail(f"tts_model_type = {mt}，不是 voice_design",
                    "V2 必须用 VoiceDesign 权重；Base/CustomVoice 走的是另一套条件")
    ok("tts_model_type = voice_design")
    return m


# ---------------------------------------------------------------- 4 音频 tokenizer


def s4_codes(model_path, workdir, real_jsonl=None, n_synth=8, device="npu:0"):
    stage(4, "音频 tokenizer（抽 audio_codes）")
    import numpy as np
    import soundfile as sf
    import torch
    from qwen_tts.inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer

    raw = os.path.join(workdir, "train_raw.jsonl")
    if real_jsonl:
        rows = [json.loads(l) for l in open(real_jsonl) if l.strip()][:8]
        with open(raw, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        ok(f"用真实数据 {len(rows)} 条")
    else:
        wd = os.path.join(workdir, "wav")
        os.makedirs(wd, exist_ok=True)
        rng = np.random.default_rng(0)
        rows = []
        cases = [
            ("打开前排车窗", "用户指令包含明确的车控意图；音频中用户发音清晰且为近场发令。"),
            ("他妈，但他不是临汾", "用户文本包含语气词和闲聊内容；音频听起来是向他人解释或吐槽。"),
            ("导航到公司", "用户发出明确的导航请求；指令简短清晰，为对车机发令。"),
            ("咱这是往哪走", "用户使用咱字，属于与乘客聊天商量；非对车机助手的指令。"),
        ]
        for i in range(n_synth):
            text, ins = cases[i % len(cases)]
            p = os.path.join(wd, f"s{i}.wav")
            t = np.arange(int(2.5 * 16000)) / 16000
            f0 = rng.uniform(100, 200)
            sig = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in (1, 2, 3))
            sig = (0.25 * sig * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))).astype("float32")
            sf.write(p, sig, 16000, subtype="PCM_16")
            rows.append({"audio": p, "text": text, "instruct": ins})
        with open(raw, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        ok(f"造了 {len(rows)} 条合成样本")

    try:
        # 同 s3：device_map 在 torch 2.1 + torch_npu 上走 meta 加载路径会炸
        tok = Qwen3TTSTokenizer.from_pretrained(
            os.path.join(model_path, "speech_tokenizer"))   # 不传 dtype，与生产路径一致
        tok.model.to(device)
        tok.device = torch.device(device)
    except Exception as e:
        return fail(f"tokenizer 加载失败: {type(e).__name__}: {e}",
                    "确认 speech_tokenizer 子目录存在")

    out = os.path.join(workdir, "train_codes.jsonl")
    done = []
    for r in rows:
        try:
            c = tok.encode(r["audio"])
            # encode 返回 Qwen3TTSTokenizerV2EncoderOutput（ModelOutput），
            # audio_codes 是 List[Tensor]，取 batch 第 0 条转 list
            r["audio_codes"] = c.audio_codes[0].cpu().tolist()
            done.append(r)
        except Exception as e:
            return fail(f"encode 失败: {type(e).__name__}: {e}",
                        "音频 tokenizer 在 NPU 上有算子不支持；\n"
                        "临时方案：这一步放 CPU 跑（--device cpu），只训练用 NPU")
    with open(out, "w") as f:
        for r in done:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    shape = (len(done[0]["audio_codes"]), len(done[0]["audio_codes"][0]))
    ok(f"抽码成功，{len(done)} 条，shape={shape}（应为 (T, 16)）")
    if shape[1] != 16:
        return fail(f"码本数 {shape[1]} != 16", "tokenizer 版本不对")
    del tok
    return out


# ------------------------------------------------------------ 4b 真实数据准备链路


def s4b_pipeline(model_path, workdir, device, n=12):
    """走**生产路径** prepare_v2_data.py → prepare_data.py。

    阶段 4 是自己造 wav 直接调 tokenizer，绕开了整个数据准备脚本。而那两个脚本
    （VAD、一致性筛选、疑似清单、断点续跑、编码）改动最多却从没在昇腾上跑过，
    §11.1 是手工跑的、不在冒烟里。这一阶段把它们纳入回归。
    """
    stage("4b", "真实数据准备链路（prepare_v2_data.py → prepare_data.py）")
    import subprocess

    import numpy as np
    import soundfile as sf

    ft = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    wd = os.path.join(workdir, "pipe")
    wavd = os.path.join(wd, "wavs")
    os.makedirs(wavd, exist_ok=True)

    # 造打标 jsonl（真实 schema）+ wav：含首尾静音、一条标签不一致、一条纯底噪
    rng = np.random.default_rng(3)
    rows = []
    noisy_uid = None
    for i in range(n):
        uid = f"{i:032x}"
        sr = 16000
        sil = np.zeros(int(0.8 * sr), dtype="float32")
        if i == n - 1:
            # 纯底噪 → 应进"无动态"疑似清单。**整条**都是低幅噪声，不能加零静音垫边：
            # db_range 是在整段原始音频上算的，数字零帧约 -200dB、底噪约 -60dB，
            # 垫了边 db_range 就有 140dB，远超 6dB 阈值，永远判不成"无动态"。
            wav = (rng.standard_normal(int(3.6 * sr)) * 0.001).astype("float32")
            noisy_uid = uid
        else:
            t = np.arange(int(2.0 * sr)) / sr
            body = (0.25 * np.sin(2 * np.pi * rng.uniform(120, 200) * t)).astype("float32")
            wav = np.concatenate([sil, body, sil])
        sf.write(os.path.join(wavd, f"{uid}.wav"), wav, sr, subtype="PCM_16")
        lab = i % 2
        rows.append({
            "uttid": uid, "text": "打开前排车窗" if lab == 0 else "咱这是往哪走",
            "rej_label": lab,
            "true_rej_label": 1 - lab if i == 0 else lab,   # 第 0 条标签不一致 → 应被筛掉
            "evidence": ["用户指令包含明确的车控意图", "音频中用户发音清晰且为近场发令"],
            "reason": "用户明确发出车窗控制指令，应予响应。",
        })
    src = os.path.join(wd, "in.jsonl")
    with open(src, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    out = os.path.join(wd, "out")
    r1 = subprocess.run(
        f"cd {shlex.quote(ft)} && {shlex.quote(sys.executable)} prepare_v2_data.py "
        f"--dataset smoke {shlex.quote(src)} {shlex.quote(wavd)} "
        f"--out-dir {shlex.quote(out)} --workers 2 --val-size 2",
        shell=True, capture_output=True, text=True)
    if r1.returncode != 0:
        for line in (r1.stdout or r1.stderr or "")[-1500:].split("\n")[-15:]:
            log(f"  | {line}")
        return fail("prepare_v2_data.py 失败", "见上方输出")
    for line in (r1.stdout or "").split("\n"):
        if any(k in line for k in ("保留 ", "label_mismatch", "保留比例", "疑似", "无动态")):
            log(f"  | {line.rstrip()}")

    raw = os.path.join(out, "train_raw.jsonl")
    if not os.path.exists(raw):
        return fail("没有产出 train_raw.jsonl")
    got = [json.loads(l) for l in open(raw, encoding="utf-8")]
    if any("ref_audio" in g for g in got):
        return fail("产出里有 ref_audio", "VoiceDesign 不用参考音频")
    if not all({"audio", "text", "instruct"} <= set(g) for g in got):
        return fail("字段不全，应为 audio/text/instruct")
    # 12 条里第 0 条标签不一致必被筛掉；其余是否保留取决于 VAD，故只断言上界与筛除
    if len(got) >= n:
        return fail(f"产出 {len(got)} 条 >= 输入 {n} 条，标签不一致的那条没被筛掉")
    ok(f"prepare_v2_data.py 通过，产出 {len(got)} 条 {sorted(got[0])}（已筛除标签不一致）")

    # VAD 确实切了静音：正常样本原始 3.6s，切后应明显更短
    import soundfile as sf2

    # 只看正常样本：纯底噪那条整条都是噪声、没有首尾静音可切，3.6s 原样保留是对的
    durs = [sf2.info(g["audio"]).duration for g in got
            if noisy_uid not in os.path.basename(g["audio"])]
    if not durs or max(durs) > 3.4:
        return fail(f"VAD 似乎没切静音，正常样本最长产出 "
                    f"{max(durs) if durs else -1:.2f}s（原始 3.6s，含首尾各 0.8s 静音）")
    ok(f"VAD 切静音生效，正常样本 {min(durs):.2f}-{max(durs):.2f}s（原始 3.6s）")

    # 疑似清单必须生成且含那条纯底噪 —— 原先写成 if exists 则 ok，
    # 文件不存在时连 warn 都没有，测试照样全绿
    sp = os.path.join(out, "suspicious_vad.jsonl")
    if not os.path.exists(sp):
        return fail("没有生成 suspicious_vad.jsonl",
                    "纯底噪样本应被 db_range < 6 判为「无动态」，检查该判据是否失效")
    hits = [json.loads(l) for l in open(sp, encoding="utf-8")]
    if not any(h["uttid"] == noisy_uid and "无动态" in h["why"] for h in hits):
        return fail(f"纯底噪样本 {noisy_uid[:8]}… 未被标为「无动态」",
                    f"清单里有 {len(hits)} 条：{[(h['uttid'][:8], h['why']) for h in hits][:3]}")
    ok(f"suspicious_vad.jsonl 生成，纯底噪样本已标「无动态」（共 {len(hits)} 条）")

    # prepare_data.py：抽码 + 断点续跑
    codes = os.path.join(wd, "codes.jsonl")
    cmd = (f"cd {shlex.quote(ft)} && {shlex.quote(sys.executable)} prepare_data.py "
           f"--device {device} --tokenizer_model_path "
           f"{shlex.quote(os.path.join(model_path, 'speech_tokenizer'))} "
           f"--input_jsonl {shlex.quote(raw)} --output_jsonl {shlex.quote(codes)}")
    r2 = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r2.returncode != 0:
        for line in (r2.stdout or r2.stderr or "")[-1500:].split("\n")[-15:]:
            log(f"  | {line}")
        return fail("prepare_data.py 失败",
                    "NPU 上 --device 必须是 npu:N；若是算子不支持，试 --device cpu")
    n_all = sum(1 for _ in open(codes, encoding="utf-8"))
    ok(f"prepare_data.py 通过，抽码 {n_all} 条")

    # 断点续跑：两种截断都要验。行边界是 happy path，**半行才是真实崩溃的样子**
    # （写单行途中被 kill），而半行曾被当成完整一行跳过、下一条 JSON 粘在后面。
    src_audio = [json.loads(l)["audio"] for l in open(raw, encoding="utf-8")]
    for tag, half in (("行边界", False), ("半行", True)):
        keep = max(1, n_all // 2)
        lines = open(codes, encoding="utf-8").readlines()[:keep]
        body = "".join(lines)
        if half:
            body += lines[-1][: len(lines[-1]) // 2]   # 再粘半行
        with open(codes, "w", encoding="utf-8") as f:
            f.write(body)
        r3 = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if r3.returncode != 0:
            return fail(f"续跑（{tag}截断）失败", (r3.stdout or r3.stderr or "")[-400:])
        try:
            got2 = [json.loads(l) for l in open(codes, encoding="utf-8")]
        except Exception as e:
            return fail(f"续跑（{tag}截断）后 JSON 损坏: {e}",
                        "半行没被截掉，下一条被粘在后面了")
        if [g["audio"] for g in got2] != src_audio:
            return fail(f"续跑（{tag}截断）后与输入不是一一对应",
                        f"得 {len(got2)} 条，应 {len(src_audio)} 条且顺序一致")
        ok(f"断点续跑正确（{tag}截断到 {keep} → 补齐 {len(got2)}，逐条对应）")
    return codes


# ---------------------------------------------------------------- 5 序列布局


def s5_layout(model_path, codes_jsonl, language):
    stage(5, "序列布局（与推理侧逐格核对，纯 CPU）")
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import torch
    from dataset_voicedesign import VoiceDesignTTSDataset
    from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
    from qwen_tts.core.models.processing_qwen3_tts import Qwen3TTSProcessor

    cfg = Qwen3TTSConfig.from_pretrained(model_path)
    tc = cfg.talker_config
    proc = Qwen3TTSProcessor.from_pretrained(model_path)
    rows = [json.loads(l) for l in open(codes_jsonl)][:2]
    ds = VoiceDesignTTSDataset(rows, proc, cfg, language=language)
    items = [ds[0]] if len(ds) < 2 else [ds[0], ds[1]]   # 只有 1 条时不要 IndexError
    b = ds.collate_fn(items)
    ii, lab = b["input_ids"][0], b["codec_0_labels"][0]

    checks = []
    bos = [p for p in range(ii.shape[0]) if int(ii[p, 1]) == tc.codec_bos_id]
    eos = [p for p in range(ii.shape[0]) if int(ii[p, 0]) == cfg.tts_eos_token_id]
    if not bos or not eos:
        return fail("找不到 codec_bos 或 tts_eos", "collate 逻辑异常")
    bp, ep = bos[0], eos[0]
    checks.append(("codec_bos 与 tts_eos 分处两格", bp == ep + 1))
    checks.append(("tts_eos 那格 codec = codec_pad", int(ii[ep, 1]) == tc.codec_pad_id))
    checks.append(("codec_bos 那格 text = tts_pad", int(ii[bp, 0]) == cfg.tts_pad_token_id))
    nz = (lab != -100).nonzero()
    checks.append(("label 起点 = codec_bos 位", int(nz[0]) == bp))
    checks.append(("label 末位 = codec_eos", int(lab[int(nz[-1])]) == tc.codec_eos_token_id))
    # 对齐块紧邻在正文之前：从 codec_bos 往前数 = 正文长度 + 1(tts_eos) + nb
    prefix_expect, _nb = ds._codec_prefix()
    n_body = ds[0]["text_ids"].shape[1] - 3
    a = bp - 1 - n_body - _nb
    got = [int(ii[p, 1]) for p in range(a, a + _nb)]
    checks.append((f"对齐块 codec 前缀逐格相符（block 宽 {_nb}，{'think' if _nb == 5 else 'nothink'} 开头）",
                   got == prefix_expect))
    checks.append(("instruct 段 codec_embedding_mask 为 False",
                   not bool(b["codec_embedding_mask"][0, 0, 0])))
    allok = True
    for name, r in checks:
        (ok if r else lambda m: fail(m))(f"{name}: {'PASS' if r else 'FAIL'}")
        allok &= r
    if not allok:
        return fail("序列布局与推理侧不一致",
                    "这类错误不报错、loss 照降，但推理必崩。把上面 FAIL 项贴回来")
    return True


# ---------------------------------------------------------------- 6 前反向


def s6_train(model_path, codes_jsonl, workdir, attn, language, steps,
             dtype="fp32", batch=1, bucket=64, save_every=0):
    stage(6, f"前反向（{steps} 步，batch={batch}，bucket={bucket}）")
    ft = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(workdir, "ckpt")
    # 用 sys.executable：昇腾 conda 里不保证有 "python" 这个名字，命中时
    # returncode=127，而 s6 的兜底诊断会误报成 OOM/算子/accelerate 问题
    cmd = (f"cd {shlex.quote(ft)} && {shlex.quote(sys.executable)} sft_12hz_voicedesign.py "
           f"--init_model_path {shlex.quote(model_path)} "
           f"--train_jsonl {shlex.quote(codes_jsonl)} "
           f"--output_model_path {shlex.quote(out)} --batch_size {batch} --grad_accum 1 "
           f"--num_epochs 1 --max_steps {steps} --log_every 1 "
           f"--attn {attn} --language {language} --dtype {dtype} "
           f"--length-bucket {bucket} --save-every {save_every} 2>&1")
    log(f"  $ {cmd}\n")
    import subprocess

    t0 = time.time()
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    wall = time.time() - t0
    tail = (p.stdout or "")[-3000:]
    for line in tail.strip().split("\n")[-25:]:
        log(f"  | {line}")
    if p.returncode != 0:
        return fail("训练失败，见上方输出",
                    "OOM → 降 batch_size；\n"
                    "算子不支持 → 换 --attn eager；\n"
                    "accelerate 报错 → 版本问题，见阶段 0")
    losses = []
    for line in (p.stdout or "").split("\n"):
        if "Loss:" in line:
            try:
                losses.append(float(line.split("Loss:")[1].strip()))
            except ValueError:
                pass
    if len(losses) < 2:
        return fail("没解析到 loss", "训练可能没真正开始")
    # 不能比首末两个单点：batch_size=1 时每步是**单样本** loss。实测同配置的
    # 真实数据跑了两次，首末分别是 3.33→3.02 和 3.61→3.72（一降一升），
    # 而两次尾段均值都是 4.0 上下、标准差约 0.62 —— 首末点差纯属噪声。
    k = max(3, min(20, len(losses) // 4))
    head, tail = sum(losses[:k]) / k, sum(losses[-k:]) / k
    ok(f"loss 前 {k} 步均值 {head:.4f} → 后 {k} 步均值 {tail:.4f}（共 {len(losses)} 步；"
       f"首末单点 {losses[0]:.4f}/{losses[-1]:.4f} 仅供参考，单样本噪声大）")
    # 用**实际**步数：epoch 提前结束时（样本数/batch < steps）按请求步数算会虚高。
    # §12.6 那次 batch=8 就是 256 样本一个 epoch 只跑了 32 步，按 50 步算得 3.2，
    # 实际是 2.07 样本/秒。
    ok(f"batch={batch} → {len(losses) * batch / max(wall, 1e-9):.2f} 样本/秒"
       f"（实跑 {len(losses)} 步，含加载/编译摊销）")
    ok(f"训练总耗时 {wall:.1f}s（含加载/编译），约 {wall / len(losses):.1f}s/step")
    if tail >= head:
        warn("loss 均值没下降。注意 batch=1 且样本数 < 步数时，一个 epoch 内每条只见"
             "一次，本就不该有可见收敛——要看收敛请固定小样本集跑多个 epoch。"
             "若几百步仍不降且用的是 --dtype bf16，"
             "换 --dtype fp32 再试（bf16 下 Adam 动量也是 bf16，更新易下溢）")
    return out


# ---------------------------------------------------------------- 7 保存加载


def s7_ckpt(ckpt_root):
    stage(7, "checkpoint 完整性")
    d = os.path.join(ckpt_root, "checkpoint-epoch-0")
    if not os.path.isdir(d):
        return fail(f"没有产出 {d}", "训练提前退出且未落盘")
    # speech_tokenizer/model.safetensors 必须在：from_pretrained 强制从
    # <ckpt>/speech_tokenizer 加载，缺了就 OSError，ckpt 等于废的
    need = ["config.json", "model.safetensors",
            os.path.join("speech_tokenizer", "model.safetensors")]
    for f in need:
        if not os.path.exists(os.path.join(d, f)):
            return fail(f"缺 {f}")
    ok("config.json + 顶层 model.safetensors + speech_tokenizer/model.safetensors 都在")
    if os.path.exists(os.path.join(d, "model.safetensors.index.json")):
        return fail("残留 index.json", "会让新写的单文件失效，加载时静默读回原始权重")
    ok("无残留 index.json")
    c = json.load(open(os.path.join(d, "config.json")))
    if c.get("tts_model_type") != "voice_design":
        return fail(f"tts_model_type = {c.get('tts_model_type')}",
                    "会被 qwen3_tts_model.py 的 != voice_design 检查拒掉")
    ok(f"tts_model_type = voice_design, v2_train_language = {c.get('v2_train_language')}")
    return d


# ---------------------------------------------------------------- 8 推理


def s8_infer(ckpt_dir, attn, workdir, language):
    stage(8, "推理（验证 instruct 通路仍生效）")
    log(f"  language={language}（必须与训练一致，否则 codec 前缀块宽不同 = 静默失配）")
    import soundfile as sf
    import torch
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

    try:
        # 同 s3：不用 device_map，CPU 加载后整体搬 NPU。
        # speech_tokenizer 由 Qwen3TTSForConditionalGeneration.to() 的覆写一并搬走
        # （它是普通属性不是 nn.Module，nn.Module.to() 递归不到它），故这里不用再单独处理
        m = Qwen3TTSModel.from_pretrained(ckpt_dir, torch_dtype=torch.bfloat16,
                                          attn_implementation=attn)
        m.model.to("npu:0")
        m.device = torch.device("npu:0")
    except Exception as e:
        return fail(f"微调后的 ckpt 加载失败: {type(e).__name__}: {e}")
    st = getattr(m.model, "speech_tokenizer", None)
    if st is not None and getattr(st, "model", None) is not None:
        d = next(st.model.parameters()).device
        if d.type != "npu":
            return fail(f"speech_tokenizer 仍在 {d}",
                        "decode 会在 CPU 上跑 170M 模型逐帧前向，每条约 20 分钟。\n"
                        "检查 modeling_qwen3_tts.py 里 to() 的覆写是否还在")
        ok(f"speech_tokenizer 已在 {d}（decode 不会退回 CPU）")
    ok("微调后的 ckpt 可加载")

    text = "帮我把空调温度调低两度"
    ins = {"对机": "坐在车里对车载语音助手下指令，吐字清晰干脆，音调平稳，句尾利落收住",
           "对人": "开着车侧头跟旁边的人随口说话，语气随意、自然连读，不刻意咬字"}
    durs = {}
    for tag, s in ins.items():
        try:
            wavs, sr = m.generate_voice_design(text=text, instruct=s, language=language)
        except Exception as e:
            return fail(f"生成失败: {type(e).__name__}: {e}")
        p = os.path.join(workdir, f"infer_{tag}.wav")
        sf.write(p, wavs[0], sr)
        durs[tag] = len(wavs[0]) / sr
        ok(f"{tag}腔 生成成功 {durs[tag]:.2f}s → {p}")
    d = abs(durs["对机"] - durs["对人"]) / max(durs.values()) * 100
    log(f"\n  同一句文本、两条 instruct 的时长差 = {d:.1f}%")
    if d < 3:
        warn("时长几乎无差异 —— instruct 可能没起作用。听一下两个 wav 再判断")
    else:
        ok("时长有明显差异，instruct 通路在起作用（腔调对不对仍需听辨）")
    return True


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(description="V2 VoiceDesign 微调链路 NPU 冒烟测试")
    ap.add_argument("--model_path", required=True, help="VoiceDesign 权重目录")
    ap.add_argument("--stages", default="0-8", help="如 0-6 或 0,1,5")
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--language", default="Chinese")
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="fp32")
    ap.add_argument("--batch", type=int, default=1, help="阶段 6 的 batch_size")
    ap.add_argument("--bucket", type=int, default=64, help="阶段 6 的 --length-bucket")
    ap.add_argument("--save-every", type=int, default=0, help="阶段 6 的 --save-every")
    ap.add_argument("--real_jsonl", default=None, help="用真实数据（含 audio/text/instruct）")
    ap.add_argument("--synthetic_count", type=int, default=8,
                    help="合成样本条数。阶段 6 一个 epoch 只有 N 步，要跑 --steps 200 就传 256")
    ap.add_argument("--codes_jsonl", default=None,
                    help="已抽好 audio_codes 的 jsonl，给定则跳过阶段 4。"
                         "音频 tokenizer 在 NPU 上跑不动时，可先在 CPU 上抽好再用这个")
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--device", default="npu:0", help="阶段 4b 传给 prepare_data.py")
    ap.add_argument("--skip-pipeline", action="store_true",
                    help="跳过阶段 4b（生产数据准备链路）")
    ap.add_argument("--use-pipeline-codes", action="store_true",
                    help="后续阶段用 4b 产出的 codes，而非阶段 4 的")
    ap.add_argument("--report", default="npu_smoke_report.txt")
    args = ap.parse_args()

    want = set()
    for part in args.stages.split(","):
        if "-" in part:
            a, b = part.split("-")
            want |= set(range(int(a), int(b) + 1))
        else:
            want.add(int(part))

    wd = args.workdir or tempfile.mkdtemp(prefix="npu_smoke_")
    os.makedirs(wd, exist_ok=True)
    log(f"工作目录 {wd}\n模型 {args.model_path}\n阶段 {sorted(want)}")

    state = {"codes": args.codes_jsonl, "ckpt_root": None, "ckpt": None}
    failed = []
    ran = set()
    try:
        if 0 in want:
            ran.add(0)
            if not s0_env():
                failed.append(0)
                raise SystemExit
        if 1 in want:
            ran.add(1)
            if not s1_ops():
                failed.append(1)
        if 2 in want:
            ran.add(2)
            if not s2_attn(args.model_path):
                failed.append(2)
        if 3 in want:
            ran.add(3)
            if not s3_load(args.model_path, args.attn):
                failed.append(3)
                raise SystemExit
        # 4b：生产数据准备链路回归（prepare_v2_data.py → prepare_data.py）。
        # 与阶段 4 一起默认跑；--skip-pipeline 跳过，--use-pipeline-codes 让后续
        # 阶段改用 4b 产出的 codes（而非阶段 4 的合成数据）
        if 4 in want and not args.skip_pipeline and not args.codes_jsonl:
            pipe_codes = s4b_pipeline(args.model_path, wd, args.device)
            if not pipe_codes:
                failed.append("4b")
                raise SystemExit
            if args.use_pipeline_codes:
                state["codes"] = pipe_codes
        if 4 in want:
            ran.add(4)   # 给了 --codes_jsonl 也视为已满足（用户自带产物）
            # 注意 not (use_pipeline_codes and state["codes"])：4b 在上面已经把
            # 产物写进 state["codes"]，这里若无条件覆盖，--use-pipeline-codes
            # 就成了死参数（曾经如此，且文档还写着它生效）
            if not args.codes_jsonl and not (args.use_pipeline_codes and state["codes"]):
                state["codes"] = s4_codes(args.model_path, wd, args.real_jsonl,
                                          n_synth=args.synthetic_count,
                                          device=args.device)
                if not state["codes"]:
                    failed.append(4)
                    raise SystemExit
        if 5 in want and state["codes"]:
            ran.add(5)
            if not s5_layout(args.model_path, state["codes"], args.language):
                failed.append(5)
        if 6 in want and state["codes"]:
            ran.add(6)
            state["ckpt_root"] = s6_train(args.model_path, state["codes"], wd,
                                          args.attn, args.language, args.steps,
                                          args.dtype, args.batch, args.bucket,
                                          args.save_every)
            if not state["ckpt_root"]:
                failed.append(6)
                raise SystemExit
        if 7 in want and state["ckpt_root"]:
            ran.add(7)
            state["ckpt"] = s7_ckpt(state["ckpt_root"])
            if not state["ckpt"]:
                failed.append(7)
        if 8 in want and state["ckpt"]:
            ran.add(8)
            if not s8_infer(state["ckpt"], args.attn, wd, args.language):
                failed.append(8)
    except SystemExit:
        pass
    except Exception:
        log("\n未预期异常：")
        for line in traceback.format_exc().split("\n"):
            log(f"  {line}")
        failed.append(-1)

    # 请求了但因缺前置产物被静默跳过的阶段：必须显式报出来，
    # 否则「失败阶段: 无 —— 全部通过 ✅」可能意味着一步都没跑
    skipped = sorted(want - ran)
    log(f"\n{'=' * 66}")
    log("失败阶段: " + (", ".join(map(str, failed)) if failed else "无 —— 全部通过 ✅"))
    if skipped:
        log(f"⚠️  请求但未执行的阶段: {skipped}"
            f"（缺前置产物：5/6 需要 --codes_jsonl 或先跑阶段 4，7/8 需要先跑 6）")
    log(f"产物在 {wd}")
    log("=" * 66)
    with open(args.report, "w", encoding="utf-8") as f:
        f.write("\n".join(REPORT) + "\n")
    print(f"\n报告已写入 {args.report}（把它贴回来即可定位问题）")
    sys.exit(1 if (failed or skipped) else 0)


if __name__ == "__main__":
    main()
