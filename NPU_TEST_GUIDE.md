# V2 链路 NPU 上机测试指南

> 分支 `v2_exp`。目标：在昇腾 910B2 上验证 V2 VoiceDesign 微调链路能跑通。
>
> 全部代码在 CUDA（H200）上端到端验证过，**NPU 侧有三处我在开发机上验证不了**，
> 这份测试就是为了把它们打掉。

---

## 0. 一句话

```bash
cd finetuning/scripts
. ./npu_env.sh
python npu_smoke_test.py --model_path /path/to/Qwen3-TTS-12Hz-1.7B-VoiceDesign
```

跑完把 `npu_smoke_report.txt` 贴回来。**不需要真实数据**——脚本自己造合成音频。

---

## 1. 三个待验证的未知

| # | 未知项 | 为什么本机验证不了 | 落在哪个阶段 |
|---|---|---|---|
| 1 | `sdpa` 在 torch_npu 2.1.0 上的覆盖度 | 开发机无 NPU。可能可用、可能静默退化到 math 分支（能跑但慢且费显存） | 阶段 2 |
| 2 | `accelerate 1.0.0` 够不够用 | `pyproject` 钉 1.12.0（要 py≥3.10），而昇腾现役 conda 是 **py3.9**，只能装 1.0.0 | 阶段 0 |
| 3 | bf16 无 fp32 主权重时能否收敛 | 需要几百步才看得出来 | 阶段 6（只能看趋势） |

另有一个已知必踩项：**`transformers` 在那台机器上没装**（`NPU_ENV.md §2` 确认）。
`pyproject` 要求 4.57.3，其 `requires-python >=3.9.0`，py3.9 可以装。

---

## 2. 装依赖

昇腾现役 conda `PyTorch-2.1.0`（py3.9）里缺 `transformers` 和 `qwen_tts`：

```bash
pip install transformers==4.57.3          # requires-python >=3.9.0，py3.9 可用
pip install -e . --no-deps                # 装 qwen_tts（在仓库根目录），必须 --no-deps
```

**`pip install -e .` 必须带 `--no-deps`**，两个原因：
1. pyproject 钉了 `accelerate==1.12.0`，它要求 py≥3.10，py3.9 下 pip 直接解析失败；
2. pyproject 的 `torchaudio` 未钉版本，pip 会拉最新 torchaudio（其依赖精确钉死
   torch==2.x），把机器上的 torch 2.1.0 顶掉，torch_npu 2.1.0 随即报废。

**`accelerate` 不要照 pyproject 装 1.12.0** —— 它要求 py≥3.10。用机器上已有的
1.0.0 即可，本链路只用到 `Accelerator` 的 6 个接口（`accumulate` / `backward` /
`clip_grad_norm_` / `prepare` / `unwrap_model` / `is_main_process`），阶段 0 会逐个校验。

其余运行依赖（safetensors / librosa / soundfile 等）NPU_ENV.md §2 确认已装。
`pip install transformers==4.57.3` 会连带装 huggingface_hub/tokenizers；py3.9 下
若 tokenizers 解析失败，退回 transformers 4.5x 的 py3.9 组合，阶段 0 会验 import。

pip 走华为内部镜像（`pip.conf` 已配好）。

---

## 3. 分阶段测什么

每阶段独立可诊断，失败会直接告诉你是哪一层的问题。

| 阶段 | 测什么 | 失败意味着 |
|---|---|---|
| **0** | torch_npu / transformers / accelerate / qwen_tts | 环境没装好，先解决这个 |
| **1** | bf16 matmul、显存查询 | CANN 与 torch_npu 版本不配套 |
| **2** | `sdpa` vs `eager` 能否跑 + 相对耗时 | sdpa 不可用 → 启动脚本传 `ATTN=eager` |
| **3** | VoiceDesign 权重加载到 NPU | 若报 flash_attention_2 → 传 `--attn sdpa` |
| **4** | 抽 `audio_codes`（`prepare_data.py` 那一步） | 音频 tokenizer 有算子不支持 → 这步放 CPU 跑 |
| **5** | 序列布局与推理侧逐格核对（纯 CPU） | **最要命的一类**，见 §5 |
| **6** | 若干步前反向，看 loss | OOM → 降 batch；算子不支持 → 换 eager |
| **7** | ckpt 完整性 | 缺文件 / 残留 index.json / model_type 不对 |
| **8** | 用微调后 ckpt 合成，验 instruct 通路 | 时长无差异 → instruct 可能没起作用 |

常用变体：

```bash
# 只验环境和布局，不训练（最快，1 分钟内）
python npu_smoke_test.py --model_path ... --stages 0-5

# sdpa 有问题时
python npu_smoke_test.py --model_path ... --attn eager

# 用真实数据（jsonl 需含 audio/text/instruct）
python npu_smoke_test.py --model_path ... --real_jsonl /path/train_raw.jsonl

# 多训几步看 loss 趋势（阶段 6 需要阶段 4 的产物，不能只跑 6）
python npu_smoke_test.py --model_path ... --stages 4-6 --steps 200

# 音频码已抽好的话：
python npu_smoke_test.py --model_path ... --stages 5-6 \
  --codes_jsonl /path/train_codes.jsonl --steps 200
```

---

## 4. 正式训练

冒烟过了之后：

```bash
# 单卡 debug
cd finetuning/scripts
MODEL_PATH=/path/VoiceDesign RAW_JSONL=./data_v2/train_raw.jsonl \
  bash run_v2_npu_debug.sh

# ModelArts 集群（作业变量由平台注入）
bash run_v2_npu_cluster.sh
```

**1.7B 全参微调单卡就够**：静态显存约 21GB（bf16 参数 3.4 + 梯度 3.4 +
Adam m/v fp32 13.6；accelerate 的 bf16 混合精度**不保留** fp32 主权重——
这也正是 §1 未知项 #3 的由来），910B2 是 64GB。集群多卡只为吞吐，走
accelerate 原生 DDP，**不需要 FSDP / ZeRO / MindSpeed**。

> MindSpeed 的价值在模型并行。原本 MiMo-Audio 7B 那条线要 128GB、单卡放不下，
> 才有移植的必要；换成 1.7B 后这个前提没了。

---

## 5. 为什么阶段 5 单列出来

序列布局错了**不会报错、loss 照样降**，只有推理时才崩。这类问题在本项目里已经
踩到两次：

1. 官方 `sft_12hz.py` 是「脚本手工 shift + HF `loss_function` 内部再 shift」= **双 shift**
2. 我第一版把 `tts_eos` 和 `codec_bos` 并进了同一格，导致尾部整体错位一格

阶段 5 会逐格核对这几项（纯 CPU，不占 NPU）：

```
codec_bos 与 tts_eos 分处两格
tts_eos 那格的 codec = codec_pad
codec_bos 那格的 text = tts_pad
label 起点 = codec_bos 位          ← 该格预测第一帧音频码
label 末位 = codec_eos
codec 前缀首 token = think（Chinese）/ nothink（Auto）
instruct 段的 codec_embedding_mask = False
```

任何一项 FAIL 都不要往下训，直接把报告贴回来。

---

## 6. 一个训练/推理必须对齐的参数

`--language` 决定 codec 前缀的形状：

| language | codec 前缀 | 块宽 |
|---|---|---|
| `Chinese`（训练默认） | `[think, think_bos, lang_id, think_eos, codec_pad]` | 5 |
| `Auto` | `[nothink, think_bos, think_eos, codec_pad]` | 4 |

而 `generate_voice_design(language=...)` **默认是 Auto**。训练用 Chinese、推理用
Auto 就是静默失配。

训练用的值已写进 ckpt 的 `config.json` 的 `v2_train_language` 字段，
**推理时显式传同一个值**。

---

## 7. 报告怎么用

`npu_smoke_report.txt` 里每阶段都有 ✅/⚠️/❌ 和具体数值。贴回来我能直接定位。
特别关注这几行：

- 阶段 2 的 `sdpa` / `eager` 耗时对比 —— 决定用哪个后端
- 阶段 5 的 7 项 PASS/FAIL —— 布局是否正确
- 阶段 6 的 loss 首末值 —— 是否在降
- 阶段 8 的时长差百分比 —— instruct 是否起作用

---

## 附：已知会失败的地方（已在代码里处理，列出来备查）

| 项 | 处理 |
|---|---|
| `flash_attention_2` 是 CUDA kernel | `--attn` 默认已改 `sdpa` |
| `prepare_data.py --device` 默认 `cuda:0` | NPU 上必须显式传 `npu:0`，已加 help 提示 |
| `modeling_qwen3_tts.py` 有 PEP 604 注解，py3.9 import 即崩 | 已加 `from __future__ import annotations` |
| 集群上自设 `ASCEND_RT_VISIBLE_DEVICES` 会和调度打架 | 集群脚本不设，交给 ModelArts |
| ckpt 保存后残留 `index.json` 会让新权重失效 | 保存前已删 |

---

## 复核问题清单（2026-08-15）

对本文档逐条对照了冒烟脚本 / 训练脚本 / 数据集 / 推理代码 / NPU_ENV.md，发现
4 个问题，均已在正文或脚本中就地修正：

1. **[已修] `npu_smoke_test.py` 阶段 4 必崩（encode 返回值处理错）**
   `Qwen3TTSTokenizer.encode()` 返回 `Qwen3TTSTokenizerV2EncoderOutput`
   （ModelOutput），不是 list/tuple/tensor。原代码
   `c = c[0] if isinstance(c, (list, tuple)) else c` 不触发，
   `hasattr(c, "tolist")` 为 False → `audio_codes` 存进整个 ModelOutput 对象，
   随后 json.dumps 抛 TypeError，阶段 4 以「未预期异常」收场。
   已改为 `c.audio_codes[0].cpu().tolist()`（返回类型见
   `qwen_tts_tokenizer.py:208-260`、`modeling_qwen3_tts_tokenizer_v2.py:54-60`）。

2. **[已修] §3 原示例 `--stages 6` 会静默什么都不跑**
   原 main() 里 `if 6 in want and state["codes"]:`——没跑阶段 4 且没传
   `--codes_jsonl` 时 codes 为 None，阶段 6 被静默跳过，报告还打出
   「失败阶段: 无 —— 全部通过 ✅」，看起来全绿其实一步没跑
   （`--stages 5`、`--stages 8` 单独跑同样中招）。两处修复：
   - 指南示例改为 `--stages 4-6`（另给了带 `--codes_jsonl` 的写法）；
   - smoke 脚本加了「请求但未执行的阶段」告警并把退出码置 1，静默跳过不再可能。

3. **[已修] §2 原 `pip install -e .` 在 py3.9 上会失败且危险**
   会按 pyproject 解析 `accelerate==1.12.0`（py≥3.10）直接失败；即便不失败，
   未钉版本的 `torchaudio` 会把 torch 2.1.0 顶掉、连带废掉 torch_npu。
   已改为 `pip install -e . --no-deps` 并写明原因。
   （accelerate 1.12.0 的 requires-python 未联网核实，以 §1 自身叙述为准；
   无论它是否支持 py3.9，torchaudio 顶掉 torch 的风险都成立。）

4. **[已修] §4 显存估算与未知项 #3 自相矛盾**
   原估算 27GB 含「fp32 master 6.8」，但 accelerate 的 bf16 混合精度
   **不保留** fp32 主权重（这正是未知项 #3 要验证的事）。实际静态显存
   ≈ 20.4GB + 激活。已把指南 §4、run_v2_npu_debug.sh 注释、cluster 脚本注释、
   smoke 阶段 1 告警阈值统一为 ~21GB。结论（单卡 64GB 够）不变。

核对过且无误的项（备查）：
- pyproject 钉 `transformers==4.57.3` / `accelerate==1.12.0`，6 个 Accelerator
  接口清单与阶段 0 校验一致
- 阶段 5 的 7 项布局检查与 `dataset_voicedesign.py` collate 逐格对应
- §6 前缀表（Chinese 5 格 / Auto 4 格）与 `_codec_prefix()` 一致；
  `generate_voice_design(language=...)` 默认 Auto（`qwen3_tts_model.py:641,695`）
- `v2_train_language` 写入 config（`sft_12hz_voicedesign.py:190`）
- `prepare_data.py --device` 默认 cuda:0；`modeling_qwen3_tts.py` 的
  `from __future__ import annotations`；ckpt 保存前删 index.json
  （`sft_12hz_voicedesign.py:199-201`）；集群脚本不设 ASCEND_RT_VISIBLE_DEVICES
- NPU_ENV.md §2（transformers 缺失、accelerate 1.0.0、py3.9）与 §6.1（CANN 配方）
  与本指南叙述一致
