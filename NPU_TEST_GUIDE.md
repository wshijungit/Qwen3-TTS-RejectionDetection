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
| 3 | bf16 下能否收敛 | 需要几百步才看得出来 | 阶段 6（只能看趋势） |

关于 #3，**开发机上已实测出更确切的结论**（`--dtype bf16` 时）：

```
参数 dtype / 梯度 dtype        : bfloat16
exp_avg (m) / exp_avg_sq (v)   : bfloat16      <- Adam 两个动量也是 bf16
有 fp32 主权重吗               : False
-> 1.7B 静态约 12.7 GiB
```

不只是没有 fp32 主权重，**Adam 动量本身也是 bf16**。`exp_avg_sq` 用 8 位尾数
累积梯度平方，lr 2e-5 下更新量下溢的风险比原先判断的大。

**因此 `--dtype` 默认已改为 `fp32`**（fp32 权重 + autocast bf16 计算，约 25GiB）。
910B2 是 64GB，完全吃得下，没有理由为省显存冒收敛风险。显存真紧再传 `--dtype bf16`。

> 附带修掉的一个坑：训练循环原先直接调 `model.talker(...)`，**绕过了被 accelerate
> `prepare` 的模块的 forward**，导致 autocast 完全不生效（fp32 时就是纯 fp32 跑，
> 慢数倍）、多卡时 DDP 梯度也**完全不同步**（每张卡各训各的，不报错、loss 照降）。
> 现已把整个计算封进 `V2TrainStep.forward`，两者一并解决。

另有一个已知必踩项：**`transformers` 在那台机器上没装**（`NPU_ENV.md §2` 确认）。
`pyproject` 要求 4.57.3，其 `requires-python >=3.9.0`，py3.9 可以装。

---

## 2. 装依赖

昇腾现役 conda `PyTorch-2.1.0`（py3.9）里缺 `transformers` 和 `qwen_tts`：

```bash
pip install transformers==4.57.3          # requires-python >=3.9.0，py3.9 可用
pip install -e . --no-deps                # 装 qwen_tts（在仓库根目录），必须 --no-deps
pip install sox onnxruntime einops "torchaudio==2.1.0"   # --no-deps 漏掉的运行时依赖
```

**第三条不能省。** `--no-deps` 只装了 qwen_tts 本身，而 `import qwen_tts` 的依赖链是：

```
qwen_tts/__init__ → inference/qwen3_tts_model → core/__init__
  → tokenizer_25hz/modeling_..._v1.py → vq/speech_vq.py  → sox / onnxruntime / torchaudio
                                       → vq/core_vq.py   → einops
```

这四个 `NPU_ENV.md §2` 的已装清单里**一个都没有**，不装的话阶段 0 最后一行
`import qwen_tts` 直接 `ModuleNotFoundError`。

`torchaudio` **必须钉 2.1.0** —— 不钉版本 pip 会拉最新的，其依赖精确钉死
`torch==2.x`，会把机器上的 torch 2.1.0 顶掉、连带废掉 torch_npu。
（aarch64 wheel 可用性需上机确认；装不上的话，把 `qwen_tts/core/__init__.py` 里
25Hz tokenizer 的 import 改成惰性加载亦可——本链路走 12Hz，根本用不到它。）

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

# 用 bf16 精度测（显存紧时）
python npu_smoke_test.py --model_path ... --dtype bf16

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

**1.7B 全参微调单卡就够**，两种精度实测（开发机 H200，均跑通并落盘）：

| `--dtype` | 参数/梯度/m/v | 静态显存 | 说明 |
|---|---|---|---|
| `fp32`（**默认**） | fp32 | 约 25 GiB | fp32 权重 + autocast bf16 计算，收敛更稳 |
| `bf16` | 全 bf16 | 约 12.7 GiB | 省显存，但 Adam 动量也是 bf16，有下溢风险 |

（1.7B × 4 份状态 × 4B = 27.2 GB = 25.3 GiB；bf16 那行 1.7B × 4 × 2B = 13.6 GB = 12.7 GiB。
两行都按 GiB 记。）

正式训练也可用 `DTYPE=bf16 bash run_v2_npu_debug.sh` 切换。

910B2 是 64GB，默认的 fp32 完全吃得下。集群多卡只为吞吐，走 accelerate 原生
DDP，**不需要 FSDP / ZeRO / MindSpeed**。

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

## 8. 尚未确认的点（上机时请留意，有结论请回填本节）

以下几条在开发机（CUDA）上无法验证，按「踩到的概率 × 踩到后的代价」排序。
**遇到任何一条，请把现象补在对应条目下，别只在群里说。**

### 8.1 `torchaudio==2.1.0` 的 aarch64 wheel 是否装得上 —— 阻塞级

§2 第三条依赖里最不确定的一个。装不上就 `import qwen_tts` 直接失败，整条链路起不来。

**兜底方案**（不用装 torchaudio）：把 `qwen_tts/core/__init__.py` 里 25Hz tokenizer
的 import 改成惰性加载。本链路走 12Hz，`tokenizer_25hz/` 全程用不到，
`sox` / `onnxruntime` / `torchaudio` / `einops` 四个依赖都是被它拖进来的。

### 8.2 `transformers 4.57.3` 在 torch 2.1.0 上能否正常运行 —— 阻塞级

4.56+ 的 setup 声明 `torch>=2.2`。好消息是 torch 不在它的 `install_requires` 里，
所以装它**不会**顶掉 torch 2.1.0；坏消息是运行期是否用到 2.2+ 的 API
（`torch.nn.attention` 系列、新的 pytree 注册等）离线核实不了。

**上机第一步就单独验**：`python -c "import transformers, qwen_tts"`。
真不兼容的话退到 transformers 4.5x 中支持 py3.9 + torch 2.1 的版本，
但**必须 ≥ 4.41**（见 §1）。

### 8.3 CANN 的 `set_env.sh` 与 `set -u` —— 中等

两个启动脚本现在自己设 `set -euo pipefail`，且在 source `npu_env.sh` **之前**设。
Ascend 的 `set_env.sh` 里常有 `export PYTHONPATH=...:$PYTHONPATH` 这类对未设变量的
裸引用，`set -u` 下会 unbound variable 报错。具体取决于 CANN 版本。

若上机报这个错：把启动脚本里的 `set -euo pipefail` 改成 `set -eo pipefail`（去掉 `-u`）。

### 8.4 `sdpa` 是否真的走融合 kernel —— 影响性能不影响正确性

阶段 2 现在测的是 `is_causal=True` 路径，而**训练实际走的是 4D mask 路径**
（collate 产出的 `attention_mask` 带 padding 0，transformers 的
`sdpa_attention_forward` 在 `attention_mask is not None` 时会置 `is_causal=False`
并传 `attn_mask=`）。这条路在不少后端上不走融合 kernel。

也就是说**阶段 2 的结论未必迁移到训练**。判断依据看实际训练的 step 耗时和显存：
明显偏离预期就试 `ATTN=eager` 对比。

### 8.5 bf16 下能否收敛 —— 已通过默认值规避，但值得验证

`--dtype` 默认已是 `fp32`，规避了这个风险。若出于显存原因必须用 `bf16`，
注意此时**参数 / 梯度 / Adam 两个动量全是 bf16**（`torch.optim.AdamW` 用
`zeros_like(p)` 建 state）。`exp_avg_sq` 用 8 位尾数累积梯度平方，lr 2e-5 下
更新量有下溢风险。跑几百步看 loss 是真降还是在抖。

### 8.6 多卡梯度同步 —— 已修，但没在真多卡上验证过

`14c3e42` 把计算封进 `V2TrainStep.forward` 后，DDP 的 allreduce 应该恢复了，
但开发机上只做了单卡验证。

**上机跑多卡时请验一次**：训练几步后比对两个 rank 上同一个权重张量的
`.sum()`，应完全相同。不同就说明还在各训各的。

### 8.7 分片 checkpoint 的孤儿文件 —— 低

`_save` 用 `shutil.copytree` 把源模型目录整份拷过来再覆盖 `model.safetensors`，
并删掉 `index.json`。若源目录是分片格式，**shard 文件本身不会被删**，
每个 epoch 目录会多出若干 GB 的孤儿文件。1.7B bf16 约 3.4GB < 5GB 默认分片阈值，
大概率是单文件，但值得看一眼 ckpt 目录大小。


## 9. 请 NPU 侧执行并回填

下面是**开发机上做不了、必须上机测**的项。按顺序做，把结果填进「实测」列，
连同 `npu_smoke_report.txt` 一起推回来。任何一步失败就停下，把现象补进 §8 对应条目。

### 9.1 装完先验 import（对应 §8.1 / §8.2，两条阻塞项）

```bash
python -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__)"
python -c "import transformers; print(transformers.__version__)"
python -c "import qwen_tts; print('qwen_tts OK')"
```

| 项 | 预期 | 实测 |
|---|---|---|
| torch / torch_npu | 2.1.0 / 2.1.0.post8 | |
| transformers | ≥ 4.41（目标 4.57.3） | |
| `import qwen_tts` | 无报错 | |
| `torchaudio==2.1.0` 装上了吗 | 是 | |

装不上 torchaudio 就走 §8.1 的兜底（25Hz tokenizer 改惰性 import）。

### 9.2 冒烟测试

```bash
cd finetuning/scripts && . ./npu_env.sh
python npu_smoke_test.py --model_path <VoiceDesign 目录>
```

| 阶段 | 要回填的数 | 实测 |
|---|---|---|
| 1 | bf16 matmul 耗时、显存总量 | |
| 2 | sdpa 耗时 / eager 耗时 | |
| 4 | audio_codes shape | |
| 5 | 7 项是否全 PASS | |
| 6 | loss 首值 → 末值 | |
| 8 | 两条 instruct 的时长差 % | |

### 9.3 多卡梯度同步（对应 §8.6）

**多卡正式训练前必做。** 开发机上只有 CUDA，NPU 的 HCCL 路径没验过。

```bash
torchrun --nproc_per_node 2 check_ddp_sync.py \
    --model_path <VoiceDesign> --train_jsonl <train_codes.jsonl>
```

各 rank 喂不同样本，梯度同步正常则 `grad.sum()` 完全一致。
开发机 2×H200 上的参考输出：

```
各 rank 的 grad.sum(): ['-6.535583e-01', '-6.535583e-01']
✅ 梯度已同步（各 rank 一致）—— DDP allreduce 正常
```

| 项 | 预期 | 实测 |
|---|---|---|
| 各 rank grad.sum() | 完全一致 | |

打 ❌ 就**不要跑多卡正式训练**，等于白跑。

### 9.4 真实训练路径下的 sdpa vs eager（对应 §8.4）

阶段 2 测的是 `is_causal=True` 的合成 benchmark，而训练走的是 4D mask 路径，
**结论未必迁移**。用真实训练各跑一次对比：

```bash
for A in sdpa eager; do
  ATTN=$A MAX_STEPS=20 OUTPUT_DIR=/tmp/attn_$A bash run_v2_npu_debug.sh
done
```

| ATTN | 20 步耗时 | 峰值显存（npu-smi） | 实测 |
|---|---|---|---|
| sdpa | | | |
| eager | | | |

sdpa 不比 eager 快就说明没走融合 kernel，按快的那个定 §4 的默认值。

### 9.5 收敛趋势（对应 §8.5）

默认 `--dtype fp32` 已规避 bf16 的下溢风险，但仍要确认 loss 是真降不是抖：

```bash
python npu_smoke_test.py --model_path ... --stages 4-6 --steps 200
```

| 项 | 实测 |
|---|---|
| 200 步 loss 首 / 中 / 末 | |
| 单步耗时 | |

有余力再跑一次 `--dtype bf16` 对比，能直接回答 §8.5。

### 9.6 定正式训练的规模

上面几项跑完就能算了：

| 项 | 来自 | 实测 |
|---|---|---|
| 单步耗时（batch=2） | 9.5 | |
| 峰值显存 | 9.4 | |
| 可用的 batch_size | 由显存反推 | |
| 43.5w 条 1 epoch 预估耗时 | 单步耗时 × 步数 | |


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
