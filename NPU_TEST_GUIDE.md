# V2 链路 NPU 上机测试指南

> 分支 `v2_exp`。目标：在昇腾 910B1 上验证 V2 VoiceDesign 微调链路能跑通。
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
910B1 是 64GB，完全吃得下，没有理由为省显存冒收敛风险。显存真紧再传 `--dtype bf16`。

> 附带修掉的一个坑：训练循环原先直接调 `model.talker(...)`，**绕过了被 accelerate
> `prepare` 的模块的 forward**，导致 autocast 完全不生效（fp32 时就是纯 fp32 跑，
> 慢数倍）、多卡时 DDP 梯度也**完全不同步**（每张卡各训各的，不报错、loss 照降）。
> 现已把整个计算封进 `V2TrainStep.forward`，两者一并解决。

另有一个已知必踩项：**`transformers` 在那台机器上没装**（`NPU_ENV.md §2` 确认），
且**不能照 pyproject 装 4.57.3**——上机实测 4.56+ 无条件调用 torch>=2.2 的
`torch.utils._pytree.register_pytree_node`，torch 2.1.0 上 `import transformers`
即崩（§8.2 已回填实测结论）。现役版本钉 **4.55.2**。

---

## 2. 装依赖

昇腾现役 conda `PyTorch-2.1.0`（py3.9）里缺 `transformers` 和 `qwen_tts`：

```bash
pip install --user transformers==4.55.2   # 4.57.3 在 torch 2.1.0 上 import 即崩，见 §8.2
pip install --user -e . --no-deps         # 装 qwen_tts（在仓库根目录），必须 --no-deps
pip install --user sox onnxruntime einops "torchaudio==2.1.0"   # --no-deps 漏掉的运行时依赖
```

**必须 `--user`（实测）**：base 镜像的 conda 环境里有 root 所有的文件（如
tokenizers 的 dist-info），普通 `pip install` 卸载/覆盖时报
`Permission denied: 'METADATA'`。user site-packages 优先级高于环境目录，正好覆盖；
这也是参考仓库脚本对 zarr/mindspore 用 `--user` 的原因。

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
（**实测**：本机环境里已有 torchaudio 2.1.0（aarch64），§8.1 阻塞项解除。）

**`pip install -e .` 必须带 `--no-deps`**，两个原因：
1. pyproject 钉了 `accelerate==1.12.0`，它要求 py≥3.10，py3.9 下 pip 直接解析失败；
2. pyproject 的 `torchaudio` 未钉版本，pip 会拉最新 torchaudio（其依赖精确钉死
   torch==2.x），把机器上的 torch 2.1.0 顶掉，torch_npu 2.1.0 随即报废。

**`accelerate` 不要照 pyproject 装 1.12.0** —— 它要求 py≥3.10。用机器上已有的
1.0.0 即可，本链路只用到 `Accelerator` 的 6 个接口（`accumulate` / `backward` /
`clip_grad_norm_` / `prepare` / `unwrap_model` / `is_main_process`），阶段 0 会逐个校验。

其余运行依赖（safetensors / librosa / soundfile 等）NPU_ENV.md §2 确认已装。
transformers 4.55.2 连带 tokenizers 0.21.4（实测可用），阶段 0 会验 import。

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
# 单卡 debug（数据走默认路径 $FT_DIR/data_v2/train_{raw,codes}.jsonl）
cd finetuning/scripts
MODEL_PATH=/path/VoiceDesign bash run_v2_npu_debug.sh
# 要覆盖数据路径请传绝对路径：脚本里 RAW_JSONL/TRAIN_JSONL 会 realpath，
# 相对路径按脚本启动目录（finetuning/scripts）解析，不是 $FT_DIR

# ModelArts 集群（作业变量由平台注入）
bash run_v2_npu_cluster.sh
```

两个启动脚本都**内置依赖安装**（§2 三条 + accelerate/safetensors/librosa/soundfile
钉版本），`SKIP_INSTALL=1` 可跳过——与参考仓库 `duplex_huanyu_qwen35/pretrain_al_cls.sh`
的「脚本内 pip install」惯例一致，集群容器是干净的。启动器用 `torch.distributed.run`
而非参考仓库的 `msrun`：本链路不依赖 megatron/mindspore，torch_npu 的 hccl 后端
直接可用（这是与参考仓库唯一刻意保留的差异）。端口约定对齐参考仓库：
`MASTER_PORT=6411`、`HCCL_IF_BASE_PORT=64111`。

**1.7B 全参微调单卡就够**，两种精度实测（开发机 H200，均跑通并落盘）：

| `--dtype` | 参数/梯度/m/v | 静态显存 | 说明 |
|---|---|---|---|
| `fp32`（**默认**） | fp32 | 约 25 GiB | fp32 权重 + autocast bf16 计算，收敛更稳 |
| `bf16` | 全 bf16 | 约 12.7 GiB | 省显存，但 Adam 动量也是 bf16，有下溢风险 |

（1.7B × 4 份状态 × 4B = 27.2 GB = 25.3 GiB；bf16 那行 1.7B × 4 × 2B = 13.6 GB = 12.7 GiB。
两行都按 GiB 记。）

正式训练也可用 `DTYPE=bf16 bash run_v2_npu_debug.sh` 切换。

910B1 是 64GB，默认的 fp32 完全吃得下。集群多卡只为吞吐，走 accelerate 原生
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

**✅ 已解决（2026-08-15 实测）**：本机环境里已有 torchaudio 2.1.0（aarch64，
`/home/ma-user/anaconda3/envs/PyTorch-2.1.0/lib/python3.9/site-packages/torchaudio`），
import 正常。兜底方案保留备查但没用到。

**兜底方案**（不用装 torchaudio）：把 `qwen_tts/core/__init__.py` 里 25Hz tokenizer
的 import 改成惰性加载。本链路走 12Hz，`tokenizer_25hz/` 全程用不到，
`sox` / `onnxruntime` / `torchaudio` / `einops` 四个依赖都是被它拖进来的。

### 8.2 `transformers 4.57.3` 在 torch 2.1.0 上能否正常运行 —— 阻塞级

**✅ 已解决（2026-08-15 实测）**：**4.57.3 确实崩**，而且比预想更狠——
`torch_npu` 的 import 链经 `torch.onnx._internal.fx.patcher` 会 import transformers，
所以 transformers 一崩连 `import torch_npu` 都失败。根因正是 pytree 注册：
4.56+ 无条件调 `torch.utils._pytree.register_pytree_node`（torch 2.2+ 才有），
4.57.3 在 torch 2.1.0 上报
`AttributeError: module 'torch.utils._pytree' has no attribute 'register_pytree_node'`。

实测结论：**钉 `transformers==4.55.2`**（4.5x 中最后一个带 `torch>=2.2` 版本守卫
的版本，已逐版本核对源码；连带 tokenizers 0.21.4）。同时给 qwen_tts 加了两处
版本兼容（开发机 4.57.3 不受影响）：
- `@check_model_inputs()`：4.55.2 里还是裸装饰器（缺 func 报 TypeError），
  4.57+ 才是工厂形式 → `modeling_qwen3_tts_tokenizer_v2.py` 加了两种形式都认的垫片
- `from_pretrained(dtype=...)`：4.55.2 不认这个新参数 → 冒烟脚本改回老牌的
  `torch_dtype=...`（新旧通吃）

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


## 8.8 speech_tokenizer 设备搬迁 —— **已在库代码修复**（2026-08-15）

NPU 侧实测「推理像没在用 NPU、每条约 20 分钟、AICore 0%」，根因有二：

**① `speech_tokenizer` 从来没被搬到 NPU。** 它是普通 class 不是 `nn.Module`
（`inference/qwen3_tts_tokenizer.py:44`），当普通属性挂在模型上，
`nn.Module.to()` 只递归注册过的子模块，**走不到它**。于是
`model.to("npu:0")` 之后，decode 用的那个约 170M 编解码器仍留在 CPU，
逐帧前向全在 CPU 上跑。

> 它不是 `nn.Module` 是有意为之——这样 `parameters()` / `state_dict()`
> 不会混进 codec 权重，checkpoint 才干净。**同一个设计事实，一面是干净的
> checkpoint，另一面就是设备搬迁漏掉它。**

**修复**：给 `Qwen3TTSForConditionalGeneration` 加了 `to()` 覆写
（`modeling_qwen3_tts.py:1868`），搬设备时把 `speech_tokenizer.model` 一起带上
并同步 `st.device`。一处修好所有调用点，各处不用再各自打补丁。
冒烟阶段 8 已把原先的临时补丁换成**回归断言**（tokenizer 不在 NPU 上就报错），
防止以后退回 CPU decode。

开发机验证（CPU 加载 → `.to()`）：
```
加载后   主模型 cpu     / tokenizer cpu     st.device=cpu
.to() 后 主模型 cuda:0  / tokenizer cuda:0  st.device=cuda:0
```

**② 逐 token 动态形状触发反复编译。** 昇腾算子按形状编译，自回归生成每步
序列长度 +1，每步都是新形状 → 现场编译（CPU 编译、NPU 等喂）。
`ACLNN_CACHE_LIMIT`（`npu_env.sh` 已设）缓存后：首次 348s → 第二次 60s。

**这条改不掉，是架构特性。** 但它只影响**推理/批量合成**，
不影响训练（定长 batch、形状固定，编译一次复用）。等真到批量合成阶段再考虑
固定长度分档、批量生成摊薄、或合成回 GPU —— 现在不要优化。


## 9. 请 NPU 侧执行并回填

> **先只做 9.1 和 9.2。**
>
> 当前唯一目标是**让训练在 NPU 上跑起来**，路径就三步：
> `import qwen_tts 能过` → `模型能加载到 NPU` → `一个训练 step 能跑完`。
> 9.2 的阶段 0-6 跑通即达成，**跑通了就先停下推回来**。
>
> 9.3-9.6 是调优和定规模用的，等训练确实能跑之后再说，不要一次做到尾。

下面各项把结果填进「实测」列，连同 `npu_smoke_report.txt` 一起推回来。
任何一步失败就停下，把现象补进 §8 对应条目。

### 9.1 装完先验 import（对应 §8.1 / §8.2，两条阻塞项）

```bash
python -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__)"
python -c "import transformers; print(transformers.__version__)"
python -c "import qwen_tts; print('qwen_tts OK')"
```

| 项 | 预期 | 实测 |
|---|---|---|
| torch / torch_npu | 2.1.0 / 2.1.0.post8 | ✅ 2.1.0 / 2.1.0.post8.dev20241029 |
| transformers | ≥ 4.41（目标 4.57.3） | ✅ 4.55.2（4.57.3 实测崩，见 §8.2） |
| `import qwen_tts` | 无报错 | ✅ OK（python 3.9.10） |
| `torchaudio==2.1.0` 装上了吗 | 是 | ✅ 环境里已有 2.1.0 |

装不上 torchaudio 就走 §8.1 的兜底（25Hz tokenizer 改惰性 import）。

### 9.2 冒烟测试

```bash
cd finetuning/scripts && . ./npu_env.sh
python npu_smoke_test.py --model_path <VoiceDesign 目录>
```

| 阶段 | 要回填的数 | 实测 |
|---|---|---|
| 1 | bf16 matmul 耗时、显存总量 | ✅ 0.252s / 60.5 GiB 可用（总 61.0 GiB） |
| 2 | sdpa 耗时 / eager 耗时 | ✅ 0.001s / 0.004s（sdpa 快约 4×） |
| 4 | audio_codes shape | ✅ (32, 16) |
| 5 | 7 项是否全 PASS | ✅ 7/7 PASS |
| 6 | loss 首值 → 末值 | ✅ 3.7002 → 3.6943（6 步，步数少基本持平属正常） |
| 8 | 两条 instruct 的时长差 % | ✅ 10.5%（对机 2.72s / 对人 3.04s）；修复 decode 未上 NPU 后复测：首次 348s（含逐形状编译预热）/ 二次 60s，时长差 82.6%——采样随机性导致波动大，此判据是弱信号 |

**结论：阶段 0-8 全部通过 ✅（2026-08-15，910B1 单卡），训练链路在 NPU 上跑通。**
模型路径：`/home/ma-user/work/dataset/wsj-mimo-data/Qwen3-TTS-ckpt`。
完整日志见随本文档一起提交的 `finetuning/scripts/npu_smoke_report.txt`。

### 9.3 多卡梯度同步（对应 §8.6）—— *训练跑通之后再做*

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

### 9.4 真实训练路径下的 sdpa vs eager（对应 §8.4）—— *之后*

阶段 2 测的是 `is_causal=True` 的合成 benchmark，而训练走的是 4D mask 路径，
**结论未必迁移**。用真实训练各跑一次对比：

```bash
for A in sdpa eager; do
  ATTN=$A MAX_STEPS=20 OUTPUT_DIR=/tmp/attn_$A bash run_v2_npu_debug.sh
done
```

| ATTN | 20 步耗时 | 峰值显存（npu-smi） | 实测 |
|---|---|---|---|
| sdpa | | | 近似版（合成数据 20 步）119.8s |
| eager | | | 近似版（合成数据 20 步）119.1s |

**近似版已跑（2026-08-16，合成数据）**：20 步粒度 sdpa/eager 无差异（119.8s vs
119.1s，均含加载与编译预热）。但 20 步太短、编译占比大，分辨不出真实差异；
9.5 的 200 步 sdpa 稳态是 1.0s/step，等真实数据就绪后按上面命令补正式版对比
（正式版务必加 `SKIP_PREPARE=1 TRAIN_JSONL=<绝对路径>`，避免每次重抽码）。

### 9.5 收敛趋势（对应 §8.5）—— *之后*

默认 `--dtype fp32` 已规避 bf16 的下溢风险，但仍要确认 loss 是真降不是抖：

```bash
# 合成数据版必须带 --synthetic_count：训练脚本 --num_epochs 1，
# 默认 8 条合成样本一个 epoch 只有 8 步，--steps 200 根本到不了 200
python npu_smoke_test.py --model_path ... --stages 4-6 --steps 200 --synthetic_count 256
```

| 项 | 实测 |
|---|---|
| 200 步 loss 首 / 中 / 末 | ✅ 3.6260 → 1.1135（2026-08-16，合成 256 条；中值未采集，正式版日志里留） |
| 单步耗时 | ✅ 约 1.0s/step（200 步总耗时 192.8s 含加载编译） |

loss 从 3.6 降到 1.1，**确认在真降不是抖**——bf16 之外的收敛路径风险基本排除，
`--dtype fp32` 默认值可以放心用。有余力再跑一次 `--dtype bf16` 对比，能直接回答 §8.5。

### 9.6 定正式训练的规模 —— *之后*

上面几项跑完就能算了：

| 项 | 来自 | 实测 |
|---|---|---|
| 单步耗时（batch=2） | 9.5 | |
| 峰值显存 | 9.4 | |
| 可用的 batch_size | 由显存反推 | |
| 43.5w 条 1 epoch 预估耗时 | 单步耗时 × 步数 | |


## 10. 上机实测记录（2026-08-15，910B1 单卡）

§9.1/§9.2 之外，这次上机还踩出/确认了以下几条，均已在代码或文档中处理：

1. **模型下载走 curl，不走 huggingface-cli**：本机代理
   （proxy-notebook.modelarts.com:8083）对 Python requests 的 CONNECT 隧道返回
   503 Tunnel failed，但 curl 正常。且 VoiceDesign 仓库里
   `speech_tokenizer/speech_tokenizer/` 是内嵌子仓库，resolve 只返回
   「Entry not found」空指针——tokenizer 权重要从独立仓库
   `Qwen/Qwen3-TTS-Tokenizer-12Hz` 下载，摆平到 `<模型>/speech_tokenizer/` 下。
   本机模型已就位：`/home/ma-user/work/dataset/wsj-mimo-data/Qwen3-TTS-ckpt`
   （主模型 404 张量 / 1.917B / BF16；tokenizer 496 张量 / 170.6M / F32，
   均用 safetensors 头校验过）。

2. **`device_map` 的 meta 快速加载路径在 torch 2.1 + torch_npu 上不可用**：
   `from_pretrained(..., device_map="npu:0")` 走 low_cpu_mem_usage 的 meta 加载，
   `param[...]` 触发 `torch.cuda._lazy_init` →
   `AssertionError: Torch not compiled with CUDA enabled`（与 transformers 版本无关，
   是 torch 2.1 + torch_npu 的组合问题）。修法：**统一 CPU 加载后 `.to("npu:0")`**，
   已改冒烟 s3/s4/s8 与 `prepare_data.py`；训练脚本本来就不走 device_map，不受影响。
   集群和调试机同栈，不会复发。

3. **芯片型号是 910B1**：npu-smi 与 `torch.npu.get_device_name(0)` 都是
   `Ascend910B1`，NPU_ENV.md 与本文档原写 910B2，已全部更正。64GB 显存一致。

4. **pip 必须 `--user`**：base 镜像 conda 环境里有 root 所有的文件（tokenizers
   的 dist-info），普通 pip 卸载时报 `Permission denied: 'METADATA'`。user site
   优先级更高正好覆盖（§2 已写明）。

5. **阶段 8 推理慢的根因定位（两次实测对比）**：
   - 第一次跑：两条各 ~10/~20 分钟，CPU 92% / AICore 0%。两个原因叠加——
     ① `speech_tokenizer` 是**普通属性不是 nn.Module**，`.to("npu:0")` 不递归到
     它，decode（170M 模型逐帧前向）一直留在 CPU；② 生成循环逐 token 动态形状，
     每步触发 aclnn 内核编译（CPU 编译、NPU 等喂指令）。
   - 修掉 ① 后复测：**首次生成 348s → 第二次 60s**。差距即编译预热：
     第二次的形状全部命中 ACLNN_CACHE_LIMIT=1000 的编译缓存。NPU 确实在算
     （HBM 4→8GB、AICore 4-10% 波动、训练 step 正常）。
   - 残留的 60s 仍比 CUDA 慢不少（小算子 launch + 4D mask sdpa 路径，正是
     §8.4 的疑问），留待 9.4 复测；训练吞吐不受此影响（阶段 6 几步含编译几分钟）。
   - 另：两次实测时长差 10.5% vs 82.6%，`do_sample=True` 随机性大，
     阶段 8 的时长差判据只能当弱信号，腔调对不对最终以听辨为准。
   （sox CLI 缺失的告警无害——python sox 包可用，12Hz 链路不依赖 CLI。）

6. **真实数据现状（2026-08-16 对照 MiMo 仓库 data_report.md 实测）**：
   - 三份拒识打标 jsonl 与旧报告**完全一致**：行数 99,231 / 120,462 / 277,087
     （合计 496,780）、schema 25 字段、每集抽样 3000 条 wav 命中 100%、
     5月批次扁平池 `car_all_0415_0424_wavs` 753,287 个 wav 精确一致。
   - **报告之后新增**：`duplex_whj_data/CC_new_0601_0630_wavs`（147,190）与
     `CC_new_0701_0730_wavs`（179,939）两个抽取好的扁平池（抽样 100% 命中，
     建议作为 V2 数据管线的 wav 来源）；数据源扩到 ~10 个
     （car_all_0719_0731、70w_see_glass_car_0626×3、glass、seeworld、phone、
     mingchen_synthetic 等）；`processed_train_0805/` 下已有 1,466,798 条的
     拒识检测多轮训练集（Qwen-VL 格式，lineage 完整，音频路径是集群
     `/opt/huawei/dataset/...`）。
   - 本仓库 V2 管线的消费入口是 `finetuning/prepare_v2_data.py`
     （打标 jsonl + 扁平 wav 池 → `{audio, text, instruct}` + VAD 切静音），
     三份数据直接可用。建议先 `--limit 100` sanity check 再全量
     （全量 43.5 万条 VAD 预计数小时，`--skip-existing` 断点续跑；
     全量 trimmed wav 约 50-60GB）。


## 附：已知会失败的地方（已在代码里处理，列出来备查）

| 项 | 处理 |
|---|---|
| `flash_attention_2` 是 CUDA kernel | `--attn` 默认已改 `sdpa` |
| `prepare_data.py --device` 默认 `cuda:0` | NPU 上必须显式传 `npu:0`，已加 help 提示 |
| `modeling_qwen3_tts.py` 有 PEP 604 注解，py3.9 import 即崩 | 已加 `from __future__ import annotations` |
| 集群上自设 `ASCEND_RT_VISIBLE_DEVICES` 会和调度打架 | 集群脚本不设，交给 ModelArts |
| ckpt 保存后残留 `index.json` 会让新权重失效 | 保存前已删 |
| `device_map` 走 meta 加载路径，torch 2.1 + torch_npu 上必崩 | 统一 CPU 加载 + `.to("npu:0")`（见 §10.2） |
| 装包写不进 conda 环境（root 所有的 dist-info） | `pip install --user`（见 §2/§10.4） |

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

### 二轮复核（pull 到 e3d9f8f 后，2026-08-15）

`bb5926d`（--dtype fp32 默认）、`14c3e42`（V2TrainStep 封装 + review 其余）、
`e3d9f8f`（§8 未确认点）三个提交逐条验过，方向都对：V2TrainStep 让 autocast
与 DDP allreduce 同时生效（两者都挂在被 prepare 模块的 forward 上）；bf16 落盘
补丁括号/键名处理正确；模块级 torch_npu import 解决跳过阶段 0 的误报；阶段 5
前缀逐格断言的算术与 collate 吻合；§2 补的 sox/onnxruntime/einops/torchaudio
==2.1.0 与 import 链一致。又修了两处残留：

1. 27GiB 未改干净：run_v2_npu_debug.sh / run_v2_npu_cluster.sh 头注、smoke 阶段 1
   告警仍写 27GiB（与 14c3e42 自己的「统一 25GiB」矛盾），已统一为 25GiB。
2. §4 示例 `RAW_JSONL=./data_v2/train_raw.jsonl` 与脚本的 realpath -m 不匹配：
   相对路径按脚本启动目录解析，会指向 scripts/data_v2/ 而非 prepare_v2_data 的
   落盘位置。示例已改为用默认路径并注明覆盖需传绝对路径。

### 三轮对齐（对照参考仓库 duplex_huanyu_qwen35 脚本对，2026-08-15）

真跑是「提交一个脚本到集群」，对照 `pretrain_al_cls.sh`（及其 fb128 新版本）：
- 集群变量（VC_WORKER_HOSTS / MA_NUM_HOSTS / VC_TASK_INDEX）与路径布局
  （/opt/huawei/quoteModel、/opt/huawei/dataset）本就一致
- 已补齐：`HCCL_IF_BASE_PORT=64111`、`MASTER_PORT=6411` + netstat 端口检查、
  `npu-smi info`、脚本内 pip install（SKIP_INSTALL=1 可跳）、CANN source 后显式
  补 LD_LIBRARY_PATH/PATH
- 刻意保留差异：启动器用 torchrun 不用 msrun（不依赖 megatron/mindspore）；
  nnal/atb/set_env.sh 不 source（MindSpeed 专用算子库，HF 路线用不到）

### 四轮：上机实测 + 收尾修正（2026-08-15）

在 910B1 上完成了 §9.1/§9.2 全流程（0-8 全过，详见 §9 回填与 §10），过程中修掉
5 个只有上机才会暴露的问题，均已改代码并写回本文档：

1. transformers 4.57.3 在 torch 2.1.0 上 import 即崩（pytree API）→ 钉 4.55.2
   （§8.2 回填）；qwen_tts 的 `@check_model_inputs()` 与
   `from_pretrained(dtype=)` 两处加版本兼容（新旧 transformers 都通）
2. pip 写不进 conda 环境（root 所有的 dist-info）→ 安装统一 `--user`，
   两个启动脚本安装块同步更新
3. `device_map` meta 加载路径在 torch 2.1 + torch_npu 上必崩 →
   冒烟 s3/s4/s8 与 prepare_data.py 改为 CPU 加载 + `.to("npu:0")`
4. 芯片型号 910B2 → **910B1**（npu-smi 与 torch 实测），NPU_ENV.md 同步更正
5. 阶段 8 推理慢：定位为「speech_tokenizer 普通属性没被 .to() 搬到 NPU（decode 留
   CPU）」+「逐 token 动态形状 aclnn 编译预热」。前者已修（冒烟 s8，后由远程
   agent 升级为 modeling_qwen3_tts.py 的 to() 覆写，一处修好所有调用点），
   后者实测首次 348s / 二次 60s（缓存命中），记入 §10.5 待 9.4/9.5 复测

### 五轮：9.4/9.5 实测 + 真实数据核对（2026-08-16）

1. 冒烟脚本加 `--synthetic_count`（阶段 6 一个 epoch 只有 N 步，`--steps 200`
   配默认 8 条合成样本实际只跑 8 步）+ 阶段 6 计时（s/step）
2. 9.5 跑通：200 步 loss 3.63 → 1.11（确认收敛，fp32 默认可用），稳态约 1.0s/step
3. 9.4 近似版（合成数据 20 步）：sdpa 119.8s vs eager 119.1s 无差异，正式版
   待真实数据
4. 三份真实拒识打标数据与 MiMo 仓库 data_report.md 逐项核对：核心结论仍成立
   （行数/schema/wav 100% 在位），但新增了 CC 两个扁平 wav 池、~7 个新数据源、
   processed_train_08xx 多轮训练线——详情见 §10.6
