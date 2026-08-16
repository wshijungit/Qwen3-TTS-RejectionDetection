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
| **4** | 抽 `audio_codes`（直接调 tokenizer） | 音频 tokenizer 有算子不支持 → 这步放 CPU 跑 |
| **4b** | **走生产路径**：`prepare_v2_data.py` → `prepare_data.py`，含断点续跑 | 数据准备链路在 NPU 上不通 |
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
# 单卡 debug（默认路径见下，与参考仓库约定对齐）
cd finetuning/scripts
bash run_v2_npu_debug.sh

# ModelArts 集群（作业变量由平台注入）
bash run_v2_npu_cluster.sh
```

**路径约定（对齐 duplex_huanyu_qwen35 / MiMo v2 脚本）**：

| 用途 | debug 机 | 集群 |
|---|---|---|
| 模型 | `/home/ma-user/work/model/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | `/opt/huawei/quoteModel/...` |
| 数据 | `/home/ma-user/work/dataset/duplex_whj_data/v2/train_{raw,codes}.jsonl` | `/opt/huawei/dataset/...` |
| 输出 | `/home/ma-user/work/quoteModel/duplex_whj_exp/v2_single_text_turn_tts/debug` | `/opt/huawei/quoteModel/duplex_whj_exp/v2_single_text_turn_tts/${RUN_NAME:-v1}` |

覆盖一律传**绝对路径**：脚本里 `RAW_JSONL/TRAIN_JSONL` 会 realpath，相对路径按脚本
启动目录解析，容易漂。

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
| sdpa | | | 近似版（合成 20 步）119.8s；**正式版（真实 200 步）241.4s ≈ 1.2s/step** |
| eager | | | 近似版（合成 20 步）119.1s；**正式版（真实 200 步）248.5s ≈ 1.2s/step** |

**正式版已跑（2026-08-16，真实数据 256 条 × 200 步）**：sdpa 241.4s vs eager
248.5s，差异 ~3%，**4D mask 训练路径下 sdpa 没有显著加速、也没有负收益**——
§8.4 的疑问关闭，按默认 sdpa 继续即可（急着要基准时 eager 也可用，二者等价）。

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
   503 Tunnel failed，但 curl 正常。且各 TTS 仓库里
   `speech_tokenizer/speech_tokenizer/` 是内嵌子仓库，resolve 只返回
   「Entry not found」空指针——tokenizer 权重要从独立仓库
   `Qwen/Qwen3-TTS-Tokenizer-12Hz` 下载，摆平到 `<模型>/speech_tokenizer/` 下。
   本机四个模型已就位（均用 safetensors 头校验过，软链在
   `/home/ma-user/work/model/`）：

   | 模型 | 主模型 | tokenizer | 用途 |
   |---|---|---|---|
   | Qwen3-TTS-12Hz-1.7B-VoiceDesign | 404 张量 / 1.917B / BF16 | 496 / 170.6M / F32 | **V2 训练线** |
   | Qwen3-TTS-12Hz-1.7B-Base | speaker_encoder 76 张量，enc_dim=2048 | 同上 | spk 抽取（§17） |
   | Qwen3-TTS-12Hz-0.6B-Base | enc_dim=1024 | 同上 | ⚠️ 与 1.7B 的 2048 不匹配，**抽不了 V2 的 spk**（§17.6 回填） |
   | Qwen3-TTS-12Hz-1.7B-CustomVoice | 404 / 1.917B / BF16 | 同上 | speaker_id 音色克隆路线（暂无训练线） |

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
     全量 trimmed wav 约 50-60GB）。输出按上表路径约定：
     `--out-dir /home/ma-user/work/dataset/duplex_whj_data/v2`
     （train_raw/valid_raw 直接落在启动脚本的默认数据目录）。
     §11.1 sanity check 已通过（见 §11 回填）。

7. **numba JIT × libsndfile 加载 = 必崩（aarch64，2026-08-16 实测）**：
   `import soundfile` 之后首次触发 librosa 的 numba JIT（`effects.trim` →
   `util.frame`），llvmlite 的 RuntimeDyldELF 报 relocation overflow
   （`Assertion 'isInt<33>(Result)' failed`，进程 SIGABRT）。根因是 libsndfile
   的 .so 加载改变进程地址空间布局，LLVM JIT 代码段与全局符号距离超 4GB——
   与 fork/spawn、数据长短、主/子进程都无关（三种组合都实测复现）。
   修复：`prepare_v2_data.py` 的 energy VAD 换成**纯 numpy 实现**
   （frame 2048 / hop 512，语义与 librosa.effects.trim 一致）；worker 池改
   spawn 防 fork 继承的其它问题（全量约多花 12 分钟，可接受）。


## 11. 接真实数据（下一步）

冒烟与收敛都过了，接下来把真实拒识数据接进管线。**分三步，每步验完再往下。**

三个数据集现在都有抽取好的扁平 wav 池（§10.6 实测），正是
`prepare_v2_data.py` 期望的 `<wav_dir>/<uttid>.wav` 形式，直接可用。

### 11.1 先 100 条 sanity check

```bash
S=/home/ma-user/work/dataset/stc_data/dataset/cabin_duplex_data_artif
W=/home/ma-user/work/dataset/duplex_whj_data
cd finetuning

python prepare_v2_data.py \
  --dataset cc0601 $S/yibuapi_outputs/CC_new_0601_0630/setup1/gemini-3.5-flash/setup1_exp05.jsonl  $W/CC_new_0601_0630_wavs \
  --dataset cc0701 $S/yibuapi_outputs/CC_new_0701_0730/setup1/gemini-3.5-flash/setup1_exp05.jsonl  $W/CC_new_0701_0730_wavs \
  --dataset car05  $S/yibuapi_outputs/car_all_0415_0424/setup1/gemini-3.5-flash/setup1_exp05.jsonl $W/car_all_0415_0424_wavs \
  --out-dir ./data_v2_smoke --wav-out-dir /tmp/v2_wav_smoke \
  --limit 100 --workers 16 --val-size 10
```

（集群对应 `/opt/huawei/dataset/...`，只换前缀。）

**要看的（脚本会自己打印）**：

| 项 | 关注什么 |
|---|---|
| 各数据集 kept / 各类丢弃数 | `wav_not_found` 应为 0（池子命中 100%） |
| 一致性筛选前后 reject 占比 | 预期从约 29/35/49% 降 5-6 个百分点 |
| VAD 前后时长、切掉比例 | 切太多说明 `--top-db` 偏激进 |
| **类别偏斜告警** | 有 ⚠️ 就停下，别全量跑 |

再抽几条 `train_raw.jsonl` 看 instruct 转写通不通顺——**这是第一次在真实
gemini 数据上跑转写**，之前都是合成样本。

**11.1 实测（2026-08-16）**：三集各 100 条全过（音频处理 100/100，`wav_not_found`=0）。
reject 占比筛选前后 19.8→12.0% / 33.9→29.0% / 48.6→47.0%（car05 降得少，limit 100
样本小，全量再看）；VAD 切掉 48-63%（如 6.09s→2.24s），偏多——样本太小不下结论，
全量统计时重点看这条，切太多就把 `--top-db` 上调。instruct 转写抽检 6 条通顺
（能纠正 ASR 错字、结论词已剥净、无「应予拒识」尾巴）。train 290 / validate 10，
accept 70.7% / reject 29.3%，无偏斜告警。

**踩坑记录（§10.7）**：首跑 100/100 全崩——`import soundfile`（加载 libsndfile）
之后再触发 librosa 的 numba JIT，本机 aarch64 上 llvmlite 必报 relocation
overflow（SIGABRT）。已把 energy VAD 换成纯 numpy 实现，worker 池改 spawn。

### 11.2 小批量真实数据训一次

```bash
MDL=/home/ma-user/work/model/Qwen3-TTS-12Hz-1.7B-VoiceDesign
head -256 data_v2_smoke/train_raw.jsonl > /tmp/real256.jsonl
. scripts/npu_env.sh        # 必须 source：不 source 时系统 CANN 8.1 下 TBE 编译子进程崩
                            # （"TBE Subprocess raise error, main process disappeared"，
                            # 实测 batch 大小无关，8/32 都复现；8.5.0 环境无此问题）
python prepare_data.py --device npu:0 \
  --tokenizer_model_path $MDL/speech_tokenizer \
  --input_jsonl /tmp/real256.jsonl --output_jsonl /tmp/real256_codes.jsonl

cd scripts
python npu_smoke_test.py --model_path $MDL \
  --codes_jsonl /tmp/real256_codes.jsonl --stages 5-8 --steps 200
```

真实数据的 instruct 比合成的长得多（gemini 的 evidence + reason 几十字），
**序列会长不少**——阶段 5 的布局断言和显存都要重新确认。

顺带这一步也是 **9.4 的正式版**：真实数据下再比一次 `--attn sdpa` / `eager`。

**11.2 实测（2026-08-16）**：256 条真实数据全链路通过——抽码成功（source
npu_env.sh 后 TBE 崩溃消失，见命令注释）、阶段 5 七项全 PASS（真实 instruct
长序列下布局断言依然成立）、200 步 loss 3.3253 → 3.0212、**约 1.2s/step**
（与合成数据 1.0s 接近，batch=1 的瓶颈不在序列长度）、阶段 8 时长差 34.6%
（对机 1.36s / 对人 2.08s），instruct 通路生效。真实数据 smoke 报告见
`npu_smoke_report_real256.txt`。

### 11.3 全量（参数已由实测定死）

所有前置都验完了：冒烟 0-8 含 4b 在 NPU 通过、数据准备链路（含断点续跑）通过、
分桶与 batch 都有实测（§12.6）、VAD 的「过切」疑虑已被量化否定（§14.2）。

#### 结论先行：用 batch=8

§12.6 实测 **batch 1→8 时 s/step 几乎不涨、吞吐涨约 8 倍**（0.4 → 3.2 样本/秒），
与 H200 的 7.7× 一致 —— 说明单步时间几乎全是固定开销，**batch 是唯一有效的杠杆**。

| 配置 | 吞吐 | 43.5w 条 1 epoch |
|---|---|---|
| batch=1（最初） | 0.4 样本/秒 | ~145 小时 ❌ |
| **batch=8 单卡** | **3.2 样本/秒** | **~19-21 小时** ✅ |
| batch=8 × 8 卡 DDP | 约 25 样本/秒 | ~2.5-3 小时 |

显存：fp32 静态约 25GiB / 61GiB，激活占比极小（H200 上 batch 1→8 显存不变），
**batch 还能往上试**。真跑 OOM 再降。

#### 第一步：全量数据准备

```bash
S=/home/ma-user/work/dataset/stc_data/dataset/cabin_duplex_data_artif
W=/home/ma-user/work/dataset/duplex_whj_data
cd finetuning

python prepare_v2_data.py \
  --dataset cc0601 $S/yibuapi_outputs/CC_new_0601_0630/setup1/gemini-3.5-flash/setup1_exp05.jsonl  $W/CC_new_0601_0630_wavs \
  --dataset cc0701 $S/yibuapi_outputs/CC_new_0701_0730/setup1/gemini-3.5-flash/setup1_exp05.jsonl  $W/CC_new_0701_0730_wavs \
  --dataset car05  $S/yibuapi_outputs/car_all_0415_0424/setup1/gemini-3.5-flash/setup1_exp05.jsonl $W/car_all_0415_0424_wavs \
  --out-dir $W/v2 --wav-out-dir $W/v2/wav \
  --workers 32 --skip-existing
```

> `$S` / `$W` 与 §11.1 同一套（打标 jsonl 在 `$S/yibuapi_outputs/...`，
> wav 池在 `$W/..._wavs`）。**此处的路径必须与 §11.1 实测通过的完全一致** ——
> 早先版本这里写的是 `$D/CC_new_0601_0630.jsonl` 之类的裸文件名，
> 指南里没有任何步骤生成那些文件，照抄会 FileNotFound。

VAD 数小时，trimmed wav 约 50-60GB。`--skip-existing` 可断点续跑（VAD 参数指纹
会挡住"改了参数还复用旧产物"）。**`--ref-percentile` 保持默认 100**（§14.2）。

跑完看报告：`wav_not_found` 应为 0、类别偏斜无 ⚠️、
一致性筛选后 reject 占比比筛前低 5-6 个百分点（预期内）。

#### 第二步：全量抽码

```bash
python prepare_data.py --device npu:0 \
  --tokenizer_model_path /home/ma-user/work/model/Qwen3-TTS-12Hz-1.7B-VoiceDesign/speech_tokenizer \
  --input_jsonl $W/v2/train_raw.jsonl --output_jsonl $W/v2/train_codes.jsonl
```

**这一步现在支持断点续跑**（逐 batch 追加写，重启按已写行数跳过），
崩了直接重跑同一条命令即可。会每 50 个 batch 打一次进度。

> 已知：batch encode 不具备 padding 不变性，续跑会改变 batch 分组，个别处在
> VQ 边界上的帧会翻到邻近码字。对音质无影响，但 codes 不逐位可复现。

#### 第三步：正式训练

```bash
cd scripts
BATCH_SIZE=8 GRAD_ACCUM=1 EPOCHS=1 \
DTYPE=fp32 ATTN=sdpa \
bash run_v2_npu_debug.sh
```

启动脚本已透传 `SAVE_EVERY`（默认 **500** 步）与 `LENGTH_BUCKET`（默认 64）。
19-21 小时的 epoch 中途崩掉，只在 epoch 末落盘等于全丢，所以默认就开着。

**磁盘账**（曾因此把 /tmp 撑爆过一次）：batch=8 时一个 epoch 约 5.4 万步，
`SAVE_EVERY=500` 就是 108 个 checkpoint。故加了两道：

- `--save-total-limit`（默认 **2**）：只保留最近 2 个 step ckpt，epoch ckpt 不受限
- 保存时 `copytree` 跳过 `*.safetensors` —— 原先每次白搬 3.4GB 权重再被覆盖

**约 470GB → 约 13GB**（2 个 step ckpt + epoch ckpt，每个约 4.3GB，
含子目录里 682MB 的 speech_tokenizer）。

多卡的话先跑一次 `check_ddp_sync.py`（§9.3）确认梯度真的在同步 —— 那个检查在
NPU 的 HCCL 路径上还没验过。

#### 中途崩了怎么接着跑

数据准备和抽码都有断点续跑（重跑同一条命令即可）。**训练本体没有 `--resume`**，
唯一方式是拿最近的 step checkpoint 当新的初始权重重启：

```bash
E=/home/ma-user/work/quoteModel/duplex_whj_exp/v2_single_text_turn_tts
ls $E/debug          # 先看最近的是哪个 checkpoint-stepNNNN

cd scripts
MODEL_PATH=$E/debug/checkpoint-step54000 \
OUTPUT_DIR=$E/debug_resume1 \
SKIP_PREPARE=1 BATCH_SIZE=8 GRAD_ACCUM=1 EPOCHS=1 \
bash run_v2_npu_debug.sh
```

`$E/debug` 就是第三步不指定 `OUTPUT_DIR` 时的默认落盘位置（见
`run_v2_npu_debug.sh`）。**`OUTPUT_DIR` 要换一个新目录**：热启是权重级的，
`gstep` 从 0 重数，写回原目录的话新旧 checkpoint 的 step 号会混在一起。

这是**权重级热启**：optimizer 状态丢失、数据重新 shuffle，但权重接上了。
对单 epoch 的全量训练够用。再崩一次就 `debug_resume2`，依次往下接。

> ckpt 必须完整才能这么用。早先版本的 `_save` 用 `ignore_patterns` 跳过权重时
> **递归误伤了 `speech_tokenizer/model.safetensors`**（682MB），产出的 ckpt
> 加载即 `OSError`——`--save-every` 存的盘全是废的、崩了也接不上。已修，
> 冒烟阶段 7 加了该文件的存在性断言。

#### 训练中要盯的

| 项 | 正常 | 异常时 |
|---|---|---|
| s/step | 1.2-1.4s（batch=8 稳态） | 明显更高 → 看是不是形状没收敛（§12） |
| 磁盘 | 约 13GB（2 个 step ckpt + epoch ckpt，每个约 4.3GB 含 speech_tokenizer） | 涨个不停 → `--save-total-limit` 没生效 |
| loss 趋势 | **比前 20 步与后 20 步的均值**，别看首末单点 | 几百步不降 → 见 §8.5 |
| 显存 | 约 25GiB + 激活 | OOM → 降 batch |


## 12. 提速：1.2s/step 的主因是**动态形状反复编译**（已加分桶，待实测）

NPU 实测 batch=1 时 1.2s/step。1.7B 模型在这个序列长度（约 100-300 token）下的
真实计算量远小于此，说明绝大部分时间不在算。

### 12.1 根因

昇腾的算子**按形状编译**（aclnn）。而 `collate_fn` 原本按 batch 内最长样本
padding，**每个 batch 的序列长度都不同** —— 开发机实测：12 条真实样本产生
**12 种不同形状**，等于每一步都在现场编译内核（CPU 编译、NPU 等喂）。

这和推理慢是**同一个机理**。我此前判断"训练定长、不受影响"是错的。

### 12.2 已做：长度分桶

序列长度向上取整到 `--length-bucket`（默认 **64**）的倍数：

```
bucket=1  （原行为）: 12 种形状 [73, 77, 83, 84, 87, 95, 96, 98, 103, 107, 108, 110]
bucket=64 （新默认）:  1 种形状 [128]
```

代价是多算一些 padding（平均多 32 格，相对 100 量级的序列约 30%），
但省下的编译时间应远大于此。CUDA 上动态形状原生支持、无此问题，设 `1` 即关闭。

**这条只在 NPU 上验证才有意义**，请在 §12.4 实测。

### 12.3 其它可能的提速点（按预期收益排序，均未验证）

| # | 手段 | 预期 | 备注 |
|---|---|---|---|
| 1 | **batch 调大** | 最直接 | 显存余量很大（fp32 用 25GiB / 61GiB）。step 时间不会线性增长，单样本吞吐显著改善 |
| 2 | 长度分桶（§12.2） | 已做 | 与 batch 调大叠加：同 batch 内长度接近时 padding 浪费更少 |
| 3 | 按长度分组采样 | 中 | 让同 batch 的样本长度接近，减少 padding。需改 sampler |
| 4 | `--dtype bf16` | 低 | autocast 已让矩阵乘走 bf16，fp32 的代价主要在显存和优化器步，不在算力 |
| 5 | `grad_accum` 调大 | 无 | 不减少 forward/backward 次数，只影响优化器步频率 |

**先做 1 和 2，测出数再考虑 3。**

### 12.4 开发机（H200 / CUDA）的 batch 缩放实测

绝对数字不能迁移到昇腾（架构不同、且这边没有编译开销），但**缩放形状**有参考价值：

| 配置 | 单步耗时 | 样本/秒 | 峰值显存 |
|---|---|---|---|
| bs=1, bucket=64 | 136 ms | 7.3 | 35.8 GiB |
| bs=2, bucket=64 | 137 ms | 14.6 | 35.8 GiB |
| bs=4, bucket=64 | 136 ms | 29.5 | 35.8 GiB |
| bs=8, bucket=64 | **142 ms** | **56.2** | 35.8 GiB |
| bs=4, bucket=1 | 137 ms | 29.1 | 35.8 GiB |

**batch 从 1 涨到 8，单步耗时几乎不变（136→142ms），吞吐线性涨 7.7 倍。**

说明这个规模下单步时间**几乎全是固定开销**（算子下发、优化器步），
1.7B 在约 100 token 序列上的真实计算量小到可忽略。
昇腾上还额外叠了编译开销，固定成本占比只会更高 —— **batch 调大是最高优先级**。

顺带：bucket=1 与 64 在 CUDA 上无差异（动态形状原生支持），符合预期；
显存不随 batch 变化，因为 35.8GiB 几乎全是 fp32 权重+梯度+Adam 状态，
激活占比极小 —— 这也意味着 **batch 还能往上加很多**。

### 12.5 batch>1 的正确性（已验）

padding 区的 `attention_mask=0` / `codec_0_labels=-100` /
两个 embedding mask 均 False，**样本间无串扰**，bs=1/2/4 × bucket=1/64 全过。

但注意 padding 浪费：真实序列约 82-86，bucket=64 会补到 128，**浪费约 36%**。
若昇腾实测表明分桶收益不明显，可试 `--length-bucket 32`（补到 96，浪费降到 ~13%），
或按实际长度分布选桶宽。

### 12.6 请在 NPU 实测（跑完把数填进来）

```bash
cd finetuning/scripts && . ./npu_env.sh
# 分桶开关对比
for BK in 1 64; do
  python npu_smoke_test.py --model_path <VoiceDesign> \
    --codes_jsonl <real_codes.jsonl> --stages 6 --steps 50 \
    2>&1 | tee /tmp/bk_$BK.log
done
```

（`--length-bucket` 目前只在 `sft_12hz_voicedesign.py` 上，冒烟脚本用默认值；
要对比就直接调训练脚本传 `--length-bucket 1` / `64`。）

| 配置 | 50 步耗时 | s/step | 样本/秒 | 峰值显存 | 实测 |
|---|---|---|---|---|---|
| bucket=1,  batch=1 | 146.7s | 2.9s | 0.3 | | ✅ 2026-08-16 实测 |
| bucket=64, batch=1 | 129.7s | 2.6s | 0.4 | | ✅ 同上 |
| bucket=64, batch=4 | 148.7s | 3.0s | 1.3 | | ✅ 同上 |
| bucket=64, batch=8 | 123.8s | 3.9s | 3.2 | | ✅ 同上（256 样本一 epoch 只 32 步） |
| bucket=32, batch=8 | | | | | 未测，收益预期低于 1 vs 64 的差 |

**NPU 实测结论（2026-08-16）**：① bucket 1 vs 64 只差 ~13%（2.9→2.6 s/step）——
VAD 切静音后真实样本 codec 长度本就集中在 25-40 帧，形状种类少，编译节省有限；
原始未切音频的分桶收益会更大。② batch 1→8 **s/step 几乎不涨**（2.7→3.9 含
batch=8 只跑 32 步的摊销误差），吞吐 0.4→3.2 样本/秒 ≈ 8×——与 H200 的 7.7×
一致，**batch 放心往大调**。这些数字含加载/首编译（约 60-90s），稳态约
1.2-1.4s/step（§11.2 的 200 步实测）。

由此定正式训练的 batch，并重算 43.5w 条 1 epoch 的耗时。


## 13. 开发机侧两轮 review 的结论（2026-08-16）

对 `prepare_v2_data.py` / `dataset_voicedesign.py` / `sft_12hz_voicedesign.py` /
modeling 三处改动 / 冒烟与启动脚本做了两轮独立审查，12 个问题全部修完。
**下面几条与上机直接相关**：

### 13.1 装机脚本里有个从未被执行过的 bug（已修）

两个启动脚本的 `pip install -e "$FT_DIR"` 指向 `finetuning/`，而 `pyproject.toml`
在**仓库根**。脚本带 `set -euo pipefail`，装依赖阶段直接终止，训练一步跑不到。

之所以之前没暴露：上机时是照本指南 §2 手工装的（指南写的是仓库根，没错），
**脚本里这条从来没真正跑过**。现已改为 `REPO_ROOT`。

### 13.2 两处"证据会静默消失"（已修）

- **`--skip-existing` 续跑会丢掉 `db_range`**：复用产物时不重跑 VAD，该样本永远
  不会被标成"无动态"。全量跑一半崩了再续跑，`suspicious_vad.jsonl` 就系统性
  缺失一半内容且无提示。现已在复用分支补算。
- **`check_ddp_sync.py` 单样本假通过**：jsonl 只有 1 条时各 rank 内容相同，
  梯度天然一致，即使 allreduce 根本没发生也打 ✅。现会先校验各 rank 喂入是否
  真的不同，相同则拒绝下结论。

### 13.3 序列布局已 bitwise 确认

review 用 monkeypatch 捕获推理 `generate` 的 prefill embedding，与训练侧
`collate + V2TrainStep` 构造的做逐位对比：**Chinese(nb=5) 与 Auto(nb=4) 两种模式
下，有效区语义完全对齐（prefill 长度逐样本相等、padding 区 embedding 严格全零），
数值差 ≤ 1e-6**。

> ⚠️ 早先版本这里写的是「max|diff| = 0（bitwise 一致）」，**不对**。复跑实测是
> 2.4e-7 ~ 9.5e-7，GPU 和 CPU 上都非零。根因是 GEMM 的形状舍入：同一个 pad token
> 过 `text_projection`，序列长 1 与 128 的结果就差 2.4e-7（训练侧整条序列一次投影，
> 推理侧分段投影，矩阵形状不同）。作为参照，bf16 的机器精度约 7.8e-3，比这个误差
> 大 4 个数量级 —— **训练精度下毫无意义**。
> 记在这里是为了让下一个复跑这项检查的人别把 1e-6 当回归误报。
> 分桶不影响该结论：bucket=1 与 64 的 diff 逐样本完全相同。

同时确认：404 个参数全部拿到非零梯度（无 DDP unused-parameter 隐患）、
sub-talker 的 hidden 位置与推理 `past_hidden` 语义一致、
`text_projection` 路径与推理 bitwise 一致。

### 13.4 一条要知道的边界

`to()` 覆写只在**直接调 `model.to()`** 时生效。若该模型作为子模块被外层
`nn.Module.to()` 搬运，PyTorch 的 `_apply` 递归**不会**走子模块的 `to()`，
`speech_tokenizer` 仍会被漏掉。当前训练路径不用 decode、冒烟 s8 是直接调
`m.model.to()`，都没问题；写新的推理/合成脚本时留意。

---

## 14. 下一轮请跑什么

按顺序，每步完了把数填回对应小节：

| # | 做什么 | 看什么 | 对应小节 |
|---|---|---|---|
| 1 | 重跑 §11.1 的 100 条 | **保留比例分位** + **疑似清单**（过切/无动态各多少） | §11.1 |
| 2 | 抽听 `suspicious_vad.jsonl` 前 10 条 | 是不是强瞬态锁死了参考值 | §8.8 |
| 3 | 分桶开关对比（bucket 1 vs 64） | s/step 差多少 | §12.4 |
| 4 | batch 1/4/8 对比 | s/step 与峰值显存 | §12.4 |
| 5 | 由 3、4 定 batch，重算全量耗时 | | §9.6 |

**第 1、2 步优先**——`--ref-percentile` 要不要动、能不能放心跑全量，全看那两个数。
第 3 步是这轮代码改动里唯一没在 NPU 上验过的（分桶只在 CUDA 上确认了形状收敛
12 种 → 1 种，实际提速多少只有昇腾能测）。

**§14 实测回填（2026-08-16，NPU 910B1）**：

1. ✅ 重跑 100 条：保留比例 p50 仅 0.30/0.50/0.54（ref=100），疑似过切
   （保留<25%）39/19/21 条 ≈ **26%**。抽听清单见下方。
2. **前 10 条逐条量化分析后，瞬态锁死假说被否**：重算每个可疑样本的 VAD 边界
   （ref=95），被切部分的最高帧能量比保留段峰值低 **36-65 dB**（如 -23.8dB 保留
   vs -60.5dB 被切）——被切的全是近底噪静音，没有一条符合「瞬态锁死」特征
   （那应是被切段含接近保留段水平的人声）。「保留<25%」的主因是**车内录音
   turn 边界静音本来就长**（data_report：turn 中位 5.2s），VAD 切得对。
   `db_range` 大只是因为「底噪 + 正常说话」跨 60+ dB，不是瞬态证据。
   **结论：VAD 行为正常，`--ref-percentile` 保持默认 100 即可，不必改。**
   人耳抽听降级为可选（听 2-3 条确认保留段是完整句子即可）。
3/4. 见 §12.6 回填（本机实测进行中）。
5. 由 §12.6 定。


## 15. 冒烟测试的覆盖缺口（已补，2026-08-16）

之前的冒烟（阶段 0-8）**没有覆盖改动最多的那部分**：阶段 4 是自己造 wav 直接调
tokenizer，**绕开了 `prepare_v2_data.py` 和 `prepare_data.py` 这两个脚本**。
而 §11.1 是手工跑的、不在冒烟里 —— 也就是说整条数据准备链路改完之后，
在昇腾上一次都没再跑过。

补了 **阶段 4b**：造 12 条带真实 schema 的打标 jsonl + wav（含首尾静音、
一条标签不一致、一条纯底噪），走完整生产路径并校验：

- `prepare_v2_data.py` 跑通，标签不一致的被筛掉、产出字段是
  `{audio, text, instruct}` 且**无 `ref_audio`**、`suspicious_vad.jsonl` 生成
- `prepare_data.py` 抽码跑通
- **断点续跑**：截断一半后重跑能正确补齐

开发机（CUDA 代替 npu 设备）实测通过：12 → 11 条（筛掉 1 条标签不一致），
截断到 5 → 补齐 10。

同时给阶段 6 加了 `--batch` / `--bucket` / `--save-every`，
这三样此前也从未被冒烟覆盖（batch 写死 1、bucket 只用默认、save-every 全无）。

### 15.1 现在能一条命令跑完的

```bash
cd finetuning/scripts && . ./npu_env.sh
python npu_smoke_test.py --model_path <VoiceDesign>          # 0-8 含 4b
```

### 15.2 §12.6 的提速对比现在也能用冒烟跑

```bash
for BK in 1 64; do
  python npu_smoke_test.py --model_path <VoiceDesign> --codes_jsonl <codes> \
    --stages 6 --steps 50 --bucket $BK --report /tmp/bk_$BK.txt
done
for BS in 1 4 8; do
  python npu_smoke_test.py --model_path <VoiceDesign> --codes_jsonl <codes> \
    --stages 6 --steps 50 --batch $BS --report /tmp/bs_$BS.txt
done
```

阶段 6 现在会打「N 样本/秒」，直接填 §12.6 的表。

### 15.3 仍未被冒烟覆盖的

| 项 | 为什么 | 怎么办 |
|---|---|---|
| 多卡 DDP 梯度同步 | 需要 torchrun 多进程，不适合塞进单进程冒烟 | 单独跑 `check_ddp_sync.py`（§9.3） |
| 真实 43.5w 条的规模行为 | 冒烟是小样本 | §11 分三步做 |
| 批量合成（推理侧）的吞吐 | V2 训练完才需要 | 那时再说 |


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

### 六轮：真实数据管线跑通（2026-08-16）

1. §11.1 sanity check 通过：三集各 100 条，reject 占比筛选前后符合预期，
   instruct 转写抽检通顺；VAD 切掉 48-63% 偏多，全量时重点看
2. 修两个 worker 崩溃：numba JIT × libsndfile 加载（aarch64 必崩，energy VAD
   改纯 numpy 实现，§10.7）；TBE 子进程崩溃（prepare_data.py 必须 source
   npu_env.sh 的 CANN 8.5.0，系统 8.1 下必崩，§11.2 命令已注明）
3. §11.2 通过：256 条真实数据抽码 + 阶段 5 七项全 PASS + 200 步
   loss 3.33→3.02（1.2s/step）+ 阶段 8 时长差 34.6%
4. §9.4 正式版结论：真实数据 200 步 sdpa 241.4s vs eager 248.5s 无差异，
   §8.4 关闭，按默认 sdpa 继续

### 七轮：review 后的新冒烟跑通 + 两个远端 bug（2026-08-16）

1. §15.1 全量冒烟（0-8 含新阶段 4b）在 NPU 实测通过——但**首跑抓到远端两个 bug**：
   - `s6_train` 里 `dt` 未定义（NameError，训练成功后被炸停，7/8 没跑到）→ 改用 `wall`
   - 阶段 4b **没有接进 main()**（--skip-pipeline/--use-pipeline-codes 参数存在但
     从未调用 s4b_pipeline，文档说「0-8 含 4b」实际不含）→ 已接线：默认随阶段 4
     一起跑，`--use-pipeline-codes` 让后续阶段用 4b 产物
2. §14.1/14.2 回填：过切 26%（ref=100）→ 21%（ref=95），抽听清单已给出
3. §12.6 回填：bucket 1 vs 64 差 ~13%（VAD 后长度已聚拢，收益有限）；
   batch 1→8 s/step 几乎不涨、吞吐 8×，batch 放心往大调
4. 环境事故：/tmp 被冒烟工作目录（每次 4.3GB ckpt）塞满导致任务 ENOSPC，
   已清理旧工作目录并恢复（后续冒烟默认 workdir 在 /tmp，注意定期清）

### 八轮 / 九轮：两轮 fable review 的修复（2026-08-16，均在开发机 H200 实测）

`fd6b5cf..7e22ea1`。两轮都是**修复本身出问题**，列出来是因为同类失败反复发生。

1. **我上一轮的 `copytree` 修复把 ckpt 弄成不可加载**：`ignore_patterns("*.safetensors")`
   **递归**作用于每层子目录，漏掉 `speech_tokenizer/model.safetensors`（682MB），
   而 `from_pretrained` 强制从 `<ckpt>/speech_tokenizer` 加载 → 加载即 OSError。
   `--save-every` 存的盘全是废的，恰好抵消了它存在的意义。改为只跳顶层。
2. **`--use-pipeline-codes` 上一轮声称修了但根本没落地**——本项目**第五次**
   「字符串替换静默失败」。此后改用逐处 Edit，不再脚本批量替换。
3. **全量内存**：43.5w 条 audio_codes 存 Python list ≈ 9.8GB/进程，8 卡 DDP ≈ 78GB。
   压成 numpy int16 后 **78GB → 10.5GB**（collate 输出逐位不变，已核）。
4. **热启后 `--save-every` 立即失效**：滚动清理按 step 序号排，而热启 `gstep` 从 0
   重数、目录里还留着 `step54000`——新存的 `step500` 永远最小，**存完当场被自己删掉**。
   改按落盘 mtime 排。另加 `src==dst` 守卫（否则 gstep 撞回同序号时 `SameFileError`
   会在第 19 小时崩掉训练）。
5. **§11.3 的路径两轮都是改一半**：先是 `$D/xxx.jsonl` 裸文件名凭空捏造，
   再是只改齐 `$S` 之下的相对路径、`$S` 本身没对齐。热启命令的 `MODEL_PATH`
   也指向一个任何步骤都不产生的目录。均已按 §11.1 实测通过的路径统一。
6. `prepare_data.py`：`assert` 改 `raise`（`python -O` 会剥掉 assert，
   且报错现在带上本批文件名）；续跑不再把整个输出文件 `read()` 进内存。

**教训**：六条修复里有一条引入高危回归、一条根本没生效，共同点是**没有配一个
「修前必失败」的验证**。九轮起每条修复都附反证用例（如「反证旧排序确实删掉新 ckpt」）。

### 十轮（NPU 侧）：§16 任务 0/1 完成 + 四个模型就位（2026-08-16）

1. §17.6 四条全部回填：sessionid 有（无说话人标识）；**0.6B-Base 的
   enc_dim=1024 与 V2 的 2048 不匹配不可用**；已下载 1.7B-Base
   （speaker_encoder 76 张量、enc_dim=2048）；$S 前缀确认正确
2. 带 spk 冒烟 0-8 全过（§16 #1 回填）：4b 的 spk 与 jsonl 断点续跑同步对齐、
   NPU/CPU 抽取余弦 1.0（max|Δ|=4.08e-04）、阶段 5 十三项全 PASS、
   含 spk 训练 6 步正常、阶段 8 时长差 53.3%
3. 模型清单与下载法补进 §10.1（四个模型 + 各自主模型/tokenizer 校验值，
   0.6B-Base 的不匹配明确标注，防止后人拿错）

## 16. NPU 侧下一轮请跑什么（§14 的表已全部回填完毕）

§14 那张表五项全部有结论了，**不要再按它跑**。现在只剩两件：

| # | 做什么 | 前置 | 卡在哪就停 |
|---|---|---|---|
| 0 | 回答 §17.6 的两个数据问题（session 字段、Base 权重路径） | — | 没有 Base 权重就抽不了 spk |
| 1 | 冒烟带 spk 跑一遍：`--spk_model_path <Base> --use-pipeline-codes` | 问题 2 有答案 | 阶段 5 任一 spk 项 FAIL |
| 2 | **全量 43.5w 条三步**：§11.3 数据准备 → 抽码（带 `SPK_FILE`）→ 训练 | 开跑前先 `ls $S` 确认路径存在（见下） | 第一步报告里 `wav_not_found` ≠ 0 |
| 3 | 多卡才需要：`check_ddp_sync.py`（§9.3） | 单卡跑全量的话可跳过 | 打 ❌ 就别跑多卡 |

**2026-08-16 NPU 实测回填**：
- ✅ #0：§17.6 四条已全部回填（sessionid 有、1.7B-Base 已下载、$S 前缀确认）
- ✅ #1：**带 spk 冒烟 0-8 全过**（`npu_smoke_report_spk.txt` 已随提交）——阶段 4b
  的 spk 随 jsonl 断点续跑同步对齐（行边界/半行两种截断均验证）、NPU vs CPU
  抽取余弦 1.0（最大绝对差 4.08e-04）、两次抽取逐位一致；阶段 5 十三项全 PASS
  （块宽 6、spk 槽位位置、spk_pos、mask、逐位一致、非全零）；阶段 6 含 spk 训练
  6 步正常（(10, 2048) 向量加载）；阶段 8 时长差 53.3% instruct 生效。
- 剩 #2（全量三步）与 #3（多卡才需要）。

### 16.1 spk 该什么时候抽（推荐顺序）

spk 有两种抽法，**产出逐位相同**（已验）：

| 方式 | 命令 | 什么时候用 |
|---|---|---|
| 跟 codes 同一遍 | `SPK_FILE=... bash run_v2_npu_debug.sh` | 从头跑，省一次 wav 读取 |
| **独立抽** | `prepare_data.py --spk-only --device npu:0 --input_jsonl <codes.jsonl> --spk_out ... --spk_model_path <1.7B-Base>` | codes 已经抽完了；或想重抽 spk 而不动 codes |

独立模式的对齐由 **codes.jsonl 自身的行顺序**保证（第 i 行的 audio 抽出第 i 行
向量），天然不可能错位；带断点续跑，不需要 tokenizer、也不占 codes 那份显存。

**推荐这么排**（关键是把唯一没验的假设提前到 5 分钟就能试出来）：

```
1. §11.3 第一步  全量数据准备（VAD 等）          数小时
2. §11.3 第二步  全量抽 codes                    数小时
3. 独立抽 spk    --spk-only（SPK_MEL_THREADS 用冒烟给的值）   约 0.6h
4. ★ 小批量验 spk：拿前 256 条 + 对应 spk 跑 200 步再听        约 5 分钟
5. §11.3 第三步  全量训练                        19-21h
```

第 4 步是重点。**「Base 的 embedding 空间与 VoiceDesign 的 talker 是否兼容」
是当前唯一没验证的假设**（§17.5 末尾），而它只能靠训练来验。放在这里花 5 分钟，
比训完 19-21 小时才发现音色那一格没学到要划算得多：

```bash
head -256 $W/v2/train_codes.jsonl > /tmp/real256_codes.jsonl
python prepare_data.py --spk-only --device npu:0   --input_jsonl /tmp/real256_codes.jsonl --spk_out /tmp/real256_spk.f16   --spk_model_path /home/ma-user/work/model/Qwen3-TTS-12Hz-1.7B-Base
cd scripts && python npu_smoke_test.py --model_path $MDL   --codes_jsonl /tmp/real256_codes.jsonl --spk_file /tmp/real256_spk.f16   --stages 5-8 --steps 200
```

对比基线：§11.2 不带 spk 的同一批数据是 200 步 loss **3.3253 → 3.0212**、
阶段 8 时长差 34.6%。带 spk 后**看两件事**：loss 是否仍降到同一量级（明显更差
说明那一格在干扰而不是帮忙），以及阶段 8 的时长差是否还在（掉到个位数说明
spk 把 instruct 的控制力压掉了 —— 那正是 ref==target 最担心的 shortcut）。

**开跑前先确认这一条**（开发机无法验证，是本轮唯一的存疑项）：

```bash
ls /home/ma-user/work/dataset/stc_data/dataset/cabin_duplex_data_artif/yibuapi_outputs/
```

§11.1 实测通过时用的是这个前缀，§11.3 已按它统一。若实际不存在，
说明两处必有一处记错了，把对的那个填回 §11.1 和 §11.3 **两处**。

`check_ddp_sync.py` 在 NCCL 和 HCCL 上都还没真跑过（脚本逻辑本身也未验证），
单卡全量不需要它——batch=8 单卡约 19-21 小时，够用。

## 17. speaker embedding（2026-08-16 新增，开发机 H200 已全验）

V2 原本没有 speaker 槽位。加它的目的是**连续对话里同一说话人音色一致**
（V4 要用），顺带让 V2 训出来的 ckpt 到 V4 不用重训架构。

### 17.1 先搞清楚一件事：speaker_encoder 只在 Base 里

| 权重 | `tts_model_type` | `speaker_encoder` 张量 |
|---|---|---|
| **Base** | `base` | **76** |
| CustomVoice | `custom_voice` | 0（用 9 个预设 spk_id 查表） |
| VoiceDesign | `voice_design` | 0 |

`modeling_qwen3_tts.py:1845` 只在 `tts_model_type == "base"` 时才实例化它。
所以想在 VoiceDesign 上用 speaker 槽位，**向量只能从 Base 那份权重来**。

好在维度对得上（已核）：Base 的 `enc_dim = 2048` == VoiceDesign talker
`hidden_size = 2048`，两边 codec 特殊 id（think/bos/eos/pad）也逐个相同。

### 17.2 做法：离线抽，训练图里不含 encoder

官方 `sft_12hz.py:82` 那行是 `speaker_encoder(ref_mels).detach()` —— **全程冻结**。
既然冻结，离线算完存盘与在线现算数学上等价，而离线的好处是训练时模型里
根本不含 speaker_encoder：显存/计算/checkpoint 全都不变，产出的 ckpt 仍是干净的
VoiceDesign。新增 `finetuning/spk_encoder.py` 只加载那 76 个张量。

**已验**：独立抽取器与 Base 模型自带的 `extract_speaker_embedding`
**逐位相同**（最大绝对差 0.0）。

存储：`[N, 2048]` 裸 fp16，43.5w 条 = **1.78GB**，训练侧 memmap 打开，
不进常驻内存（audio_codes 存 Python list 撑到 78GB 的教训）。

### 17.3 序列布局：块宽 5 → 6

speaker 那格夹在 head 与 `[codec_pad, codec_bos]` 之间，与推理侧
`cat(emb0, speaker_embed.view(1,1,-1), emb1)`（modeling:2216-2218）逐格对齐：

```
think, think_bos, lang, think_eos, [spk], codec_pad
                                    ^ 这格的 embedding 被 spk 向量**整格覆盖**
```

两个容易写错的地方：

- **是覆盖不是相加**。写成 `+=` 会掺进 codec_pad 的 embedding，训练与推理
  在这一格上不一致，且不报错。
- **位置不能写死**。官方 `sft_12hz.py:91` 是 `input_codec_embedding[:, 6, :]`，
  但 V2 的 instruct 是变长前缀，每条样本的绝对位置都不同 —— collate 额外吐
  `spk_pos` 逐条指定。

不给 `--spk-file` 时块宽仍是 5，行为与加 spk 之前完全一致（已回归）。

### 17.4 怎么跑

```bash
# 抽码时顺带抽 spk（Base 权重路径见 17.6 的问题 2）
SPK_FILE=$W/v2/spk.f16 \
SPK_MODEL_PATH=$WORK_ROOT/model/Qwen3-TTS-12Hz-1.7B-Base \
bash run_v2_npu_debug.sh

# 冒烟（阶段 4b 抽 + 5 逐格核对 + 6 带 spk 训练）
python npu_smoke_test.py --model_path <VoiceDesign> \
  --spk_model_path <Base> --use-pipeline-codes
```

`SPK_FILE` 留空 = 完全不启用，两个启动脚本都已透传（实测空值时不多出任何参数）。

**`--spk-drop-prob` 默认 0**，即训练时每条都有 spk。这意味着
**训完的模型必须永远给 spk** —— 不给就是训练中从未出现过的输入，不报错，
只表现为音色/音质乱。config 里会记 `v2_train_spk` / `v2_train_spk_drop_prob`
供推理侧对齐。

⚠️ **drop 置零 ≠ 不传 spk，这两件事不一样**：drop 掉的样本仍是 **6 格块 +
全零向量**；而推理侧 `speaker=None` 走的是 **5 格块、根本没有这一格**
（modeling:2213-2215）。所以即使把 drop 调到 0.1-0.3，做「纯 instruct」推理时
**也要照样带这一格、传全零向量**；完全不传 spk 在任何 drop 取值下都是失配。

### 17.5 开发机已验的（H200，全部实测）

| 验什么 | 结果 |
|---|---|
| 独立抽取器 vs 官方 `extract_speaker_embedding` | **逐位相同**，最大差 0.0 |
| 说话人判别力 | 原始余弦只差 +0.031（ECAPA 公共分量大），**去均值后同人 0.514 / 跨人 −0.311，间隔 +0.826 且不重叠** |
| 抽取确定性 | 同一音频两次抽取完全一致 |
| 布局逐格核对 | 块宽 5→6、spk 位置、label 整体右移 1、mask、两条 instruct 不等长时 spk_pos 各不相同 —— 18 项全过 |
| 覆盖而非相加 | spk 格的 codec 分量 == spk 向量本身；确认没掺 codec_pad |
| **这一格真的参与计算** | 换随机 spk → loss 4.15、置零 → 4.05、真实 → **3.90**（刚训 6 步就在用了） |
| `--spk-drop-prob=1.0` | 与手工置零逐位一致 |
| 续跑 spk 与 jsonl 同步截断 | 行边界/半行两种截断后行数都对齐 |
| 不给 spk 的回归 | 块宽仍 5，冒烟 5/6/7 全过 |

### 17.5b 昇腾上必须自己验的（冒烟已自动做，跑 `--spk_model_path` 就会执行）

抽 spk 这条路径在昇腾上有两个不确定处，冒烟阶段 4b 现在会自动查：

| 查什么 | 为什么 | 判据 |
|---|---|---|
| **NPU vs CPU 抽出的向量一致** | ECAPA 的 Conv1d/BatchNorm1d/attentive-pooling 在昇腾上数值是否一致没人验过。这批向量要喂进 43.5w 条，错了事后从 loss 上完全看不出来 | 余弦 ≥ 0.999，否则当场 fail 并提示"别用它抽全量" |
| **同设备重复抽取确定性** | 非确定的话，续跑接出来的后半段与前半段不是同一套向量 | 两次逐位相同 |

开发机实测：`cuda:0` vs CPU **余弦 1.000000，最大绝对差 9.83e-04**
（fp32 累加顺序差异，正常）；重复抽取逐位相同。

**已经绕开的一个坑**：`mel_spectrogram` 内含 `torch.stft`，而 torch_npu 2.1 的
stft 支持一直不稳。当前实现里 **mel 在 CPU 上算**，只有 ECAPA 前向在设备上
（与官方 `extract_speaker_embedding` 的做法相同），所以不碰 NPU 的 stft。
**改这段代码时不要顺手把 mel 挪到设备上。**

### 17.5c 抽 spk 的耗时：注意 `SPK_MEL_THREADS`

mel 在 CPU 上算，而 torch 默认把机器上**所有核**都拿去跑 —— 对一段 1-4 秒音频的
小 STFT 来说纯粹是线程调度开销。开发机 192 核上实测（48 条 1-4s，含
librosa.load + mel + ECAPA 前向）：

| | ms/条 | 43.5w 条 |
|---|---|---|
| torch 默认线程（192） | 77.6 | **9.4 h** |
| `SPK_MEL_THREADS=4`（默认值） | **6.1** | **0.74 h** |

**12.7 倍**，而且两者输出**逐位相同**（已验，线程数不影响数值）。
代码里已默认设为 4 并在算完还原，不影响进程内其它 torch 计算。

**不用手工量**：冒烟阶段 4b 现在会自动跑 `SPK_MEL_THREADS ∈ {不限,1,2,4,8}`，
打出本机最优值与 43.5w 条的折算耗时；若最优值不等于代码默认的 4，会直接 ⚠️
提示该设多少、能省几小时。**全量抽码前照它给的值设一下环境变量即可**：

```bash
SPK_MEL_THREADS=<冒烟给的值> SPK_FILE=... bash run_v2_npu_debug.sh
```

开发机上冒烟实测最优是 **8**（4.6 ms/条 → 0.6h），而默认全核是 104.4 ms/条
——**差 22 倍、12.1 小时**。核数不同的机器最优值不同，所以别照抄，看冒烟输出。

**没验的**：Base 的 embedding 空间与 VoiceDesign 的 talker 是否真的兼容。
VoiceDesign 是从 Base 继续训的，可能已经漂了。这一格是靠微调重新学出来的，
不是拿来即用 —— 只有真跑完听音色才知道。

### 17.6 需要 NPU 侧回答的数据问题

pull 到这版后请把答案直接写在本节下面：

1. **打标 jsonl 里有没有 session / 说话人标识？**（字段名是什么）
   现在用的是**逐 turn** 的 spk 向量 —— 即 ref 就是 target 自己。这样最简单，
   但向量里带着"这一句实际的说话状态"，模型可能学会从 spk 读腔调、
   让 instruct 退化成摆设。有 session 字段的话，以后可以改成**同 session 平均**
   （平掉逐句韵律、只留音色），这既杀掉那个 shortcut，也更贴"同 session 音色一致"
   的目标。现在不做，先跑第一版。

2. **NPU 机上有没有 Base 权重？完整路径是什么？**
   `$WORK_ROOT/model/Qwen3-TTS-12Hz-1.7B-Base` 是猜的。**没有 Base 就抽不了 spk**，
   这一步会直接卡住（`spk_encoder.py` 会报 "没有 speaker_encoder_config"）。

3. 磁盘：spk 文件 43.5w 条 ≈ 1.78GB，与 §11.3 的 13GB ckpt 账相加即可。

4. （仍未答）§16 那条 `ls` —— `$S` 前缀到底是哪个。

**2026-08-16 NPU 侧回填**：

1. **有 session 标识，无说话人标识**。打标 jsonl 里字段为
   `sessionid`（形如 `<uuid>__chunkNNN`，跨数据集唯一）+ `interactionid`
   （session 内轮次序号 1-10）+ `session_turn_count`。无 speaker/spk/role
   字段（data_report.md 早已确认：三层数据里都没有说话人信息）。
   后续「同 session 平均」可行：按 `sessionid` 分组即可。
2. **本机原只有 0.6B-Base，enc_dim=1024，与 VoiceDesign 1.7B 的 2048 不匹配，
   不可用于抽 spk**（spk_encoder.py 的 SPK_DIM=2048 硬核对）。
   已下载 **1.7B-Base**：`/home/ma-user/work/dataset/wsj-mimo-data/Qwen3-TTS-12Hz-1.7B-Base`
   （软链 `/home/ma-user/work/model/Qwen3-TTS-12Hz-1.7B-Base`），
   speaker_encoder 76 张量、`speaker_encoder_config.enc_dim=2048` ✓。
4. **`$S` 前缀存在且正确**：
   `ls /home/ma-user/work/dataset/stc_data/dataset/cabin_duplex_data_artif/yibuapi_outputs/`
   下三个 jsonl 都在（§11.1/§11.2 实测即用此路径，无歧义）。
