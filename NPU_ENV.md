# NPU 环境检查报告（2026-08-14）

目标：将 `finetuning/sft_12hz.py`（Qwen3-TTS 单说话人 SFT）从 CUDA 适配到 NPU。
参考仓库：`/home/ma-user/work/algorithm/Qwen35_huanyu_9B_whj`（MindSpeed-MM-XY，Qwen 改 MindSpeed 的范例）。

## 1. 硬件

| 项目 | 值 |
|---|---|
| 芯片 | 1× Ascend 910B1（`npu-smi` 显示 NPU 1，Bus-Id 0000:01:00.0） |
| HBM | 64 GB（当前占用 ~3.4 GB，空闲，无运行进程） |
| npu-smi 版本 | 25.5.1 |
| 健康状态 | OK |

**单卡环境**，无多卡/分布式拓扑（参考仓库脚本里 `NPUS_PER_NODE=8` 是集群配置，本机只有 1 张可用卡）。

## 2. 软件栈

- **CANN**: 8.1.RC1，位于 `/usr/local/Ascend/ascend-toolkit/latest`
- **Python**: conda env `PyTorch-2.1.0`（`/home/ma-user/anaconda3/envs/PyTorch-2.1.0`，Python 3.9.10，当前默认激活）
- **框架**:
  - torch 2.1.0
  - torch_npu 2.1.0.post8.dev20241029（✅ 实测 bf16 matmul 正常，`torch_npu.npu.device_count()==1`，device name `Ascend910B1`）
  - accelerate 1.0.0
  - safetensors 0.4.5、librosa 0.10.2.post1
- **缺失**（适配前需要安装）:
  - `transformers` ❌（当前 `import qwen_tts` 直接报 `ModuleNotFoundError: No module named 'transformers'`）
  - qwen-tts 包未安装
  - 无 deepspeed / megatron / mindspeed 的 pip 包（参考仓库不靠 pip 装，见第 4 节）
- **pip 源**: 华为内部镜像 `http://repo.huaweicloud.com/repository/pypi/simple/`（pip.conf 已配置，可正常装包）
- **环境变量**（已设好）: `ASCEND_VISIBLE_DEVICES=1`、`RANK_ID=0`、`ASCEND_TOOLKIT_HOME`、`LD_LIBRARY_PATH` 等 Ascend 相关变量齐全

## 3. 当前训练脚本（待适配）

`finetuning/sft_12hz.py` 使用：

- `accelerate.Accelerator(gradient_accumulation_steps=4, mixed_precision="bf16")`
- `Qwen3TTSModel.from_pretrained(..., attn_implementation="flash_attention_2")`
- 原生 `DataLoader` + `AdamW` + `accelerator.accumulate` 手写训练循环（非 Trainer）
- `safetensors.torch.save_file` 保存 checkpoint
- `prepare_data.py` 中 `--device cuda:0` 也需要改

适配要点：

1. **device**: 全部 `cuda` → `npu`（`import torch_npu`；accelerate 可传 `device_placement`/直接用 `"npu"`，或用 torch_npu 的 `transfer_to_npu`/msadaptor）
2. **flash_attention_2 不可用**：910B1 + HF 原生 transformers 下 `attn_implementation="flash_attention_2"` 会失败；改为 `eager`/SDPA，或装华为 patched transformers（参考仓库自带一份 `transformers/` 目录）
3. **bf16**: 910B1 原生支持，实测 OK
4. **1.7B/0.6B 模型 bf16 微调**，64GB HBM 单卡足够

## 4. 参考仓库：Qwen35_huanyu_9B_whj（MindSpeed-MM-XY）

这是华为 MindSpeed（Megatron 系）的多模态 LLM 预训练框架，Qwen 已改为 MindSpeed 的现成范例：

- **结构**: 仓库根目录直接含 `megatron/`、`mindspeed/`、`mindspeed_llm/`、`mindspeed_mm/`、`msadapter/`、`transformers/`（华为定制版）源码
- **不通过 pip 安装**，靠 PYTHONPATH 使用（见 `examples/mm_model/audio_ds_qwen35/*.sh`）:
  ```
  export PYTHONPATH=${MODEL_ENV}:${MSA_ENV}:<repo>/mindspeed:<repo>/megatron:$PYTHONPATH
  export PYTHONPATH=<repo>/msadapter/msa_thirdparty:$PYTHONPATH
  ```
- **启动方式**: megatron 式 torchrun（`--local_worker_num` 等参数，见 launch 脚本）
- **训练脚本**: `pretrain_gpt.py` / `posttrain_gpt.py`，入口 `from mindspeed_llm import megatron_adaptor`、`from mindspeed_llm.training import pretrain`
- **注意**: 它是 Megatron-Core 预训练范式，与 HF accelerate 微调范式不同；主要借鉴价值是 NPU 算子/适配写法（如 patched transformers、msadapter、bf16/通信配置），不必整体照搬
- 同目录下其他可参考仓库：`mindspeed_reject_lyk_tsl`、`minmo_mindspeed_guian_whj`、`llama_factory_whj_260410_guian` 等

## 5. 验证命令（已验证通过）

```bash
npu-smi info                                   # 1×910B1, OK, 64GB
python3 -c "import torch, torch_npu; print(torch_npu.npu.device_count(), torch_npu.npu.get_device_name(0))"  # 1 Ascend910B1
# bf16 matmul 2048×2048 ×20 次 ≈ 0.12s，正常
```

## 6. 运行形态：本机 debug，集群提交跑

- **本机（本仓库所在机器）**：单卡 910B1，用于 debug，环境问题尽量通过启动脚本解决
- **最终运行**：ModelArts 集群提交（8 卡/节点、多机），环境由集群注入（`VC_WORKER_HOSTS`/`MA_NUM_HOSTS`/`VC_TASK_INDEX`/`CANN_DIR`）
- 参考仓库已给出「debug 版 + 集群版」脚本对：
  - debug 版：`examples/mm_model/duplex_huanyu_qwen35/pretrain_debug_cls.sh`（本机路径硬编码、单机 127.0.0.1、`--is-debug`）
  - 集群版：`examples/mm_model/duplex_huanyu_qwen35/pretrain_al_cls.sh`（ModelArts 变量、`--load`/`--skip-redundent-pp-load`）
- 建议 qwen3-tts 的 NPU 适配同样产出两个启动脚本，训练代码保持 device 无关

### 6.1 启动脚本环境配方（客户现成做法，可直接套用）

以下来自 debug 版脚本（集群版仅路径来源不同，变量配方一致）：

1. **source CANN 环境**（本机用本地 CANN 8.5.0，覆盖系统 8.1.RC1）：
   ```bash
   CANN_DIR=/home/ma-user/work/dataset/CANN_805_ENV/cann85_0116
   source "${CANN_DIR}"/cann-8.5.0/set_env_local.sh
   source "${CANN_DIR}"/nnal/atb/set_env.sh
   export ASCEND_HOME_PATH="${CANN_DIR}/cann-8.5.0"
   export ASCEND_TOOLKIT_HOME="${ASCEND_HOME_PATH}"
   ```
   ✅ 已验证：source 后 torch_npu 2.1.0.post8 在本机正常 import 并完成 bf16 计算
   （集群版对应 `source "${CANN_DIR}"/ascend-toolkit/set_env.sh`，CANN_DIR 由集群注入）
2. **Ascend 调优/日志变量**（debug/集群两版相同）：
   ```bash
   export CUDA_DEVICE_MAX_CONNECTIONS=1
   export ASCEND_GLOBAL_LOG_LEVEL=3
   export TASK_QUEUE_ENABLE=2 COMBINED_ENABLE=1 CPU_AFFINITY_CONF=1
   export NPU_ASD_ENABLE=0 ASCEND_LAUNCH_BLOCKING=0 ACLNN_CACHE_LIMIT=1000
   ```
3. **HCCL 通信超时**：
   ```bash
   export MS_NODE_TIMEOUT=3600 HCCL_CONNECT_TIMEOUT=3600 HCCL_EXEC_TIMEOUT=7200
   ```
4. **依赖安装 + triton 定制**（本机包路径 `/home/ma-user/work/dataset/sfs_al/pkg/`）：
   ```bash
   pip install numba llvmlite zarr torchvision==0.16.1
   pip install <sfs_al>/pkg/szy_build_0926/triton_ascend-3.2.0+git9058aa1f-cp39-cp39-linux_aarch64.whl --no-deps
   export TRITON_BACKEND=mindspore
   # 再覆盖 site-packages/triton/backends/ascend/backend_register.py 和 testing.py
   ```
   ⚠️ 本机当前**未装 triton**（`pip show triton` 为空），跑这套前需先按上面步骤安装
5. **PYTHONPATH**（仓库内源码直用，非 pip 安装）：
   ```bash
   export PYTHONPATH=<repo>/msadapter/msa_thirdparty:<repo>/mindspeed-llm:<repo>/msadapter:<repo>/mindspeed:<repo>/megatron:$PYTHONPATH
   ```
   ⚠️ 本机这份参考仓库**缺 `mindspeed-llm` 目录**（集群版本才有）
6. **分布式**：debug 版 `NPUS_PER_NODE=4`（本机只有 1 卡，需改 1）`MASTER_ADDR=127.0.0.1`；集群版读 ModelArts 环境变量，`msrun` 启动
