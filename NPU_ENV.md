# NPU 环境检查报告（2026-08-14）

目标：将 `finetuning/sft_12hz.py`（Qwen3-TTS 单说话人 SFT）从 CUDA 适配到 NPU。
参考仓库：`/home/ma-user/work/algorithm/Qwen35_huanyu_9B_whj`（MindSpeed-MM-XY，Qwen 改 MindSpeed 的范例）。

## 1. 硬件

| 项目 | 值 |
|---|---|
| 芯片 | 1× Ascend 910B2（`npu-smi` 显示 NPU 1，Bus-Id 0000:01:00.0） |
| HBM | 64 GB（当前占用 ~3.4 GB，空闲，无运行进程） |
| npu-smi 版本 | 25.5.1 |
| 健康状态 | OK |

**单卡环境**，无多卡/分布式拓扑（参考仓库脚本里 `NPUS_PER_NODE=8` 是集群配置，本机只有 1 张可用卡）。

## 2. 软件栈

- **CANN**: 8.1.RC1，位于 `/usr/local/Ascend/ascend-toolkit/latest`
- **Python**: conda env `PyTorch-2.1.0`（`/home/ma-user/anaconda3/envs/PyTorch-2.1.0`，Python 3.9.10，当前默认激活）
- **框架**:
  - torch 2.1.0
  - torch_npu 2.1.0.post8.dev20241029（✅ 实测 bf16 matmul 正常，`torch_npu.npu.device_count()==1`，device name `Ascend910B2`）
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
2. **flash_attention_2 不可用**：910B2 + HF 原生 transformers 下 `attn_implementation="flash_attention_2"` 会失败；改为 `eager`/SDPA，或装华为 patched transformers（参考仓库自带一份 `transformers/` 目录）
3. **bf16**: 910B2 原生支持，实测 OK
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
npu-smi info                                   # 1×910B2, OK, 64GB
python3 -c "import torch, torch_npu; print(torch_npu.npu.device_count(), torch_npu.npu.get_device_name(0))"  # 1 Ascend910B2
# bf16 matmul 2048×2048 ×20 次 ≈ 0.12s，正常
```
