#!/bin/bash
# V2 VoiceDesign 微调 —— ModelArts 昇腾作业版（多机多卡）。
#
# 与 debug 版的唯一区别是**分布式与路径来源**，训练代码保持 device 无关。
# 设备可见性交给 ModelArts 调度，**不要**自设 ASCEND_RT_VISIBLE_DEVICES。
#
# 注意：1.7B 全参微调单卡约 25GiB（--dtype fp32 默认），910B2 是 64GB —— 多卡是**为了吞吐**，
# 不是因为放不下。走 accelerate 原生 DDP（数据并行），不需要 FSDP/ZeRO/MindSpeed。
#
# backend 写 nccl：torch_npu 会自动映射到 hccl（团队脚本通用写法）。
#
# 与参考仓库（duplex_huanyu_qwen35/pretrain_al_cls.sh）的唯一刻意差异是启动器：
# 它们走 megatron/mindspore 路线用 msrun；本链路是 HF + accelerate，不依赖那一套，
# torch.distributed.run + torch_npu 的 hccl 后端即可，不必引入 msrun。

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FT_DIR="$(dirname "$HERE")"
REPO_ROOT="$(dirname "$FT_DIR")"   # pyproject.toml 在仓库根，不在 finetuning/
set -euo pipefail   # npu_env.sh 不再设，由各启动脚本自己负责
. "$HERE/npu_env.sh"

PY=${PY:-python3}

# ---------- 依赖安装（集群容器是干净的，参考仓库也是在脚本内 pip install）----------
# 必须 --user：base 镜像的 conda 环境里有 root 所有的文件（如 tokenizers 的
# dist-info），普通 pip 卸载/覆盖会 Permission denied。user site 优先级更高，正好覆盖。
# transformers 钉 4.55.2 不是 4.57.3：4.56+ 无条件用 torch>=2.2 的
# torch.utils._pytree.register_pytree_node，在 torch 2.1.0 上 import 即崩（实测）。
if [ "${SKIP_INSTALL:-0}" != "1" ]; then
    pip install --user transformers==4.55.2 "accelerate==1.0.0" "safetensors>=0.4.5" \
        sox onnxruntime einops librosa soundfile "torchaudio==2.1.0" 2>&1 | tail -3
    pip install --user -e "$REPO_ROOT" --no-deps 2>&1 | tail -3
fi

npu-smi info

# ---------- ModelArts 作业变量 ----------
if [ -n "${MA_VJ_NAME:-}" ]; then
    WORK_ROOT=${WORK_ROOT:-/opt/huawei}
    NPUS_PER_NODE=${MA_NUM_GPUS:-8}
    NNODES=${MA_NUM_HOSTS:-1}
    NODE_RANK=${VC_TASK_INDEX:-0}
    MASTER_ADDR=$(echo "${VC_WORKER_HOSTS:-127.0.0.1}" | cut -d',' -f1)
    echo "=== ModelArts 作业 ${MA_VJ_NAME} ==="
else
    WORK_ROOT=${WORK_ROOT:-$PWD}
    NPUS_PER_NODE=${NPUS_PER_NODE:-1}
    NNODES=1
    NODE_RANK=0
    MASTER_ADDR=127.0.0.1
    echo "=== 非作业环境（DevContainer 调试）==="
fi
MASTER_PORT=${MASTER_PORT:-6411}   # 参考仓库同值
netstat -tuln | grep ${MASTER_PORT} || echo "端口 ${MASTER_PORT} 空闲"

MODEL_PATH=${MODEL_PATH:-${WORK_ROOT}/quoteModel/Qwen3-TTS-12Hz-1.7B-VoiceDesign}
TRAIN_JSONL=${TRAIN_JSONL:-${WORK_ROOT}/dataset/wsj-mimo-data/qwen_tts_exp/data/v2/train_codes.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-${WORK_ROOT}/dataset/wsj-mimo-data/qwen_tts_exp/runs/${RUN_NAME:-v1}}

BATCH_SIZE=${BATCH_SIZE:-4}
GRAD_ACCUM=${GRAD_ACCUM:-2}
EPOCHS=${EPOCHS:-1}
LR=${LR:-2e-5}
LANGUAGE=${LANGUAGE:-Chinese}
ATTN=${ATTN:-sdpa}
# fp32（默认）约 25GiB / bf16 约 12.7GiB，910B1 是 64GB，默认走 fp32 更稳
DTYPE=${DTYPE:-fp32}
# 每 N 步存一次。全量 1 epoch 是十几小时，只在 epoch 末落盘的话中途崩就全丢
SAVE_EVERY=${SAVE_EVERY:-500}
# 序列长度分桶：昇腾按形状编译算子，分桶让编译结果跨 step 复用（NPU 实测省 ~13%）
LENGTH_BUCKET=${LENGTH_BUCKET:-64}
# speaker 向量。SPK_MODEL_PATH 必须指向 **Base** 权重 —— speaker_encoder
# 只在 tts_model_type=base 里有，VoiceDesign/CustomVoice 都是 0 张量。
# 默认开：spk 文件放 qwen3_spk_emb/（1.78GB，行序与 train_codes.jsonl 对齐）。
# 两个都留空 = 不带 speaker 槽位，行为与加 spk 之前完全一致。
SPK_FILE=${SPK_FILE:-${WORK_ROOT}/dataset/wsj-mimo-data/qwen_tts_exp/qwen3_spk_emb/spk.f16}
SPK_MODEL_PATH=${SPK_MODEL_PATH:-${WORK_ROOT}/quoteModel/Qwen3-TTS-12Hz-1.7B-Base}
SPK_DROP_PROB=${SPK_DROP_PROB:-0}

LOGDIR=${LOGDIR:-${WORK_ROOT}/logs}
mkdir -p "$OUTPUT_DIR" "$LOGDIR"
LOG="${LOGDIR}/v2_npu_$(date +%Y%m%d_%H%M%S)_rank${NODE_RANK}.log"

WORLD=$((NPUS_PER_NODE * NNODES))
echo "  节点:        $NNODES  本机 rank $NODE_RANK  每节点 $NPUS_PER_NODE 卡  (world=$WORLD)"
echo "  master:      $MASTER_ADDR:$MASTER_PORT"
echo "  模型:        $MODEL_PATH"
echo "  训练数据:    $TRAIN_JSONL"
echo "  输出:        $OUTPUT_DIR"
echo "  全局 batch:  $((BATCH_SIZE * GRAD_ACCUM * WORLD))    attn=$ATTN  dtype=$DTYPE"
echo "  save_every:  $SAVE_EVERY    length_bucket: $LENGTH_BUCKET"
echo "  speaker:     ${SPK_FILE:-（不用）}${SPK_FILE:+  drop_prob=$SPK_DROP_PROB}"
echo

npu_preflight "$PY"

for p in "$MODEL_PATH" "$TRAIN_JSONL"; do
    [ -e "$p" ] || { echo "❌ 路径不存在: $p" >&2; exit 1; }
done

# 集群版不做数据准备（TRAIN_JSONL 必须已存在），spk 文件同理 —— 先在
# debug 机上跑 prepare_data.py --spk_out 抽好再上集群
if [ -n "$SPK_FILE" ] && [ ! -f "$SPK_FILE" ]; then
    echo "❌ 给了 SPK_FILE 但文件不存在: $SPK_FILE（先用 prepare_data.py --spk_out 抽）" >&2
    exit 1
fi

cd "$FT_DIR"

"$PY" -m torch.distributed.run \
    --nproc_per_node "$NPUS_PER_NODE" \
    --nnodes "$NNODES" \
    --node_rank "$NODE_RANK" \
    --master_addr "$MASTER_ADDR" \
    --master_port "$MASTER_PORT" \
    sft_12hz_voicedesign.py \
    --init_model_path "$MODEL_PATH" \
    --train_jsonl "$TRAIN_JSONL" \
    --output_model_path "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --grad_accum "$GRAD_ACCUM" \
    --num_epochs "$EPOCHS" \
    --lr "$LR" \
    --language "$LANGUAGE" \
    --attn "$ATTN" \
    --dtype "$DTYPE" \
    --save-every "$SAVE_EVERY" \
    --length-bucket "$LENGTH_BUCKET" \
    ${SPK_FILE:+--spk-file "$SPK_FILE" --spk-drop-prob "$SPK_DROP_PROB"} \
    --log_every 10 2>&1 | tee "$LOG"

rc=${PIPESTATUS[0]}
chmod 440 "$LOG" 2>/dev/null || true
if [ $rc -eq 0 ]; then
    echo "✅ 完成，ckpt 在 $OUTPUT_DIR"
else
    echo "❌ 失败，见 $LOG" >&2; exit 1
fi
