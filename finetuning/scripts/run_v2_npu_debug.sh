#!/bin/bash
# V2 VoiceDesign 微调 —— 本机 debug 版（单卡 910B2 / 64GB）。
#
# 1.7B 全参微调静态显存：--dtype fp32（默认）约 25GiB，--dtype bf16 约 12.7GiB
# （实测 accelerate 的 bf16 下参数/梯度/Adam 动量全是 bf16）。64GB 单卡装得下 ——
# **不需要任何分布式**，直接 python 起，
# 不用 torchrun、不用 FSDP、不用 MindSpeed。
#
#   bash run_v2_npu_debug.sh                      # 用默认路径
#   TRAIN_JSONL=xx.jsonl MAX_STEPS=20 bash run_v2_npu_debug.sh
#   SKIP_PREPARE=1 bash run_v2_npu_debug.sh       # audio_codes 已抽好，跳过

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FT_DIR="$(dirname "$HERE")"
set -euo pipefail   # npu_env.sh 不再设，由各启动脚本自己负责
. "$HERE/npu_env.sh"

PY=${PY:-python3}

# ---------- 依赖安装（与集群版一致；SKIP_INSTALL=1 可跳过）----------
# 必须 --user：base 镜像的 conda 环境里有 root 所有的文件（如 tokenizers 的
# dist-info），普通 pip 卸载/覆盖会 Permission denied。user site 优先级更高，正好覆盖。
# transformers 钉 4.55.2 不是 4.57.3：4.56+ 无条件用 torch>=2.2 的
# torch.utils._pytree.register_pytree_node，在 torch 2.1.0 上 import 即崩（实测）。
if [ "${SKIP_INSTALL:-0}" != "1" ]; then
    pip install --user transformers==4.55.2 "accelerate==1.0.0" "safetensors>=0.4.5" \
        sox onnxruntime einops librosa soundfile "torchaudio==2.1.0" 2>&1 | tail -3
    pip install --user -e "$FT_DIR" --no-deps 2>&1 | tail -3
fi

DEVICE=${DEVICE:-npu:0}
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0}

# 路径约定对齐参考仓库（duplex_huanyu_qwen35 与 MiMo v2 脚本）。
# 调试机与集群同构，只差前缀，故收敛成 WORK_ROOT 一个开关：
#   调试机  WORK_ROOT=/home/ma-user/work     （默认）
#   集群    WORK_ROOT=/opt/huawei
#   模型 $WORK_ROOT/model|quoteModel · 数据 $WORK_ROOT/dataset · 输出 $WORK_ROOT/quoteModel
WORK_ROOT=${WORK_ROOT:-/home/ma-user/work}
DATA_ROOT=${DATA_ROOT:-$WORK_ROOT/dataset/duplex_whj_data/v2}

MODEL_PATH=${MODEL_PATH:-$WORK_ROOT/model/Qwen3-TTS-12Hz-1.7B-VoiceDesign}
TOKENIZER_PATH=${TOKENIZER_PATH:-${MODEL_PATH}/speech_tokenizer}
RAW_JSONL=$(realpath -m "${RAW_JSONL:-$DATA_ROOT/train_raw.jsonl}")
TRAIN_JSONL=$(realpath -m "${TRAIN_JSONL:-$DATA_ROOT/train_codes.jsonl}")
OUTPUT_DIR=$(realpath -m "${OUTPUT_DIR:-$WORK_ROOT/quoteModel/duplex_whj_exp/v2_single_text_turn_tts/debug}")

BATCH_SIZE=${BATCH_SIZE:-2}
GRAD_ACCUM=${GRAD_ACCUM:-4}
EPOCHS=${EPOCHS:-1}
LR=${LR:-2e-5}
MAX_STEPS=${MAX_STEPS:--1}
LANGUAGE=${LANGUAGE:-Chinese}
# 昇腾上 flash_attention_2 不可用（那是 CUDA kernel）。sdpa 若在 torch_npu 2.1.0
# 上静默退化到 math 分支导致显存/速度不可接受，改 eager 再试。
ATTN=${ATTN:-sdpa}
# fp32（默认）约 25GiB / bf16 约 12.7GiB，910B2 是 64GB，默认走 fp32 更稳
DTYPE=${DTYPE:-fp32}

# 必须绝对路径：下面会 cd 到 $FT_DIR，相对路径的 ./logs 会漂到别处，
# 叠加 set -o pipefail 会让 tee 失败直接终止脚本
LOGDIR="$FT_DIR/logs"
mkdir -p "$OUTPUT_DIR" "$LOGDIR"
LOG="$LOGDIR/v2_npu_debug_$(date +%Y%m%d_%H%M%S).log"

echo "=== V2 VoiceDesign 微调（本机 debug，单卡）==="
echo "  设备:        $DEVICE  (ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES)"
echo "  模型:        $MODEL_PATH"
echo "  训练数据:    $TRAIN_JSONL"
echo "  输出:        $OUTPUT_DIR"
echo "  bs/accum/lr: $BATCH_SIZE / $GRAD_ACCUM / $LR    attn=$ATTN  dtype=$DTYPE  language=$LANGUAGE"
echo

npu_preflight "$PY"

for p in "$MODEL_PATH" "$TOKENIZER_PATH"; do
    [ -e "$p" ] || { echo "❌ 路径不存在: $p" >&2; exit 1; }
done
# SKIP_PREPARE=1 时不会生成 TRAIN_JSONL，得先确认它在
if [ "${SKIP_PREPARE:-0}" = "1" ] && [ ! -f "$TRAIN_JSONL" ]; then
    echo "❌ SKIP_PREPARE=1 但训练数据不存在: $TRAIN_JSONL" >&2; exit 1
fi

cd "$FT_DIR"

# ---- 抽 audio_codes（NPU 上 --device 必须显式传 npu:N，默认值是 cuda:0）----
if [ "${SKIP_PREPARE:-0}" != "1" ]; then
    [ -f "$RAW_JSONL" ] || { echo "❌ 找不到 $RAW_JSONL（先跑 prepare_v2_data.py）" >&2; exit 1; }
    echo ">>> 抽 audio_codes"
    "$PY" prepare_data.py \
        --device "$DEVICE" \
        --tokenizer_model_path "$TOKENIZER_PATH" \
        --input_jsonl "$RAW_JSONL" \
        --output_jsonl "$TRAIN_JSONL" 2>&1 | tee -a "$LOG"
fi

echo ">>> 训练"
"$PY" sft_12hz_voicedesign.py \
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
    --max_steps "$MAX_STEPS" \
    --log_every 1 2>&1 | tee -a "$LOG"

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "✅ 完成，ckpt 在 $OUTPUT_DIR"
else
    echo "❌ 失败，见 $LOG" >&2; exit 1
fi
