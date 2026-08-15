#!/bin/bash
# V2 VoiceDesign 微调 —— 本机 debug 版（单卡 910B2 / 64GB）。
#
# 1.7B 全参微调静态显存：--dtype fp32（默认）约 27GiB，--dtype bf16 约 12.7GiB
# （实测 accelerate 的 bf16 下参数/梯度/Adam 动量全是 bf16）。64GB 单卡装得下 ——
# **不需要任何分布式**，直接 python 起，
# 不用 torchrun、不用 FSDP、不用 MindSpeed。
#
#   bash run_v2_npu_debug.sh                      # 用默认路径
#   TRAIN_JSONL=xx.jsonl MAX_STEPS=20 bash run_v2_npu_debug.sh
#   SKIP_PREPARE=1 bash run_v2_npu_debug.sh       # audio_codes 已抽好，跳过

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FT_DIR="$(dirname "$HERE")"
. "$HERE/npu_env.sh"

PY=${PY:-python3}
DEVICE=${DEVICE:-npu:0}
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0}

MODEL_PATH=${MODEL_PATH:-/opt/huawei/quoteModel/Qwen3-TTS-12Hz-1.7B-VoiceDesign}
TOKENIZER_PATH=${TOKENIZER_PATH:-${MODEL_PATH}/speech_tokenizer}
RAW_JSONL=${RAW_JSONL:-./data_v2/train_raw.jsonl}
TRAIN_JSONL=${TRAIN_JSONL:-./data_v2/train_codes.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-./exp/v2_voicedesign_debug}

BATCH_SIZE=${BATCH_SIZE:-2}
GRAD_ACCUM=${GRAD_ACCUM:-4}
EPOCHS=${EPOCHS:-1}
LR=${LR:-2e-5}
MAX_STEPS=${MAX_STEPS:--1}
LANGUAGE=${LANGUAGE:-Chinese}
# 昇腾上 flash_attention_2 不可用（那是 CUDA kernel）。sdpa 若在 torch_npu 2.1.0
# 上静默退化到 math 分支导致显存/速度不可接受，改 eager 再试。
ATTN=${ATTN:-sdpa}

mkdir -p "$OUTPUT_DIR" ./logs
LOG=./logs/v2_npu_debug_$(date +%Y%m%d_%H%M%S).log

echo "=== V2 VoiceDesign 微调（本机 debug，单卡）==="
echo "  设备:        $DEVICE  (ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES)"
echo "  模型:        $MODEL_PATH"
echo "  训练数据:    $TRAIN_JSONL"
echo "  输出:        $OUTPUT_DIR"
echo "  bs/accum/lr: $BATCH_SIZE / $GRAD_ACCUM / $LR    attn=$ATTN  language=$LANGUAGE"
echo

npu_preflight "$PY"

for p in "$MODEL_PATH" "$TOKENIZER_PATH"; do
    [ -e "$p" ] || { echo "❌ 路径不存在: $p" >&2; exit 1; }
done

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
    --max_steps "$MAX_STEPS" \
    --log_every 1 2>&1 | tee -a "$LOG"

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "✅ 完成，ckpt 在 $OUTPUT_DIR"
else
    echo "❌ 失败，见 $LOG" >&2; exit 1
fi
