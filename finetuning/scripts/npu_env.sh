#!/bin/bash
# 昇腾 NPU 环境配方 —— debug 版与集群版共用。
# 用法：在启动脚本里 `. "$(dirname "$0")/npu_env.sh"`
#
# 配方来源：NPU_ENV.md §6.1（客户现成做法）。与团队 MindSpeed 作业的区别是
# 本链路走 HF + accelerate，**不需要** megatron / mindspeed / msadapter /
# triton_ascend 那一整套，PYTHONPATH 也不用挂它们。

# 注意：本文件是**被 source** 的环境配方，故意不设 set -euo pipefail ——
# 那会污染调用方（指南 §0 让人在交互 shell 里 source 它，set -e 下任何非零
# 退出会直接杀掉终端；set -u 还会在 source CANN 的 set_env.sh 时因其中的
# 裸 $PYTHONPATH 引用而报 unbound variable）。-e 留给两个启动脚本自己设。

# ---------- 1. CANN ----------
# 本机 debug 用打包好的 CANN 8.5.0 覆盖系统的 8.1.RC1（NPU_ENV.md 已验证
# torch_npu 2.1.0.post8 在这套下正常 import 并完成 bf16 计算）；
# 集群由 ModelArts 注入 CANN_DIR，走 ascend-toolkit/set_env.sh。
CANN_DIR=${CANN_DIR:-/home/ma-user/work/dataset/CANN_805_ENV/cann85_0116}
if [ -f "${CANN_DIR}/cann-8.5.0/set_env_local.sh" ]; then
    . "${CANN_DIR}/cann-8.5.0/set_env_local.sh"
    export ASCEND_HOME_PATH="${CANN_DIR}/cann-8.5.0"
elif [ -f "${CANN_DIR}/ascend-toolkit/set_env.sh" ]; then
    . "${CANN_DIR}/ascend-toolkit/set_env.sh"
    export ASCEND_HOME_PATH="${CANN_DIR}/ascend-toolkit/latest"
elif [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    . /usr/local/Ascend/ascend-toolkit/set_env.sh
    export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
else
    echo "❌ 找不到 CANN set_env，检查 CANN_DIR=${CANN_DIR}" >&2
    return 1 2>/dev/null || exit 1   # 被 source 时 return，直接执行时 exit
fi
export ASCEND_TOOLKIT_HOME="${ASCEND_HOME_PATH}"
# 注：nnal/atb/set_env.sh 是 MindSpeed 作业才需要的，HF 路线不用 source

# ---------- 2. 昇腾调优 / 日志 ----------
export CUDA_DEVICE_MAX_CONNECTIONS=1
export ASCEND_GLOBAL_LOG_LEVEL=${ASCEND_GLOBAL_LOG_LEVEL:-3}
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-2}
export COMBINED_ENABLE=1
export CPU_AFFINITY_CONF=1
export NPU_ASD_ENABLE=0
export ASCEND_LAUNCH_BLOCKING=${ASCEND_LAUNCH_BLOCKING:-0}
export ACLNN_CACHE_LIMIT=1000
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export OMP_NUM_THREADS=1

# ---------- 3. HCCL 通信超时 ----------
# checkpoint 保存会卡住通信，Exec 超时要给大
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-3600}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-7200}
export MS_NODE_TIMEOUT=${MS_NODE_TIMEOUT:-3600}

# ---------- 4. 预检 ----------
npu_preflight() {
    local py=$1
    "$py" - <<'PYEOF' 
import sys

try:
    import torch, torch_npu           # noqa: F401
except Exception as e:
    print(f"❌ torch_npu 不可用: {type(e).__name__}: {e}")
    print("   先 source CANN 环境；若报找不到 .so，检查 CANN 与 torch_npu 版本是否配套")
    sys.exit(1)

n = torch.npu.device_count()
if n == 0:
    print("❌ 未检测到 NPU 设备")
    sys.exit(1)
print(f"✅ torch {torch.__version__} / torch_npu {torch_npu.__version__} / {n} 卡 "
      f"({torch.npu.get_device_name(0)})")

try:
    import transformers
except ImportError:
    print("❌ 未安装 transformers。pyproject 要求 4.57.3（requires-python >=3.9.0，py3.9 可用）")
    sys.exit(1)
v = tuple(int(x) for x in transformers.__version__.split(".")[:2])
if v < (4, 41):
    print(f"❌ transformers {transformers.__version__} 过低，需 >= 4.41")
    sys.exit(1)
print(f"✅ transformers {transformers.__version__}")

try:
    import accelerate
    print(f"✅ accelerate {accelerate.__version__}")
except ImportError:
    print("❌ 未安装 accelerate")
    sys.exit(1)

# 本链路只用到 Accelerator 的 6 个接口，1.0.0 即可满足；
# pyproject 钉的 1.12.0 要求 py>=3.10，昇腾现役 conda 是 py3.9，故不要照钉
from accelerate import Accelerator                       # noqa: F401
for attr in ("accumulate", "backward", "clip_grad_norm_", "prepare",
             "unwrap_model", "is_main_process"):
    if not hasattr(Accelerator, attr):
        print(f"❌ accelerate 缺少 {attr}，版本过低")
        sys.exit(1)

import qwen_tts                                          # noqa: F401
print("✅ qwen_tts 可 import")
PYEOF
}
