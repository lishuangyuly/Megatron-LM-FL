#!/usr/bin/env bash

set -euo pipefail

# Customizable Megatron-LM-FL training entry point.
#
# Single-node example:
#   GPUS_PER_NODE=4 bash examples/aibench/train.sh
#
# Two-node example (run the same code and environment on both nodes):
#   # node 0
#   MASTER_ADDR=10.0.0.1 MASTER_PORT=6000 NNODES=2 NODE_RANK=0 \
#     GPUS_PER_NODE=8 bash examples/aibench/train.sh
#   # node 1
#   MASTER_ADDR=10.0.0.1 MASTER_PORT=6000 NNODES=2 NODE_RANK=1 \
#     GPUS_PER_NODE=8 bash examples/aibench/train.sh
#
# Model, parallelism, data, and optimization settings can all be overridden with
# environment variables. The defaults below reproduce the reference MoE shape
# from the original aibench script; tune them for the target hardware.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEGATRON_PATH="${MEGATRON_PATH:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"

export PYTHONPATH="${MEGATRON_PATH}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-32}"
export NVTE_ALLOW_NONDETERMINISTIC_ALGO="${NVTE_ALLOW_NONDETERMINISTIC_ALGO:-0}"

# Distributed configuration. Under Slurm, NNODES and NODE_RANK default to the
# allocation values; otherwise they default to single-node training.
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
NNODES="${NNODES:-${SLURM_NNODES:-1}}"
NODE_RANK="${NODE_RANK:-${SLURM_NODEID:-0}}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-6000}"
WORLD_SIZE=$((GPUS_PER_NODE * NNODES))

# Model dimensions.
NUM_LAYERS="${NUM_LAYERS:-8}"
HIDDEN_SIZE="${HIDDEN_SIZE:-4096}"
FFN_HIDDEN_SIZE="${FFN_HIDDEN_SIZE:-2048}"
NUM_ATTENTION_HEADS="${NUM_ATTENTION_HEADS:-64}"
NUM_QUERY_GROUPS="${NUM_QUERY_GROUPS:-8}"
SEQ_LENGTH="${SEQ_LENGTH:-1024}"
MAX_POSITION_EMBEDDINGS="${MAX_POSITION_EMBEDDINGS:-32768}"
ROTARY_BASE="${ROTARY_BASE:-1000000}"
INIT_METHOD_STD="${INIT_METHOD_STD:-0.01}"
ATTENTION_DROPOUT="${ATTENTION_DROPOUT:-0.0}"
HIDDEN_DROPOUT="${HIDDEN_DROPOUT:-0.0}"

# MoE dimensions. Set NUM_EXPERTS=0, EP=1, and ETP=1 for a dense model. For the
# current router configuration, MOE_ROUTER_TOPK must be at least 2.
NUM_EXPERTS="${NUM_EXPERTS:-8}"
MOE_ROUTER_TOPK="${MOE_ROUTER_TOPK:-2}"
MOE_AUX_LOSS_COEFF="${MOE_AUX_LOSS_COEFF:-1e-2}"

# Parallelism. NUM_LAYERS_PER_VIRTUAL_PIPELINE_STAGE=0 disables VPP. The
# combined FL Offload path has been validated most thoroughly with PP > 1 and
# VPP enabled.
TP="${TP:-1}"
PP="${PP:-2}"
EP="${EP:-2}"
ETP="${ETP:-1}"
CP="${CP:-1}"
NUM_LAYERS_PER_VIRTUAL_PIPELINE_STAGE="${NUM_LAYERS_PER_VIRTUAL_PIPELINE_STAGE:-1}"

# Optimization.
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
TRAIN_ITERS="${TRAIN_ITERS:-5}"
LR="${LR:-1e-4}"
MIN_LR="${MIN_LR:-1e-5}"
LR_DECAY_ITERS="${LR_DECAY_ITERS:-10000}"
LR_WARMUP_ITERS="${LR_WARMUP_ITERS:-500}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
CLIP_GRAD="${CLIP_GRAD:-1.0}"

# Data. USE_MOCK_DATA=1 requires no external dataset and is suitable for model
# and memory experiments. For real data, set USE_MOCK_DATA=0 together with
# TOKENIZER_MODEL and DATA_PATH.
USE_MOCK_DATA="${USE_MOCK_DATA:-1}"
VOCAB_SIZE="${VOCAB_SIZE:-8192}"
TOKENIZER_TYPE="${TOKENIZER_TYPE:-HuggingFaceTokenizer}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-}"
DATA_PATH="${DATA_PATH:-}"
DATA_SPLIT="${DATA_SPLIT:-98,2,0}"

# Checkpoint and logging. Empty SAVE_PATH/LOAD_PATH disable saving/loading.
SAVE_PATH="${SAVE_PATH:-}"
LOAD_PATH="${LOAD_PATH:-}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"
EVAL_INTERVAL="${EVAL_INTERVAL:-1000}"
EVAL_ITERS="${EVAL_ITERS:-0}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/$(date +%Y%m%d_%H%M%S)}"

# Optional profiler configuration. Set ENABLE_PROFILE=1 to generate traces.
ENABLE_PROFILE="${ENABLE_PROFILE:-0}"
PROFILE_RANKS="${PROFILE_RANKS:-0}"
PROFILE_STEP_START="${PROFILE_STEP_START:-2}"
PROFILE_STEP_END="${PROFILE_STEP_END:-3}"
PROFILE_DIR="${PROFILE_DIR:-${LOG_DIR}/torch_prof}"
DRY_RUN="${DRY_RUN:-0}"

# Read-only saved-tensor observer. It does not move activations and is separate
# from both PyTorch Profiler and FL Offload. The current implementation requires
# --no-gradient-accumulation-fusion, which is already present in TRAINING_ARGS.
ENABLE_FL_SAVED_TENSOR_PROFILE="${ENABLE_FL_SAVED_TENSOR_PROFILE:-0}"
FL_SAVED_TENSOR_PROFILE_SCOPES="${FL_SAVED_TENSOR_PROFILE_SCOPES:-qkv_linear core_attn attn_proj expert_fc1 moe_act expert_fc2}"
FL_SAVED_TENSOR_PROFILE_MAX_REPORTS="${FL_SAVED_TENSOR_PROFILE_MAX_REPORTS:-1}"

# FL Offload tuning values used by the commented block below.
FL_OFFLOAD_MIB="${FL_OFFLOAD_MIB:-27}"
FL_MIN_TENSOR_BYTES="${FL_MIN_TENSOR_BYTES:-1048576}"
FL_OFFLOAD_STAGES="${FL_OFFLOAD_STAGES:-4}"
FL_STAGE_ASSIGNMENT="${FL_STAGE_ASSIGNMENT:-0 1 2 3}"

for value in \
    "${GPUS_PER_NODE}" "${NNODES}" "${NODE_RANK}" "${NUM_LAYERS}" \
    "${HIDDEN_SIZE}" "${FFN_HIDDEN_SIZE}" "${NUM_ATTENTION_HEADS}" \
    "${NUM_QUERY_GROUPS}" "${SEQ_LENGTH}" "${MAX_POSITION_EMBEDDINGS}" \
    "${NUM_EXPERTS}" "${MOE_ROUTER_TOPK}" "${TP}" "${PP}" "${EP}" \
    "${ETP}" "${CP}" "${NUM_LAYERS_PER_VIRTUAL_PIPELINE_STAGE}" \
    "${MICRO_BATCH_SIZE}" "${GLOBAL_BATCH_SIZE}" "${TRAIN_ITERS}" \
    "${MASTER_PORT}" "${FL_OFFLOAD_MIB}" "${FL_MIN_TENSOR_BYTES}" \
    "${FL_OFFLOAD_STAGES}" "${FL_SAVED_TENSOR_PROFILE_MAX_REPORTS}"; do
    if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
        echo "Expected a non-negative integer, got: ${value}" >&2
        exit 2
    fi
done

if (( GPUS_PER_NODE < 1 || NNODES < 1 || NUM_LAYERS < 1 || HIDDEN_SIZE < 1 ||
      FFN_HIDDEN_SIZE < 1 || NUM_ATTENTION_HEADS < 1 || SEQ_LENGTH < 1 ||
      TP < 1 || PP < 1 || EP < 1 || ETP < 1 || CP < 1 ||
      MICRO_BATCH_SIZE < 1 || GLOBAL_BATCH_SIZE < 1 || TRAIN_ITERS < 1 ||
      FL_OFFLOAD_STAGES < 1 )); then
    echo "GPU, model, parallelism, batch, iteration, and stage sizes must be positive" >&2
    exit 2
fi
if [[ "${NODE_RANK}" -ge "${NNODES}" ]]; then
    echo "NODE_RANK=${NODE_RANK} must be smaller than NNODES=${NNODES}" >&2
    exit 2
fi
if (( MASTER_PORT < 1 || MASTER_PORT > 65535 )); then
    echo "MASTER_PORT must be in [1, 65535]" >&2
    exit 2
fi
if [[ "${NNODES}" -gt 1 && ( "${MASTER_ADDR}" == "127.0.0.1" || "${MASTER_ADDR}" == "localhost" ) ]]; then
    echo "Multi-node training requires MASTER_ADDR to be reachable from every node" >&2
    exit 2
fi

MODEL_PARALLEL_SIZE=$((TP * PP * CP))
if (( WORLD_SIZE % MODEL_PARALLEL_SIZE != 0 )); then
    echo "WORLD_SIZE=${WORLD_SIZE} must be divisible by TP*PP*CP=${MODEL_PARALLEL_SIZE}" >&2
    exit 2
fi
DATA_PARALLEL_SIZE=$((WORLD_SIZE / MODEL_PARALLEL_SIZE))
# if (( DATA_PARALLEL_SIZE % EP != 0 )); then
#     echo "Data parallel size ${DATA_PARALLEL_SIZE} must be divisible by EP=${EP}" >&2
#     exit 2
# fi
if (( NUM_LAYERS % PP != 0 )); then
    echo "NUM_LAYERS=${NUM_LAYERS} must be divisible by PP=${PP}" >&2
    exit 2
fi
if (( NUM_ATTENTION_HEADS % TP != 0 )); then
    echo "NUM_ATTENTION_HEADS=${NUM_ATTENTION_HEADS} must be divisible by TP=${TP}" >&2
    exit 2
fi
if (( HIDDEN_SIZE % NUM_ATTENTION_HEADS != 0 )); then
    echo "HIDDEN_SIZE=${HIDDEN_SIZE} must be divisible by NUM_ATTENTION_HEADS=${NUM_ATTENTION_HEADS}" >&2
    exit 2
fi
if (( NUM_QUERY_GROUPS > 0 && NUM_ATTENTION_HEADS % NUM_QUERY_GROUPS != 0 )); then
    echo "NUM_ATTENTION_HEADS must be divisible by NUM_QUERY_GROUPS" >&2
    exit 2
fi
if (( SEQ_LENGTH > MAX_POSITION_EMBEDDINGS )); then
    echo "SEQ_LENGTH cannot exceed MAX_POSITION_EMBEDDINGS" >&2
    exit 2
fi
if (( GLOBAL_BATCH_SIZE % (MICRO_BATCH_SIZE * DATA_PARALLEL_SIZE) != 0 )); then
    echo "GLOBAL_BATCH_SIZE must be divisible by MICRO_BATCH_SIZE*DATA_PARALLEL_SIZE" >&2
    exit 2
fi
if (( NUM_EXPERTS > 0 )); then
    if (( MOE_ROUTER_TOPK < 2 )); then
        echo "MOE_ROUTER_TOPK must be at least 2 for this router configuration" >&2
        exit 2
    fi
    if (( NUM_EXPERTS % EP != 0 )); then
        echo "NUM_EXPERTS=${NUM_EXPERTS} must be divisible by EP=${EP}" >&2
        exit 2
    fi
elif (( EP != 1 || ETP != 1 )); then
    echo "Dense training requires EP=1 and ETP=1" >&2
    exit 2
fi

if [[ "${USE_MOCK_DATA}" != "0" && "${USE_MOCK_DATA}" != "1" ]]; then
    echo "USE_MOCK_DATA must be 0 or 1" >&2
    exit 2
fi
if [[ "${ENABLE_PROFILE}" != "0" && "${ENABLE_PROFILE}" != "1" ]]; then
    echo "ENABLE_PROFILE must be 0 or 1" >&2
    exit 2
fi
if [[ "${DRY_RUN}" != "0" && "${DRY_RUN}" != "1" ]]; then
    echo "DRY_RUN must be 0 or 1" >&2
    exit 2
fi
if [[ "${ENABLE_FL_SAVED_TENSOR_PROFILE}" != "0" && "${ENABLE_FL_SAVED_TENSOR_PROFILE}" != "1" ]]; then
    echo "ENABLE_FL_SAVED_TENSOR_PROFILE must be 0 or 1" >&2
    exit 2
fi
if [[ "${USE_MOCK_DATA}" == "0" && ( -z "${TOKENIZER_MODEL}" || -z "${DATA_PATH}" ) ]]; then
    echo "Real-data training requires TOKENIZER_MODEL and DATA_PATH" >&2
    exit 2
fi

read -r -a FL_STAGE_ASSIGNMENT_ARGS <<< "${FL_STAGE_ASSIGNMENT}"
if (( ${#FL_STAGE_ASSIGNMENT_ARGS[@]} == 0 )); then
    echo "FL_STAGE_ASSIGNMENT must contain at least one stage index" >&2
    exit 2
fi
for stage in "${FL_STAGE_ASSIGNMENT_ARGS[@]}"; do
    if [[ ! "${stage}" =~ ^-?[0-9]+$ ]] || (( stage < -1 || stage >= FL_OFFLOAD_STAGES )); then
        echo "Each FL stage assignment must be -1 or in [0, FL_OFFLOAD_STAGES)" >&2
        exit 2
    fi
done

DISTRIBUTED_ARGS=(
    --nproc_per_node "${GPUS_PER_NODE}"
    --nnodes "${NNODES}"
    --node_rank "${NODE_RANK}"
    --master_addr "${MASTER_ADDR}"
    --master_port "${MASTER_PORT}"
)

MODEL_ARGS=(
    --use-mcore-models
    --transformer-impl transformer_engine
    --attention-backend flash
    --disable-bias-linear
    --seq-length "${SEQ_LENGTH}"
    --max-position-embeddings "${MAX_POSITION_EMBEDDINGS}"
    --num-layers "${NUM_LAYERS}"
    --hidden-size "${HIDDEN_SIZE}"
    --ffn-hidden-size "${FFN_HIDDEN_SIZE}"
    --num-attention-heads "${NUM_ATTENTION_HEADS}"
    --init-method-std "${INIT_METHOD_STD}"
    --attention-dropout "${ATTENTION_DROPOUT}"
    --hidden-dropout "${HIDDEN_DROPOUT}"
    --normalization RMSNorm
    --position-embedding-type rope
    --rotary-base "${ROTARY_BASE}"
    --swiglu
    --untie-embeddings-and-output-weights
    --attention-softmax-in-fp32
)

if (( NUM_QUERY_GROUPS > 0 )); then
    MODEL_ARGS+=(
        --group-query-attention
        --num-query-groups "${NUM_QUERY_GROUPS}"
    )
fi

MOE_ARGS=()
if (( NUM_EXPERTS > 0 )); then
    MOE_ARGS+=(
        --num-experts "${NUM_EXPERTS}"
        --moe-router-topk "${MOE_ROUTER_TOPK}"
        --moe-router-load-balancing-type aux_loss
        --moe-aux-loss-coeff "${MOE_AUX_LOSS_COEFF}"
        --moe-router-force-load-balancing
        --moe-expert-capacity-factor 1.0
        --moe-pad-expert-input-to-capacity
        --moe-grouped-gemm
        --moe-permute-fusion
        --moe-token-dispatcher-type alltoall
    )
    if (( EP > 1 )); then
        MOE_ARGS+=(--overlap-moe-expert-parallel-comm)
    fi
fi

DATA_ARGS=(--split "${DATA_SPLIT}")
if [[ "${USE_MOCK_DATA}" == "1" ]]; then
    DATA_ARGS+=(
        --tokenizer-type NullTokenizer
        --vocab-size "${VOCAB_SIZE}"
        --mock-data
    )
else
    DATA_ARGS+=(
        --tokenizer-type "${TOKENIZER_TYPE}"
        --tokenizer-model "${TOKENIZER_MODEL}"
        --data-path "${DATA_PATH}"
    )
fi

TRAINING_ARGS=(
    --micro-batch-size "${MICRO_BATCH_SIZE}"
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --lr "${LR}"
    --train-iters "${TRAIN_ITERS}"
    --lr-decay-iters "${LR_DECAY_ITERS}"
    --lr-decay-style cosine
    --min-lr "${MIN_LR}"
    --weight-decay "${WEIGHT_DECAY}"
    --lr-warmup-iters "${LR_WARMUP_ITERS}"
    --clip-grad "${CLIP_GRAD}"
    --bf16
    --use-distributed-optimizer
    --overlap-param-gather
    --overlap-grad-reduce
    --no-gradient-accumulation-fusion
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size "${TP}"
    --pipeline-model-parallel-size "${PP}"
    --expert-model-parallel-size "${EP}"
    --expert-tensor-parallel-size "${ETP}"
    --context-parallel-size "${CP}"
)

if (( TP > 1 )); then
    MODEL_PARALLEL_ARGS+=(--sequence-parallel)
fi
if (( PP > 1 && NUM_LAYERS_PER_VIRTUAL_PIPELINE_STAGE > 0 )); then
    LOCAL_LAYERS=$((NUM_LAYERS / PP))
    if (( LOCAL_LAYERS % NUM_LAYERS_PER_VIRTUAL_PIPELINE_STAGE != 0 )); then
        echo "Layers per PP rank must be divisible by NUM_LAYERS_PER_VIRTUAL_PIPELINE_STAGE" >&2
        exit 2
    fi
    MODEL_PARALLEL_ARGS+=(
        --num-layers-per-virtual-pipeline-stage "${NUM_LAYERS_PER_VIRTUAL_PIPELINE_STAGE}"
    )
fi

LOGGING_ARGS=(
    --log-throughput
    --log-interval "${LOG_INTERVAL}"
    --save-interval "${SAVE_INTERVAL}"
    --eval-interval "${EVAL_INTERVAL}"
    --eval-iters "${EVAL_ITERS}"
)

if [[ -n "${SAVE_PATH}" ]]; then
    mkdir -p "${SAVE_PATH}"
    LOGGING_ARGS+=(--save "${SAVE_PATH}")
fi
if [[ -n "${LOAD_PATH}" ]]; then
    LOGGING_ARGS+=(--load "${LOAD_PATH}")
fi

PROFILE_ARGS=()
if [[ "${ENABLE_PROFILE}" == "1" ]]; then
    read -r -a PROFILE_RANK_ARGS <<< "${PROFILE_RANKS}"
    mkdir -p "${PROFILE_DIR}"
    PROFILE_ARGS+=(
        --profile
        --use-pytorch-profiler
        --profile-pp-semantics
        --profile-ranks "${PROFILE_RANK_ARGS[@]}"
        --profile-step-start "${PROFILE_STEP_START}"
        --profile-step-end "${PROFILE_STEP_END}"
        --profile-dir "${PROFILE_DIR}"
    )
fi

FL_SAVED_TENSOR_PROFILE_ARGS=()
if [[ "${ENABLE_FL_SAVED_TENSOR_PROFILE}" == "1" ]]; then
    read -r -a FL_SAVED_TENSOR_SCOPE_ARGS <<< "${FL_SAVED_TENSOR_PROFILE_SCOPES}"
    FL_SAVED_TENSOR_PROFILE_ARGS+=(
        --fl-saved-tensor-profile
        --fl-saved-tensor-profile-scopes "${FL_SAVED_TENSOR_SCOPE_ARGS[@]}"
        --fl-saved-tensor-profile-max-reports "${FL_SAVED_TENSOR_PROFILE_MAX_REPORTS}"
    )
fi

# FL Offload is disabled by default because every line in this array is
# commented. To enable it, uncomment the options from --fl-patch-te through
# --fl-activation-offload-stages-assignment.
#
# Current supported capture names:
#   LayerNormLinear: fused norm/QKV module input and internal norm output
#   GroupedLinear:   expert FC1 and FC2 inputs
#   swiglu:          weighted SwiGLU input
#   FlashAttention:  backward-saved Q, K, V, output, and softmax LSE (FA v2)
#   UnfusedAttention: Q, K, V, output, and attention probabilities
#   MLA:              MLA projection inputs
#   SharedExpert:     shared-expert FC1, SwiGLU, and FC2 inputs
#   MTP:              MTP eh_proj concatenated 2H input (requires MTP training)
#
# FL_OFFLOAD_MIB is the selected activation budget per activation group in MiB.
# Set it to 0 to collect captured-size statistics without copying activations.
# --fl-activation-offload-ratio is retained for CLI compatibility but is not
# used by the current selector; FL_OFFLOAD_MIB controls the selected amount.
# For staged PP/VPP overlap, keep four stages and assignment "0 1 2 3". D2H
# and H2D use one dedicated FL copy stream by default. Uncomment
# --fl-use-comm-stream only when intentionally testing the normal combined
# communication stream.
# An assignment of -1 skips that schedule insertion point. Repeating a stage
# keeps the current transfer progress without issuing that stage twice. For
# example, six transfer stages over eight VPP insertion points can use
# FL_OFFLOAD_STAGES=6 and FL_STAGE_ASSIGNMENT="-1 0 1 2 2 3 4 5".
FL_OFFLOAD_ARGS=(
    # --fl-patch-te
    # --fl-offload-modules LayerNormLinear GroupedLinear swiglu FlashAttention MTP
    # --fl-activation-offload-ratio 1.0
    # --fl-per-batch-offload-size "${FL_OFFLOAD_MIB}"
    # --fl-min-offloaded-tensor-size "${FL_MIN_TENSOR_BYTES}"
    # --fl-activation-offload-stages "${FL_OFFLOAD_STAGES}"
    # --fl-activation-offload-stages-assignment "${FL_STAGE_ASSIGNMENT_ARGS[@]}"
    # --fl-use-comm-stream
)

mkdir -p "${LOG_DIR}"
cd "${MEGATRON_PATH}"

echo "[aibench] world=${WORLD_SIZE} nodes=${NNODES} gpus_per_node=${GPUS_PER_NODE} node_rank=${NODE_RANK}"
echo "[aibench] model layers=${NUM_LAYERS} hidden=${HIDDEN_SIZE} ffn=${FFN_HIDDEN_SIZE} seq=${SEQ_LENGTH}"
echo "[aibench] parallel TP=${TP} PP=${PP} EP=${EP} ETP=${ETP} CP=${CP} DP=${DATA_PARALLEL_SIZE}"
echo "[aibench] attention=flash offload_args=${#FL_OFFLOAD_ARGS[@]} log=${LOG_DIR}/train_node${NODE_RANK}.log"
echo "[aibench] saved_tensor_profile=${ENABLE_FL_SAVED_TENSOR_PROFILE} scopes=${FL_SAVED_TENSOR_PROFILE_SCOPES}"

if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[aibench] command:'
    printf ' %q' \
        "${PYTHON_BIN}" -m torch.distributed.run \
        "${DISTRIBUTED_ARGS[@]}" \
        "${MEGATRON_PATH}/pretrain_gpt.py" \
        "${MODEL_ARGS[@]}" \
        "${MOE_ARGS[@]}" \
        "${DATA_ARGS[@]}" \
        "${TRAINING_ARGS[@]}" \
        "${MODEL_PARALLEL_ARGS[@]}" \
        "${LOGGING_ARGS[@]}" \
        "${FL_OFFLOAD_ARGS[@]}" \
        "${FL_SAVED_TENSOR_PROFILE_ARGS[@]}" \
        "${PROFILE_ARGS[@]}"
    printf '\n'
    exit 0
fi

"${PYTHON_BIN}" -m torch.distributed.run \
    "${DISTRIBUTED_ARGS[@]}" \
    "${MEGATRON_PATH}/pretrain_gpt.py" \
    "${MODEL_ARGS[@]}" \
    "${MOE_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${MODEL_PARALLEL_ARGS[@]}" \
    "${LOGGING_ARGS[@]}" \
    "${FL_OFFLOAD_ARGS[@]}" \
    "${FL_SAVED_TENSOR_PROFILE_ARGS[@]}" \
    "${PROFILE_ARGS[@]}" \
    2>&1 | tee "${LOG_DIR}/train_node${NODE_RANK}.log"
