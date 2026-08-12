#!/usr/bin/env bash

set -euo pipefail

MODE="${1:-compare}"
MODEL_VARIANT="${MODEL_VARIANT:-standard}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/lsy/miniconda3/envs/fl_env/bin/python}"
LOG_DIR="${LOG_DIR:-/tmp/megatron_fl_offload_combined_smoke}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MTP_NUM_LAYERS="${MTP_NUM_LAYERS:-0}"
MTP_LOSS_SCALING_FACTOR="${MTP_LOSS_SCALING_FACTOR:-0.3}"
DEFAULT_FL_OFFLOAD_MIB=1
if [[ "${MODE}" == "memory" ]]; then
    DEFAULT_FL_OFFLOAD_MIB=8
fi
FL_OFFLOAD_MIB="${FL_OFFLOAD_MIB:-${DEFAULT_FL_OFFLOAD_MIB}}"
if [[ "${MODEL_VARIANT}" == "mla_shared" ]]; then
    ATTENTION_BACKEND="${ATTENTION_BACKEND:-unfused}"
    if [[ "${ATTENTION_BACKEND}" == "unfused" ]]; then
        FL_OFFLOAD_MODULES="${FL_OFFLOAD_MODULES:-MLA UnfusedAttention SharedExpert GroupedLinear swiglu}"
    else
        FL_OFFLOAD_MODULES="${FL_OFFLOAD_MODULES:-MLA FlashAttention SharedExpert GroupedLinear swiglu}"
    fi
else
    ATTENTION_BACKEND="${ATTENTION_BACKEND:-flash}"
    FL_OFFLOAD_MODULES="${FL_OFFLOAD_MODULES:-LayerNormLinear GroupedLinear swiglu}"
fi
FL_MIN_TENSOR_BYTES="${FL_MIN_TENSOR_BYTES:-1048576}"
FL_USE_COMM_STREAM="${FL_USE_COMM_STREAM:-0}"
TRACE_DIR="${TRACE_DIR:-${LOG_DIR}/trace_$$}"

if [[ "${FL_USE_COMM_STREAM}" != "0" && "${FL_USE_COMM_STREAM}" != "1" ]]; then
    echo "FL_USE_COMM_STREAM must be 0 or 1" >&2
    exit 2
fi

case "${ATTENTION_BACKEND}" in
    auto|flash|fused|unfused)
        ;;
    *)
        echo "ATTENTION_BACKEND must be auto, flash, fused, or unfused" >&2
        exit 2
        ;;
esac

case "${MODEL_VARIANT}" in
    standard|mla_shared)
        ;;
    *)
        echo "MODEL_VARIANT must be standard or mla_shared" >&2
        exit 2
        ;;
esac

if [[ "${NPROC_PER_NODE}" -ne 4 ]]; then
    echo "This smoke requires exactly 4 local GPUs (PP=2, EP=2)." >&2
    exit 2
fi
if [[ ! "${MTP_NUM_LAYERS}" =~ ^[0-9]+$ ]]; then
    echo "MTP_NUM_LAYERS must be a non-negative integer" >&2
    exit 2
fi
if (( MTP_NUM_LAYERS > 0 )) && [[ " ${FL_OFFLOAD_MODULES} " != *" MTP "* ]]; then
    FL_OFFLOAD_MODULES+=" MTP"
fi

case "${MODE}" in
    baseline|offload|compare|trace|memory)
        ;;
    *)
        echo "Usage: $0 [baseline|offload|compare|trace|memory]" >&2
        exit 2
        ;;
esac

mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"

export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-32}"
export NVTE_ALLOW_NONDETERMINISTIC_ALGO="${NVTE_ALLOW_NONDETERMINISTIC_ALGO:-0}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export NCCL_ALGO="${NCCL_ALGO:-Ring}"

MODEL_ARGS=(
    --num-layers 4
    --num-layers-per-virtual-pipeline-stage 1
    --hidden-size 512
    --ffn-hidden-size 2048
    --num-attention-heads 8
    --seq-length 1024
    --max-position-embeddings 1024
    --position-embedding-type rope
    --normalization RMSNorm
    --swiglu
    --micro-batch-size 1
    --global-batch-size 8
    --train-iters 3
    --lr 1.5e-4
    --min-lr 1.0e-5
    --lr-decay-style cosine
    --lr-decay-iters 3
    --weight-decay 0.01
    --clip-grad 1.0
    --seed 1234
    --tokenizer-type NullTokenizer
    --vocab-size 8192
    --mock-data
    --split 100,0,0
    --bf16
    --transformer-impl transformer_engine
    --attention-backend "${ATTENTION_BACKEND}"
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 2
    --expert-model-parallel-size 2
    --expert-tensor-parallel-size 1
    --num-experts 2
    --moe-router-topk 2
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 1e-2
    --moe-token-dispatcher-type alltoall
    --moe-grouped-gemm
    --moe-permute-fusion
    --overlap-moe-expert-parallel-comm
    --disable-bias-linear
    --distributed-backend nccl
    --log-interval 1
    --log-throughput
    --eval-interval 1000
    --eval-iters 0
    --deterministic-mode
    --no-gradient-accumulation-fusion
    --attention-softmax-in-fp32
    --use-mcore-models
)

if [[ "${MODEL_VARIANT}" == "mla_shared" ]]; then
    MODEL_ARGS+=(
        --multi-latent-attention
        --kv-lora-rank 512
        --qk-head-dim 128
        --qk-pos-emb-head-dim 64
        --v-head-dim 128
        --moe-ffn-hidden-size 256
        --moe-shared-expert-intermediate-size 512
    )
fi

if (( MTP_NUM_LAYERS > 0 )); then
    MODEL_ARGS+=(
        --mtp-num-layers "${MTP_NUM_LAYERS}"
        --mtp-loss-scaling-factor "${MTP_LOSS_SCALING_FACTOR}"
    )
fi

run_one() {
    local run_mode="$1"
    local enable_trace="${2:-false}"
    local enable_memory="${3:-false}"
    local -a offload_args=()
    local -a profile_args=()
    local -a memory_args=()
    local -a fl_offload_module_args=()

    read -r -a fl_offload_module_args <<< "${FL_OFFLOAD_MODULES}"

    if [[ "${run_mode}" == "offload" ]]; then
        offload_args=(
            --fl-patch-te
            --fl-offload-modules "${fl_offload_module_args[@]}"
            --fl-activation-offload-ratio 1.0
            --fl-per-batch-offload-size "${FL_OFFLOAD_MIB}"
            --fl-min-offloaded-tensor-size "${FL_MIN_TENSOR_BYTES}"
            --fl-activation-offload-stages 4
            --fl-activation-offload-stages-assignment 0 1 2 3
        )
        if [[ "${FL_USE_COMM_STREAM}" == "1" ]]; then
            offload_args+=(--fl-use-comm-stream)
        fi
    fi

    if [[ "${enable_trace}" == "true" ]]; then
        mkdir -p "${TRACE_DIR}"
        profile_args=(
            --profile
            --use-pytorch-profiler
            --profile-pp-semantics
            --profile-ranks 0 1 2 3
            --profile-step-start 2
            --profile-step-end 3
            --profile-dir "${TRACE_DIR}"
        )
    fi

    if [[ "${enable_memory}" == "true" ]]; then
        memory_args=(--fl-measure-training-memory)
    fi

    echo "[combined-smoke] mode=${run_mode} variant=${MODEL_VARIANT} "\
         "attention_backend=${ATTENTION_BACKEND} "\
         "comm_stream=${FL_USE_COMM_STREAM} "\
         "log=${LOG_DIR}/${run_mode}.log"
    "${PYTHON_BIN}" -m torch.distributed.run \
        --standalone \
        --nproc_per_node "${NPROC_PER_NODE}" \
        pretrain_gpt.py \
        "${MODEL_ARGS[@]}" \
        "${offload_args[@]}" \
        "${profile_args[@]}" \
        "${memory_args[@]}" \
        2>&1 | tee "${LOG_DIR}/${run_mode}.log"
}

if [[ "${MODE}" == "compare" ]]; then
    run_one baseline
    run_one offload
    if grep -q "staged .* was incomplete" "${LOG_DIR}/offload.log"; then
        echo "[combined-smoke] staged copy fallback detected" >&2
        exit 1
    fi
    echo "[combined-smoke] compare losses from ${LOG_DIR}:"
    grep -E "iteration.*lm loss|captured=.*selected=" \
        "${LOG_DIR}/baseline.log" "${LOG_DIR}/offload.log" || true
elif [[ "${MODE}" == "trace" ]]; then
    echo "[combined-smoke] trace_dir=${TRACE_DIR}"
    run_one offload true
    if grep -q "staged .* was incomplete" "${LOG_DIR}/offload.log"; then
        echo "[combined-smoke] staged copy fallback detected" >&2
        exit 1
    fi
    "${PYTHON_BIN}" examples/fl_offload/validate_trace.py \
        --trace-dir "${TRACE_DIR}" \
        --stages 4 \
        --analyze-overlap
elif [[ "${MODE}" == "memory" ]]; then
    echo "[combined-smoke] memory comparison with ${FL_OFFLOAD_MIB} MiB offload budget"
    run_one baseline false true
    run_one offload false true
    if grep -q "staged .* was incomplete" "${LOG_DIR}/offload.log"; then
        echo "[combined-smoke] staged copy fallback detected" >&2
        exit 1
    fi
    "${PYTHON_BIN}" examples/fl_offload/compare_memory.py \
        --baseline-log "${LOG_DIR}/baseline.log" \
        --offload-log "${LOG_DIR}/offload.log" \
        --warmup-iters 1 \
        --min-reduction-mib 1
else
    run_one "${MODE}"
fi
