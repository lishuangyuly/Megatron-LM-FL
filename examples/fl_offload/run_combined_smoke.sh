#!/usr/bin/env bash

set -euo pipefail

MODE="${1:-compare}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/lsy/miniconda3/envs/fl_env/bin/python}"
LOG_DIR="${LOG_DIR:-/tmp/megatron_fl_offload_combined_smoke}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
DEFAULT_FL_OFFLOAD_MIB=1
if [[ "${MODE}" == "memory" ]]; then
    DEFAULT_FL_OFFLOAD_MIB=8
fi
FL_OFFLOAD_MIB="${FL_OFFLOAD_MIB:-${DEFAULT_FL_OFFLOAD_MIB}}"
FL_USE_COMM_STREAM="${FL_USE_COMM_STREAM:-0}"
TRACE_DIR="${TRACE_DIR:-${LOG_DIR}/trace_$$}"

if [[ "${FL_USE_COMM_STREAM}" != "0" && "${FL_USE_COMM_STREAM}" != "1" ]]; then
    echo "FL_USE_COMM_STREAM must be 0 or 1" >&2
    exit 2
fi

if [[ "${NPROC_PER_NODE}" -ne 4 ]]; then
    echo "This smoke requires exactly 4 local GPUs (PP=2, EP=2)." >&2
    exit 2
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
    --attention-backend unfused
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

run_one() {
    local run_mode="$1"
    local enable_trace="${2:-false}"
    local enable_memory="${3:-false}"
    local -a offload_args=()
    local -a profile_args=()
    local -a memory_args=()

    if [[ "${run_mode}" == "offload" ]]; then
        offload_args=(
            --fl-patch-te
            --fl-offload-modules LayerNormLinear GroupedLinear swiglu
            --fl-activation-offload-ratio 1.0
            --fl-per-batch-offload-size "${FL_OFFLOAD_MIB}"
            --fl-min-offloaded-tensor-size 1048576
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

    echo "[combined-smoke] mode=${run_mode} comm_stream=${FL_USE_COMM_STREAM} "\
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
