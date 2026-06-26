#!/usr/bin/env bash

set -euo pipefail

MODE="${1:-compare}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/lsy/miniconda3/envs/fl_env/bin/python}"
LOG_DIR="${LOG_DIR:-/tmp/megatron_fl_offload_smoke}"
FL_OFFLOAD_MIB="${FL_OFFLOAD_MIB:-1}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python is not executable: ${PYTHON_BIN}" >&2
    exit 2
fi

case "${MODE}" in
    baseline|capture|offload|compare)
        ;;
    *)
        echo "Usage: $0 [baseline|capture|offload|compare]" >&2
        exit 2
        ;;
esac

mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"

export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export NVTE_ALLOW_NONDETERMINISTIC_ALGO="${NVTE_ALLOW_NONDETERMINISTIC_ALGO:-0}"

MODEL_ARGS=(
    --num-layers 2
    --hidden-size 512
    --ffn-hidden-size 2048
    --num-attention-heads 8
    --seq-length 1024
    --max-position-embeddings 1024
    --micro-batch-size 1
    --global-batch-size 1
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
    --pipeline-model-parallel-size 1
    --distributed-backend nccl
    --log-interval 1
    --eval-interval 1000
    --eval-iters 0
    --no-gradient-accumulation-fusion
    --attention-softmax-in-fp32
    --use-mcore-models
)

run_one() {
    local run_mode="$1"
    local -a offload_args=()

    if [[ "${run_mode}" == "capture" ]]; then
        offload_args=(
            --fl-patch-te
            --offload-modules LayerNormLinear
            --activation-offload-ratio 1.0
            --per-batch-offload-size 0
            --activation-offload-stages 1
            --activation-offload-stages-assignment 0 0 0 0
        )
    elif [[ "${run_mode}" == "offload" ]]; then
        offload_args=(
            --fl-patch-te
            --offload-modules LayerNormLinear
            --activation-offload-ratio 1.0
            --per-batch-offload-size "${FL_OFFLOAD_MIB}"
            --activation-offload-stages 1
            --activation-offload-stages-assignment 0 0 0 0
        )
    fi

    echo "[smoke] mode=${run_mode} log=${LOG_DIR}/${run_mode}.log"
    "${PYTHON_BIN}" -m torch.distributed.run \
        --standalone \
        --nproc_per_node 1 \
        pretrain_gpt.py \
        "${MODEL_ARGS[@]}" \
        "${offload_args[@]}" \
        2>&1 | tee "${LOG_DIR}/${run_mode}.log"
}

if [[ "${MODE}" == "compare" ]]; then
    run_one baseline
    run_one offload
    echo "[smoke] compare losses from ${LOG_DIR}:"
    grep -E "iteration.*lm loss|captured=.*selected=" \
        "${LOG_DIR}/baseline.log" "${LOG_DIR}/offload.log" || true
else
    run_one "${MODE}"
fi
