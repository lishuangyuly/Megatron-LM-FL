#!/bin/bash
MEGATRON_PATH=/home/lsy/zhiyuan/Megatron-LM-FL
export PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6000"}
NNODES=${SLURM_NNODES:-"1"}
NODE_RANK=${RANK:-"0"}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

# CHECKPOINT_PATH=$1
TOKENIZER_MODEL=/home/lsy/datasets
DATA_PATH=/home/lsy/datasets/wikidataset/my-bert_text_sentence

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

MODEL_ARGS=(
    --use-mcore-models
    --disable-bias-linear
    --seq-length 1024
    --max-position-embeddings 32768
    --num-layers 8
    --hidden-size 4096
    --ffn-hidden-size 2048
    --num-attention-heads 64
    --init-method-std 0.01
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --position-embedding-type rope
    --swiglu
    --untie-embeddings-and-output-weights
    --group-query-attention
    --num-query-groups 8
    --no-masked-softmax-fusion
    --no-position-embedding
    --rotary-base 1000000
)

MOE_ARGS=(
    --num-experts 8
    --moe-router-topk 2
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 1e-2
    --moe-grouped-gemm
    --moe-permute-fusion
    --moe-token-dispatcher-type alltoall
)

DATA_ARGS=(
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model ${TOKENIZER_MODEL}
    --data-path $DATA_PATH
    --split 98,2,0
)

TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 256
    --lr 1e-4
    --train-iters 5
    --lr-decay-iters 10000
    --lr-decay-style cosine
    --min-lr 1.0e-5
    --weight-decay 0.1
    --lr-warmup-iters 500
    --clip-grad 1.0
    --bf16
    --overlap-param-gather
    --overlap-grad-reduce
    # --use-flash-attn
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 4
    --expert-model-parallel-size 2
    --expert-tensor-parallel-size 1
    --context-parallel-size 1
    --num-layers-per-virtual-pipeline-stage 1
    --use-distributed-optimizer
    --sequence-parallel
)

LOGGING_ARGS=(
    --log-throughput
    --log-interval 1 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 5 \
    # --save $CHECKPOINT_PATH \
    # --load $CHECKPOINT_PATH \
    --tensorboard-dir "${CHECKPOINT_PATH}/tensorboard" \
    --no-load-optim \
    --no-load-rng \
    --no-save-optim
)

# if [ -n "${WANDB_API_KEY}" ]; then
#     LOGGING_ARGS+=(
#         --wandb-project ${WANDB_PROJECT:-"Mixtral"}
#         --wandb-exp-name ${WANDB_NAME:-"Mixtral_8x7B"}
#     )
# fi

TORCH_PROFILE_ARGS=(
    --profile
    --use-pytorch-profiler
    --profile-ranks 0 1 
    --profile-step-start 3
    --profile-step-end 4
    --profile-dir ./torch_prof
    --pytorch-profiler-collect-callstack
)


torchrun ${DISTRIBUTED_ARGS[@]} $MEGATRON_PATH/pretrain_gpt.py \
    ${MODEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]} \
    ${TORCH_PROFILE_ARGS[@]} \
