# 独立训练显存估算模型

`tools/standalone_memory_model.py` 是一个只依赖 Python 标准库的理论估算器。它不导入
PyTorch、Megatron、Transformer Engine，也不创建模型或执行训练。输入模型结构、并行切分、
batch 和流水调度后，它自动推导峰值在途激活数量，并估算单个最重设备的峰值显存。

脚本顶部集中提供 `MODEL_CONFIG`、`PARALLEL_CONFIG`、`PRECISION_CONFIG` 和
`PLANNING_CONFIG` 四组变量。修改这些变量后可直接运行：

```bash
python tools/standalone_memory_model.py
```

`Per-layer saved activations`、`Module tensors` 和 `FL offload estimate` 使用 MB，便于与
运行时首条 `[FL offload] captured=... MiB` 日志逐项核对；整体 `Peak device memory` 使用
GiB，便于判断设备容量。

命令行选项仍然保留，用于临时覆盖脚本变量或批量扫描配置。例如：

```bash
python tools/standalone_memory_model.py --hidden-size 8192 --pipeline-parallel 4
```

直接运行的规划配置默认预留 2560 MiB transient workspace，并增加 26.5% allocator 余量。
`Peak device memory` 中的 `total` 是 first/last/middle PP stage 候选值中的最大值。
默认值按 16 卡 H800 实测校准：transient workspace 覆盖 TE、Grouped GEMM、
FlashAttention 和 NCCL 等未逐项建模的峰值 allocated，allocator 余量再覆盖 reserved 与
allocated 的缓存、碎片差异。不同软件栈和模型形状仍需要重新校准这两个值。

流水调度相关变量位于 `PARALLEL_CONFIG`：

```python
"world_size": 16,
"pipeline_schedule": "interleaved-1f1b",
"layers_per_virtual_pipeline_stage": 1,
"microbatch_group_size_per_vp_stage": 0,
"overlap_moe_expert_parallel_comm": True,
"inflight_microbatches": 0,
```

这里的 `layers_per_virtual_pipeline_stage=1` 与训练参数
`--num-layers-per-virtual-pipeline-stage 1` 含义一致。若每个 PP rank 有 4 层，则自动得到
4 个 VPP chunk。`inflight_microbatches=0` 表示启用调度推导；设为正数则改为手动覆盖。

## 1. 输出组成

估算器分别报告：

- 模型参数；
- 梯度；
- FP32 master parameter；
- 优化器状态；
- backward 所需 saved activation；
- embedding/output/logits 等端点激活；
- 通信缓冲区；
- 算子 workspace；
- Offload GPU landing buffer；
- allocator 碎片及未建模余量；
- 上述各项之和与设备剩余空间。

默认配置启用 `use_distributed_optimizer`，与训练参数
`--use-distributed-optimizer` 对应。模型自动计算：

```text
DP        = world_size / (TP * PP * CP)
expert-DP = world_size / (ETP * EP * PP)
```

普通并行网格与 Expert 并行网格分别构造，因此不要求 `DP` 能被 `EP` 整除。需要满足的
约束是 `TP*PP*CP` 和 `ETP*EP*PP` 均不超过且能够整除 `world_size`。

共享参数（Attention、Norm、Router、Embedding 等）的 FP32 master parameter 和 Adam
状态按 DP 切分，Expert 参数的相同状态按 expert-DP 切分。模型 BF16 参数和 FP32 main
gradient 仍完整驻留在本 rank；这对应 Megatron 的 BF16 分布式优化器理论开销
`6 + 12 / d` bytes/parameter，其中 Expert 参数的 `d` 使用 expert-DP。

关闭自动模式后，参数、梯度和优化器状态的 shard factor 仍可显式输入。例如只按统一的
shard factor 切分 master parameter/optimizer state 时使用：

```text
--no-use-distributed-optimizer
--parameter-bytes 2
--gradient-bytes 4
--master-parameter-bytes 4
--optimizer-state-bytes 8
--parameter-shards 1
--gradient-shards 1
--optimizer-shards 2
```

不能用一个 `optimizer_shards=DP` 近似带 EP 的分布式优化器，否则会把 Expert 状态也错误
地按 DP 切分。例如 `world_size=8, PP=2, ETP=1, EP=4` 时，Expert 状态使用
`expert-DP=1`，并未分片；该结果独立于普通网格中的 TP、CP 和 DP。

## 2. 参数公式

令 hidden size 为 `H`，dense、routed expert、shared expert 的中间宽度分别为
`Fd`、`Fm`、`Fs`，词表为 `V`，Attention head 数为 `A`，KV head 数为 `K`，
head dimension 为 `D=H/A`，专家数为 `E`。

```text
embedding             = H * V
QKV + output proj     = H * (H + 2*K*D) + H * H
dense gated MLP       = 3 * H * Fd
MoE routed experts    = E * 3 * H * Fm
MoE shared expert     = 3 * H * Fs
router                = H * E
RMSNorm per layer     = 2 * H
```

全局参数按层数求和。单设备参数再根据 PP 层数分配、TP、EP、ETP 和端点 embedding/output
进行切分；报告 first/last/middle stage 中总显存最大的 stage。

报告同时输出模型级的 `active parameters per token`。它表示生成一个 token 时实际参与
前向计算的唯一参数量，对应模型名称中 `35B-A3B` 的 `A3B`，不是训练时保存的激活值显存。
对于 routed + shared expert 模型：

```text
active parameters
= embedding/output、attention、normalization、router 和 dense MLP 的全部参数
 + TopK * 单个 routed expert 参数 * MoE 层数
 + shared expert 参数 * MoE 层数
```

总参数中的 MoE 项使用全部 `E` 个专家，active parameters 中只使用每个 token 选中的
`TopK` 个专家。因此两者之差为：

```text
(E - TopK) * 单个专家参数 * MoE 层数
```

该模型级指标不随 TP、PP、EP、batch size、序列长度和 capacity padding 改变。并行配置只
影响每张卡持有多少参数，路由 padding 等配置则影响实际 FLOPs 和吞吐。Shared expert
始终参与每个 token 的计算，因此同时计入 total 和 active parameters。

MLA 使用 `Rq/Rkv/Dq/Dp/Dv` 分别表示 query/kv LoRA rank、QK 非位置维度、位置维度和
value 维度。未配置 query LoRA 时每层投影参数为：

```text
Q projection   = H * A * (Dq + Dp)
KV down/up     = H * (Rkv + Dp) + Rkv * A * (Dq + Dv)
output proj    = A * Dv * H
```

配置 `Rq` 时，Q projection 改为 `H*Rq + Rq*A*(Dq+Dp)`。这些公式替换普通 GQA 的
QKV 参数公式。

MTP 默认整体放在最后一个 PP/VPP stage。每个预测深度复用 embedding 和 output head，
不重复计算这两份参数；新增参数包括一个完整内部 Transformer 层以及：

```text
eh_proj               = 2 * H * H
enorm + hnorm + final = 3 * H
```

内部层为 MoE 时，总参数包含全部 routed experts，active parameters 只包含 TopK routed
experts；`mtp_layer_is_moe` 用于描述最后 decoder 层类型。MTP 参数只加入 last PP stage，
不会平均分摊到其他 stage。

## 3. 激活公式

FlashAttention 风格的每层 saved activation 分为：

```text
qkv_linear = LayerNormLinear input + ln_out
core_attn  = Q + K + V + O + FP32 softmax LSE
MLP/MoE    = FC1 input + activation input + FC2 input
```

MLA 模式下，`qkv_linear` 改为投影 backward 保存的 hidden input、`kv_up_input`，以及
可选的 `q_up_input`；开启 fused down projection 时还包含其 `ln_out`。FlashAttention
部分使用 MLA 实际维度：Q/K 为 `A/TP*(Dq+Dp)`，V/O 为 `A/TP*Dv`。

Shared expert 不经过 routed capacity，每个本地序列 token 都参与计算。其 saved activation
独立计为普通 TP MLP 的 FC1 输入、SwiGLU 输入和 FC2 输入，并加到每个 MoE 层上。

Attention projection 保存的输入 `O` 与 `core_attn` 中的 `O` 共享 storage，只计算一次。
非 Flash 路径可以通过 `--attention-score-buffers N` 增加 `N` 份
`[MBS, heads, seq, seq]` attention score/probability buffer。

MoE 固定 capacity 下，本地 routed token 估算为：

```text
router_tokens       = ceil(S / TP) * MBS
expert_capacity     = ceil(router_tokens * TopK * capacity_factor / experts)
local_routed_tokens = expert_capacity * ETP * experts
```

这是 All-to-All dispatcher 在 pad-to-capacity 路径中的实际形状：每个本地专家接收
`expert_capacity*ETP*EP` 个 token，而每个 rank 持有 `experts/EP` 个专家。因而 EP 在乘除后
抵消，但普通 TP 对 router 输入 token 数的切分不能忽略。

工具支持以下 `pipeline_schedule`：

- `1f1b`：无 VPP 的普通流水 1F1B；
- `interleaved-1f1b`：按 VPP schedule table 交错执行；
- `gpipe`：所有 forward 完成后再 backward；
- `manual`：使用 `inflight_microbatches` 手动指定。

自动模式首先计算：

```text
DP = world_size / (TP * PP * CP)
num_microbatches = global_batch_size / (micro_batch_size * DP)
```

随后逐 PP rank 模拟 warmup、steady 1F1B 和 cooldown。每次 forward 增加对应 VPP chunk
的一份 activation，每次 backward 将其释放，并记录每个 chunk 的最大并存数。对于
interleaved 1F1B，warmup virtual microbatch 数与当前 Megatron 调度一致：

```text
warmup = (PP - pp_rank - 1) * 2
       + (VPP_chunks - 1) * vp_microbatch_group_size
       + combined_moe_extra_warmup
```

报告中的 `activation_batches` 是最大驻留 VPP chunk 按本地总层数折算出的完整 activation
batch 数，可以是 `2.5` 这样的非整数；`by_chunk` 显示峰值时每个 VPP chunk 的实际份数。

模块级明细按照当前代码实际存在的显式 hook 拆分：

| `offload_modules` 名称 | 建模张量 |
|---|---|
| `LayerNormLinear` | QKV fused LayerNormLinear 的 `inputmat`、`ln_out` |
| `GroupedLinear` | Expert FC1 输入、Expert FC2 输入 |
| `swiglu` | Weighted SwiGLU 输入，即 Expert FC1 gated 输出 |
| `FlashAttention` | 普通 attention 或 MLA 的 Q、K、V、O、softmax LSE |
| `UnfusedAttention` | 普通 attention 或 MLA 的 Q、K、V、O、完整 attention probability |
| `MLA` | q/kv down projection 共享输入、q/kv up projection 输入 |
| `SharedExpert` | Shared FC1 输入、普通 SwiGLU 输入、Shared FC2 输入 |
| `MTP` | 每个预测深度中 `eh_proj` 的 `[S,B,2H]` 拼接输入 |

Dense MLP 和 MLA 内部 LayerNorm 的输入目前仍只估算、不标记为可 Offload。MLA、
SharedExpert、FlashAttention 和 UnfusedAttention 已接入 pack/unpack；由于本地环境无法加载 CUDA 版
TE/FlashAttention，仍需在目标 GPU 节点完成 baseline/offload 梯度与显存验证后再视为生产可用。

MTP 的 `eh_proj_input` 由独立 `MTP` scope 捕获；其内部 Transformer/MoE 层仍使用已有的
`MLA`、`FlashAttention`/`UnfusedAttention`、`GroupedLinear`、`swiglu` 和
`SharedExpert` 名称。建模参数为 `mtp_num_layers` 和 `mtp_layer_is_moe`，例如：

```text
--mtp-num-layers 1 --mtp-layer-is-moe --offload-modules MTP GroupedLinear swiglu
```

第一版运行时覆盖标准训练用 `MLASelfAttention`，包括 q/kv down projection、可选 q up
projection、kv up projection 和 FlashAttention v2 backward 保存的 Q/K/V/O/LSE。实验性的
absorbed MLA 不在本次范围。Shared Expert 覆盖 TE FC1/FC2 输入和融合 SwiGLU 输入；普通
forward 与 `--moe-shared-expert-overlap` 的专用 forward 路径都已放置 scope，但 overlap 路径
同样需要 GPU trace 验证。

当 `--attention-backend unfused` 时，模型不再使用 FlashAttention 的 LSE 近似，而是计算：

```text
Q/K bytes = MBS * S/CP * heads/TP * (qk_head_dim + qk_pos_emb_head_dim) * dtype_bytes
V/O bytes = MBS * S/CP * heads/TP * v_head_dim * dtype_bytes
one probability bytes = MBS * heads/TP * (S/CP)^2 * dtype_bytes
saved probability bytes = 2 * one probability bytes
```

当前 unfused 运行时在 TE `UnfusedDotProductAttention.forward` 内安装局部 saved-tensor
pack/unpack，只在 `record()` 活跃且选择了 `UnfusedAttention` 时生效。TE 2.14 实测即使
attention dropout 为 0，softmax backward 与 context BMM backward 仍分别保留一份独立的
probability storage，因此默认按两份计算。可用
`--unfused-attention-probability-buffers` 覆盖该数量；不同 TE、PyTorch 或 dropout 路径仍应以
运行时 `captured` 日志为准。

`UnfusedAttention.attention_probs` 的两个内部 backward 保存点都由局部 saved-tensor hook
接管，因此允许主动 `resize_(0)`；`MTP.eh_proj_input` 是 `torch.cat` 新建且只由 patched
projection backward 使用的完整 storage，也允许主动释放。Q/K/V/O、`MLA` 和
`SharedExpert` 当前仍采用保守 source release。保守路径完成 D2H 后只删除 FL 自己持有的 tensor 引用：没有其他别名时分配会自然
释放，存在框架合法别名时会保留该分配以保证后续 GEMM 正确。因此建模中的 selected 表示传输
和可替换的逻辑字节数，不保证 `max_allocated` 等量下降，需以 GPU memory trace 校准。

对应 MLA + routed/shared experts 的建模示例：

```bash
python tools/standalone_memory_model.py \
  --attention-backend unfused \
  --multi-latent-attention \
  --kv-lora-rank 512 \
  --qk-head-dim 128 \
  --qk-pos-emb-head-dim 64 \
  --v-head-dim 128 \
  --ffn-hidden-size 11264 \
  --moe-ffn-hidden-size 1408 \
  --moe-shared-expert-intermediate-size 2816 \
  --experts 64 \
  --offload-modules MLA UnfusedAttention SharedExpert GroupedLinear swiglu
```

未配置 query LoRA 时保持 `--q-lora-rank 0`；需要 query LoRA 时设置实际 rank。共享专家
始终计入 active parameters，routed experts 仅按 TopK 计入 active parameters。

`--offload-modules` 决定 eligible captured 集合，`--offload-min-tensor-bytes` 对应运行时的
最小张量阈值，`--offload-mib-per-layer` 对应一个 activation group/layer 的完整预算。选择
规则与当前运行时一致：只有 eligible captured 不小于完整预算时才 selected 该预算，否则
selected 为 0；不会自动把预算缩小到 captured。

模型从每种本地层的驻留激活中扣除 selected，并自动加回一份相同大小的持久 H2D landing
buffer。`--offload-gpu-buffer-mib` 仅用于显式设置更大的保守 reserve。为避免高估收益，
allocator/未建模余量始终按同一配置的未 Offload 基线计算，不会因为 selected 增大而同比
下降。

combined 1F1B 通常不会 offload 最后一个 PP rank 的最后一个 model chunk。模型根据该 rank
的 `by_chunk` 驻留数量排除这部分激活。显式选择 `MTP` 且 `mtp_num_layers>0` 时例外：运行时
会记录最后 chunk，使 `MTP.eh_proj_input` 可被恢复。最后 decoder chunk 与所有 MTP 深度位于
同一个 activation group，只共享一次固定预算，建模不会分别重复扣减。当前每个 VPP chunk
一层时是精确计算；若一个 VPP chunk 包含多层，其他 chunk 的模块 captured 和预算仍以
“每层等效值”近似，最终应通过运行时首条 `[FL offload] captured=... selected=...` 日志校准。

最终 MTP chunk 的 forward 与对应 backward 之间没有普通 chunk 所具有的完整 combined-step
预取距离。当前功能优先实现会在其 forward 后完成 D2H，并在匹配 backward 前完成 H2D；其他
chunk 继续使用四阶段 issue 调度。因此 MTP 特有 group 的建模显存收益仍成立，但这部分拷贝
暂时不计为可被计算/通信完全隐藏，吞吐影响需要通过 GPU trace 单独评估。

## 4. 当前均衡 MoE 示例

```bash
python tools/standalone_memory_model.py \
  --layers 8 \
  --hidden-size 4096 \
  --ffn-hidden-size 2048 \
  --vocab-size 8192 \
  --sequence-length 1024 \
  --micro-batch-size 1 \
  --attention-heads 64 \
  --kv-heads 8 \
  --experts 8 \
  --topk 2 \
  --moe-layers 8 \
  --capacity-factor 1 \
  --gated-mlp \
  --untied-embeddings \
  --world-size 4 \
  --global-batch-size 8 \
  --tensor-parallel 1 \
  --pipeline-parallel 2 \
  --expert-parallel 2 \
  --pipeline-schedule interleaved-1f1b \
  --layers-per-virtual-pipeline-stage 1 \
  --overlap-moe-expert-parallel-comm \
  --use-distributed-optimizer \
  --communication-buffer-factor 0.1 \
  --workspace-mib 256 \
  --overhead-percent 10 \
  --device-memory-gib 80
```

该配置的单层保存激活为：

```text
qkv_linear      16.00 MiB
core_attention  18.25 MiB
expert_mlp      40.00 MiB
total           74.25 MiB
```

这与 2026-07-30 的 balanced saved-tensor scope 实测一致。加入 `ln_out` 后，模拟当前
56 MiB/层的显式 Offload 时，在上述命令后增加：

```text
--offload-modules LayerNormLinear GroupedLinear swiglu \
--offload-mib-per-layer 56
```

当前默认大模型配置下，每个 MoE 层的模块明细为：

```text
LayerNormLinear   0.250 GiB  (inputmat 0.125 + ln_out 0.125)
FlashAttention    0.283 GiB
GroupedLinear     0.750 GiB  (FC1 input 0.500 + FC2 input 0.250)
swiglu            0.500 GiB
```

例如估算三个已支持模块、每层 1.5 GiB 预算：

```bash
python tools/standalone_memory_model.py \
  --offload-modules LayerNormLinear GroupedLinear swiglu \
  --offload-mib-per-layer 1536
```

增加 `--json` 可以输出适合脚本处理的完整 JSON。

## 5. 参考 H800 容量校准

脚本顶部的默认配置与参考训练脚本保持一致：8 层、`H=8192`、`FFN=4096`、序列长度
8192、16 专家、TopK=4、`TP=1/PP=2/EP=4`、GQA query groups=8、GBS=256，并使用
分布式优化器。注意 `kv_heads` 对应训练脚本的 `NUM_QUERY_GROUPS`，不能误填成 attention
heads 64；后者会同时高估 QKV 参数和保存激活。

该配置的容量规划结果约为：

```text
analytical subtotal                59.405 GiB
allocator_and_unmodeled_overhead   14.851 GiB
planned max reserved               74.257 GiB
```

参考实测的全局最大值为 `75918 MiB = 74.139 GiB`，误差约 0.12 GiB。这里的 25% 是
面向 `max reserved` 的经验余量，不表示有一块固定大小的单一张量。

## 6. 边界

这是容量规划模型，不是精确 allocator 模拟器。以下内容必须通过输入显式保守估计：

- cuBLAS/FlashAttention/Grouped GEMM workspace；
- 通信 bucket 和重叠导致的临时副本；
- CUDA graph 私有内存池；
- FP8 transpose/scale/cache 等额外状态；
- allocator 碎片；
- 自定义流水调度与内置 `1f1b`/`interleaved-1f1b`/`gpipe` 的差异；
- 路由不均衡时高负载 rank 的实际 routed token 数。

因此最终是否可运行仍需真实峰值测量验证，但各项分开输出后可以明确校正误差来源。
