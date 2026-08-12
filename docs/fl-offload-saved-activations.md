# FL Offload 训练激活值保存与 Hook 对应表

本文先回答两个不同的问题：

1. 不考虑 FL Offload 时，当前 Megatron + Transformer Engine 正常训练为了 backward 会保存或保留哪些激活值；
2. 上述激活值中，当前 FL Offload 的显式 Hook 实际能手动 offload 哪些。

这里的“输入/输出”始终相对于表中所写的模块，而不是相对于整个 Transformer 层。
参数、参数的 FP8 副本和纯 workspace 不计入激活值。`ctx.save_for_backward`/TE
`prepare_for_saving` 明确保存的张量与仅由 autograd 图维持生命周期的张量也分开列出。

## 1. 当前默认路径与估算配置

FL Offload 的两个 smoke 入口默认使用：

```text
--attention-backend flash
```

可通过 `ATTENTION_BACKEND=auto|flash|fused|unfused` 覆盖。该默认值只属于 FL Offload
示例，不修改 Megatron 全局的 `AttnBackend.auto`。

下文大小示例采用：

```text
SEQ=4096, MBS=1
HS=8192, FFN_HS=4096
H=64, TP=1, ETP=1
E=16, EP=4, TopK=2
BF16 普通激活，FlashAttention
```

一个 EP group 中有 4 个 rank。dropless 且路由近似均衡时，每个 rank 收到的 expert
assignment 行数期望值为：

```text
R = (4 ranks * 4096 tokens/rank * TopK 2) / 4 ranks = 8192
```

这不是“每个 rank 的 token 简单乘 EP”。每个原始 token 会产生 `TopK` 个 expert
assignment，dispatcher 再按 expert owner 重分布；`R` 是重分布后当前 EP rank 上所有本地
expert 的 assignment 总行数。实际 `R` 由路由结果决定。

## 2. 正常训练明确保存的激活值

下表按前向执行顺序列出当前 BF16 + FlashAttention + TE GroupedMLP 路径中，为 backward
明确保存的主要激活值。大小按单层、单个有效 microbatch/activation group 估算。

| 前向阶段/保存方 | 为 backward 保存的张量 | 相对该模块的输入/输出 | backward 用途 | BF16 示例大小 | FP8 的影响 |
| --- | --- | --- | --- | ---: | --- |
| fused input RMSNorm + QKV，TE `_LayerNormLinear` | `inputmat` | **模块输入**，即进入 input RMSNorm 的 hidden state | norm backward、QKV wgrad | `[4096,8192]` BF16，64 MiB | FP8 GEMM 下通常仍保存原始 BF16 输入，仍约 64 MiB |
| 同一 `_LayerNormLinear` | `ln_out` | **模块内部 RMSNorm 输出**，同时是 QKV GEMM 输入 | QKV wgrad/相关 TE backward | BF16 时 64 MiB | 可能保存为 FP8 quantized storage，数据体约 32 MiB |
| 同一 `_LayerNormLinear` | `mu`/`rsigma` | **模块内部 norm 统计量**，不是模块输出 | norm backward | RMSNorm 主要是 FP32 `rsigma`，约 0.016 MiB | 通常保持 FP32 |
| FlashAttention backward context | RoPE 后的 Q、K、V | **FlashAttention 输入**；来自 QKV projection 输出 | 计算 dQ/dK/dV | 见下文 MHA/GQA 说明 | 普通 FP8 GEMM 不会自动改变；只有 FP8 DPA 才可能保存 FP8 版本 |
| FlashAttention backward context | `O` | **FlashAttention 输出**，也是 attention projection 输入 | attention backward | `[4096,8192]` BF16，64 MiB | 普通 FP8 GEMM 下通常仍为 BF16；FP8 DPA 另行决定 |
| FlashAttention backward context | `softmax_lse` | **FlashAttention 内部输出**，不是完整 softmax probability | 重建 softmax 并反传 | `[MBS,H,SEQ]` FP32，约 1 MiB | 保持 FP32 |
| FlashAttention backward context | RNG state、sequence metadata | **内部元数据** | dropout 与变长序列 backward | 很小 | 不受 GEMM FP8 影响 |
| attention output projection，TE `_Linear` | `saved_inputmat` | **模块输入**，即同一个 FlashAttention 输出 `O` | projection wgrad | BF16 下通常只是再次引用上述 `O`，不新增 64 MiB storage | delayed FP8 可能另存约 32 MiB 的量化 storage；非 delayed FP8 当前会保存原始 `O` 以避免该副本 |
| pre-MLP RMSNorm | norm 输入及 `rsigma` | **模块输入**和内部统计量 | norm backward | 主输入 BF16 约 64 MiB | 输入通常仍为 BF16；统计量保持 FP32 |
| router gating linear/autograd | router input | **Router linear 输入**，即 pre-MLP norm 输出 | router weight gradient | BF16 约 64 MiB | Router 稳定性路径通常仍使用 BF16/FP32，不应按 FP8 自动减半 |
| expert FC1，TE `_GroupedLinear` | `inputmats` | **FC1 模块输入**，即 dispatcher 输出的 routed hidden states | FC1 wgrad | `[R,8192]` BF16，期望 128 MiB | FP8 GroupedLinear 可保存 quantized storage，数据体理论约 64 MiB |
| `WeightedSwiGLUFunction` | `input_for_backward` | **SwiGLU 模块输入**，即 expert FC1 的 gated 输出 | SiLU、乘法及可选权重 backward | `[R,2*FFN_HS]=[R,8192]` BF16，期望 128 MiB | 仅开启 FP8 GEMM 仍是 BF16；需单独启用 FP8 activation input-store 才理论约 64 MiB |
| expert FC2，TE `_GroupedLinear` | `inputmats` | **FC2 模块输入**，即 SwiGLU 输出 | FC2 wgrad | `[R,FFN_HS]=[R,4096]` BF16，期望 64 MiB | FP8 GroupedLinear 数据体理论约 32 MiB |
| fused cross entropy（仅最后 PP stage） | `exp_logits`、target mask/index | **loss 内部中间值/输入元数据** | cross-entropy backward | `exp_logits` 为 FP32，大小依 vocab/TP 而定 | 通常保持 FP32 |

### 2.1 输入和输出关系

几个最容易混淆的关系如下：

```text
layer hidden state
  -> [LayerNormLinear 的输入 inputmat]
  -> RMSNorm
  -> [LayerNormLinear 内部输出 ln_out / QKV GEMM 输入]
  -> QKV projection + RoPE
  -> [FlashAttention 输入 Q/K/V]
  -> FlashAttention
  -> [FlashAttention 输出 O / attention projection 输入 saved_inputmat]

routed hidden states
  -> [GroupedLinear FC1 输入 inputmats]
  -> expert FC1
  -> [Weighted SwiGLU 输入 input_for_backward]
  -> SwiGLU
  -> [GroupedLinear FC2 输入 inputmats]
  -> expert FC2
```

所以：

- QKV projection 的融合输出不是当前 `LayerNormLinear` Hook 捕获的值；当前 Hook 捕获的是
  QKV projection 之前、input RMSNorm 之前的模块输入。
- expert FC1 的输出由 SwiGLU 作为**输入**保存。
- SwiGLU 的输出由 expert FC2 的 GroupedLinear 作为**输入**保存。
- TE Linear/GroupedLinear 为 wgrad 保存的是模块输入，不是它们自己的 GEMM 输出。

### 2.2 两个 backward 保存方不等于两份物理张量

FlashAttention backward 和 attention projection backward 都需要 `O`，所以它会出现在两个
autograd 节点的保存列表中。但 BF16 下，TE `_Linear.saved_inputmat` 直接引用传入的 `O`；
`save_for_backward` 增加的是对同一底层 storage 的生命周期引用，不会再复制一份 64 MiB
张量。因此显存统计应按一块 64 MiB storage 计算，不能把表中两行相加为 128 MiB。

FP8 时需要分情况：

- 非 delayed FP8 recipe 且 TE 版本满足要求时，当前 `attention.py` 对 `linear_proj` 调用
  `set_save_original_input()`。projection backward 继续引用原始 `O`，避免额外保存量化副本。
- delayed FP8 recipe 不走上述设置。FlashAttention 仍可能保存 BF16 `O`，projection 另外为
  wgrad 保存 FP8 quantized storage，此时确实存在两块不同的物理数据，约为 64 MiB 加
  32 MiB，并另有 scale/metadata。
- 若单独启用 FP8 DPA，FlashAttention 自己保存哪种 `O` 还要由 DPA recipe 决定，不能套用
  普通 FP8 GEMM 的结论。

当前第一版 FlashAttention v2 Patch 已捕获这块 `O`。TE `_Linear` 只有在 projection 输入
与 Flash `O` 精确共用同一 data pointer 和字节范围时才复用该 pack；独立 allocation 仍按
TE 原逻辑保存，不会被重复计入 Flash `captured`。

### 2.3 Q/K/V 大小必须区分 MHA 与 GQA

`head_dim = HS / H = 8192 / 64 = 128`。若使用 BF16：

| attention 形式 | Q | K | V | Q+K+V |
| --- | ---: | ---: | ---: | ---: |
| MHA，`num_query_groups=64` | 64 MiB | 64 MiB | 64 MiB | 192 MiB |
| GQA，`num_query_groups=8` | 64 MiB | 8 MiB | 8 MiB | 80 MiB |

因此“QKV 192 MiB”与“K 8 MiB”不能同时属于同一个配置。训练命令必须明确
`--num-query-groups`，否则估算表不能确定 K/V 大小。

### 2.4 FlashAttention 不保存完整 attention matrix

当前 FlashAttention backward context 明确保存 Q、K、V、O、`softmax_lse`、RNG state
和 sequence metadata。它不会物化并保存 `[B,H,SEQ,SEQ]` 的完整 softmax probability。
因此 fused/unfused softmax 中保存的完整 `softmax_results` 不能计入当前 Flash 路径。

### 2.5 MLA unfused 保存完整 probability

TE unfused attention 由 QK BMM、softmax/dropout 和 PV BMM 组成。其 backward 保存 Q、K、V
以及完整 `[MBS, heads/TP, SEQ, SEQ]` attention probability；TE 2.14 实测 softmax backward
和 PV BMM backward 各保留一份独立 probability storage，即这一项按两份计算。attention
projection 还会保存 unfused attention 输出 O。真实 MLA 的 Q/K head dimension 是
`qk_head_dim + qk_pos_emb_head_dim`，V/O 使用 `v_head_dim`，两者可以不同。

## 3. 由 autograd 图保留、但不是上述显式 save 表的值

正常训练还会让部分张量保持存活，例如：

| 阶段 | 保留的逻辑值 | 输入/输出关系 | 说明 |
| --- | --- | --- | --- |
| attention BDA | projection 输出、residual、dropout 状态 | projection 的**输出**、BDA 的**输入** | 具体保存形式取决于 fusion 与 dropout 配置，不等同于 TE Linear 的 `saved_inputmat` |
| MoE router/dispatcher | scores、top-k indices、routing map、sorted indices、split metadata | Router/dispatcher 的**输出** | 大多是 FP32、整数或布尔元数据；用于路由 backward 和逆置换 |
| token dispatcher/combine | routed token 与通信缓冲的必要引用 | dispatcher/combine 的**输入或输出** | 生命周期受 EP 通信实现与 overlap 配置影响，不能只按模型形状固定估算 |
| MLP BDA | expert combine 输出、residual、dropout 状态 | combine 的**输出**、BDA 的**输入** | 当前没有 FL Hook |

这些值会影响训练峰值显存，但不能因为它们出现在“activation memory breakdown”中，就认为
当前 FL `pack_hook` 能捕获它们。

## 4. 当前 Hook 可以手动 offload 的子集

当前代码修改 TE `_LayerNormLinear`、TE `_GroupedLinear`、`WeightedSwiGLUFunction`、
FlashAttention v2 fixed/varlen Function、TE unfused attention 局部保存边界，以及用于识别
attention `O` 精确别名的 TE `_Linear`。
实际可捕获项如下：

| FL `op_name` | 当前捕获对象 | 输入/输出的准确含义 | BF16 示例大小 | 当前状态 |
| --- | --- | --- | ---: | --- |
| `LayerNormLinear` | `inputmat` | fused input RMSNorm + QKV **模块输入** | 64 MiB | BF16 已验证 |
| `LayerNormLinear` | `ln_out` | fused RMSNorm **模块内部输出**，也是 QKV GEMM 输入 | 64 MiB | BF16 GPU 已验证 |
| `GroupedLinear` | expert FC1 `inputmats` | expert FC1 **模块输入** | 期望 128 MiB | BF16 已验证 |
| `swiglu` | `input_for_backward` | Weighted SwiGLU **模块输入**，即 FC1 gated 输出 | 期望 128 MiB | BF16 已验证 |
| `GroupedLinear` | expert FC2 `inputmats` | expert FC2 **模块输入**，即 SwiGLU 输出 | 期望 64 MiB | BF16 已验证 |
| `FlashAttention` | Q/K/V | FlashAttention **模块输入** | GQA 示例 64/8/8 MiB | BF16 GPU 梯度已验证 |
| `FlashAttention` | O | FlashAttention **模块输出** | 64 MiB | BF16 GPU 梯度已验证 |
| `FlashAttention` | `softmax_lse` | FlashAttention **内部输出** | 1 MiB | BF16 GPU 梯度已验证 |
| `UnfusedAttention` | Q/K/V | TE unfused attention **模块输入** | 由 MLA head dimension 决定 | MLA GPU 训练已验证 |
| `UnfusedAttention` | probability | softmax/dropout **输出**、PV BMM 输入 | `MBS*H/TP*S^2*dtype` | MLA full-budget GPU 已验证 |
| `UnfusedAttention` | O | TE unfused attention **模块输出**、projection 输入 | `MBS*S*H/TP*Dv*dtype` | MLA GPU 训练已验证 |
| `MTP` | `eh_proj_input` | MTP 归一化 embedding 与上一深度 hidden state 拼接后的 **2H projection 输入** | `MBS*S/TP*2H*dtype` | combined GPU 训练已验证 |

配置：

```text
--fl-offload-modules LayerNormLinear GroupedLinear swiglu FlashAttention
```

在路由均衡的上述大模型 BF16 示例中，一个有效 activation group 的期望 `captured` 为：

```text
64 + 64 + 128 + 128 + 64 + 64 + 8 + 8 + 64 + 1 = 593 MiB
```

实际值会随 `R` 变化。当前 `captured` 来自 Hook 收到的运行时张量，并按连续 storage
去重，不是根据模型配置套公式生成。

### 4.1 当前不能由 Hook 手动 offload 的保存值

| 保存值 | 当前是否有 Hook | 说明 |
| --- | --- | --- |
| `_LayerNormLinear.mu`、`rsigma` | 否 | 张量较小，仍保留在 TE saved tensors 中；`ln_out` 已进入显式 FL pack/unpack |
| FlashAttention RNG state、varlen sequence metadata | 否 | 张量较小，继续由原 autograd context 保存 |
| 独立的 attention projection `_Linear.saved_inputmat` | 否 | 只有与 Flash `O` 精确别名时才复用其 pack；独立 allocation 不重复捕获 |
| attention/MLP BDA 激活 | 否 | 当前没有 BDA Hook |
| router 与 dispatcher 激活/元数据 | 否 | 当前没有 Router/dispatcher Hook |
| cross-entropy 保存值 | 否 | 当前没有 loss Hook，且调度会排除 terminal chunk |

特别地，当前代码**没有 attention softmax Hook**。之前讨论过的 AttentionSoftmax 只是扩展
候选；在默认 FlashAttention 路径中也没有完整 softmax matrix 可供这种 Hook 捕获。

### 4.2 当前只观测的 Semantic Scope

以下 scope 已能通过 `--fl-saved-tensor-profile` 观测正常 forward 中实际进入
`save_for_backward` 的张量。FlashAttention Q/K/V/O/LSE 已有显式 FL 路径，其余对象仍为
只观测：

| Scope | 当前观测边界 | 预期识别对象 |
| --- | --- | --- |
| `qkv_linear` | QKV projection 调用 | TE fused norm/QKV 保存的输入、`ln_out`、统计量和参数 |
| `core_attn` | Flash/Core Attention 调用 | Q/K/V/O、LSE、RNG/sequence metadata 中的 Tensor 部分 |
| `attn_proj` | Attention output projection 调用 | projection 输入 `O` 及参数 |
| `expert_fc1` | TE GroupedLinear FC1 调用 | routed hidden input 及 FC1 参数 |
| `moe_act` | expert activation 调用 | Weighted SwiGLU 输入和 routing weight |
| `expert_fc2` | TE GroupedLinear FC2 调用 | SwiGLU 输出及 FC2 参数 |

Observer 输出同时记录 `logical_bytes` 与 `unique_storage_bytes`。同一 storage 的 view 会保留
各自的 shape/stride/storage offset，但只计一次物理 storage；若 `O` 同时被 `core_attn` 和
`attn_proj` 保存，后出现的记录会在 `shared_with_scopes` 中标出前一个 scope。

已有显式 `pack_hook` 会以 `source=explicit` 计入活动 scope，其余正常 autograd 保存项为
`source=autograd`。由于 PyTorch saved-tensor Hook 不保持自定义 Function backward 中的
Parameter 身份，Observer 当前要求 `--no-gradient-accumulation-fusion`。

## 5. FP8 对当前 Hook 的限制

“启用 FP8”不等于“表中所有激活都减半”：

- `LayerNormLinear` Hook 捕获原始 `inputmat` 和内部 `ln_out`；BF16 路径代码已完成，
  `ln_out` 的 GPU 数值与显存闭环仍待验证。
- FP8 `_GroupedLinear.inputmats` 可能是 TE `QuantizedTensorStorage`，不是普通
  `torch.Tensor`。当前 `TensorWrap` 复制路径尚未适配其数据、scale 元数据与恢复生命周期，
  所以 64/32 MiB 只是 TE 数据体理论值，不能视为当前已验证的 FL FP8 功能。
- Weighted SwiGLU 只有在 `activation_func_fp8_input_store=True` 时才把 backward 输入转为
  Float8；它与 FP8 GEMM 是两个独立开关，当前 FL 尚未完成该路径的 GPU 验证。
- FlashAttention 是否使用 FP8 DPA 由单独的 attention/recipe 配置决定，不能从线性层启用
  FP8 推导出 Q/K/V/O 已按 FP8 保存。

当前功能闭环和 `captured` 结论以 BF16 路径为准。

## 6. 如何理解运行日志

运行时捕获大小按 Hook 收到的实际对象计算：

```text
bytes = tensor.numel() * tensor.element_size()
```

并对连续 storage 精确去重。因此：

- `[FL offload] captured=... modules=[...]` 是当前 Hook、实际 dtype、实际路由和 TE recipe
  下的实测值；
- `captured` 是单个有效 `ActivationGroup` 可候选的总量，不是单个 rank 一轮训练的累计传输量；
- `selected` 是 budget 从这些候选中实际选择的量；
- activation memory breakdown 包含未被当前 Hook 捕获的值，不能直接等同于 `captured`；
- 同一逻辑值可能同时是上游模块输出和下游模块输入，统计时必须以真正保存它的 autograd
  节点和 storage 为准，不能按两种名称重复相加。
