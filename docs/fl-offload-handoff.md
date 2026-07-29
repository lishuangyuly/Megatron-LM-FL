# FL 激活值 Offload 直接移植：交接文档与当前状态

最后更新：2026-07-29（Asia/Shanghai）

本文档用于在新对话中继续推进激活值 Offload 的直接移植工作。文档记录仓库边界、
已有设计决策、当前实现、已验证行为、已知限制和后续任务。继续修改前应先阅读本文档。

训练中实际保存的激活值、当前 Hook 对应关系和 FP8 影响见
`docs/fl-offload-saved-activations.md`。

## 当前状态速览

- 仅在 `/home/lsy/zhiyuan/Megatron-LM-FL-DCU-Offload` 的 `fl-offload-direct` 分支工作。
- 已实现 FL 运行时、独立的 `--fl-*` 参数、普通调度封装、combined 1F1B 分阶段调度、
  Trace Profiler 和显存检查工具。
- 当前有效 Hook 为 `LayerNormLinear`、`GroupedLinear` 和 weighted `swiglu`。
- 当前功能闭环仅验证 BF16；TE FP8 `GroupedLinear` 的 `QuantizedTensorStorage` 尚未适配。
- 当前 combined MoE smoke 的每个有效激活组实际捕获 27 MiB，即 1 + 10 + 16 MiB。
- FL Offload 的基础与 combined smoke 均默认使用 FlashAttention，可通过
  `ATTENTION_BACKEND` 覆盖。
- Flash + 完整 27 MiB 的 baseline/offload 对比已通过，loss 和 grad norm 完全一致。
- Flash + 完整 27 MiB 的四阶段 Trace 已在 4 个 rank 上通过，不存在不完整生命周期。
- rank 2/3 的 Offload 序列较少，因为调度会主动跳过最后一个 PP rank 的最后一个 VPP chunk。
- 已分析 `AttentionSoftmax`，但未实现 Hook，因为 FlashAttention 不生成完整 Softmax 矩阵。
- Trace 中的 `copy_time_us` 只是匹配到的事件时间下界，不能证明实际传输字节量。
- 在真实模型的显存和性能数据支持其他设计前，保留 DCU 风格的持久 H2D 落地缓冲区和
  恢复张量 clone。

## 1. 仓库与环境边界

- 当前仓库：`/home/lsy/zhiyuan/Megatron-LM-FL-DCU-Offload`
- 当前分支：`fl-offload-direct`
- 当前已提交 HEAD：`cf07db325 feat(fl-offload): validate swiglu transfer overlap`
- 移植来源：`/home/lsy/zhiyuan/dcu_megatron`
- Python 环境：`/home/lsy/miniconda3/envs/fl_env`
- 本次移植前的基准提交：`ecb5dfade`
- 本工作不得修改 `/home/lsy/zhiyuan/Megatron-LM-FL`。另有人正在该仓库的
  `fl-offload` 分支工作，直接移植的所有修改必须留在上述独立仓库中。

创建本文档时，工作区已有未提交修改：

- `examples/fl_offload/run_smoke.sh` 和 `examples/fl_offload/run_combined_smoke.sh`
  默认使用 FlashAttention，并接受
  `ATTENTION_BACKEND` 环境变量，可取 `auto`、`flash`、`fused` 或 `unfused`。
- `docs/fl-offload-saved-activations.md` 记录训练保存激活、Hook 对应关系和 FP8 影响。

本文档本身在明确提交前也是一个未提交文件。

## 2. 目标与已有决策

当前目标是先把 DCU Megatron 的激活值 Offload 行为直接移植并跑通，再分析缺陷并优化。
现阶段正确性和功能闭环优先于性能调优。

除非有明确证据需要调整，否则必须保留以下决策：

1. 功能使用独立的 `fl-` 命令行参数命名空间，不得复用 Megatron 原有的
   `--offload-modules`、细粒度激活值 Offload 配置或运行时。
2. 本移植中原有的用户可见和代码级 `dcu` 命名均已改为 `fl`。
3. 当前已验证基线使用算子显式接入的 `pack_hook`/`unpack_hook`。后续计划借鉴 Megatron
   原生 Fine-grained Activation Offloading，引入按 `qkv_linear/core_attn/moe_act` 等
   语义范围工作的 scoped saved-tensor Hook；在新路径完成验证前不得删除显式 Patch，
   也不得让两条捕获路径同时处理同一保存张量。
4. Reload 使用当前 DCU 的有效设计：H2D 先写入持久 GPU 字节缓冲区，再将每个恢复张量
   clone 出落地缓冲区。该设计可能短暂同时持有落地缓冲区和恢复张量，但能避免持久缓冲区
   复用时覆盖已经恢复的张量。
5. D2H 与 H2D 默认共用一条专用 FL copy stream，以固定 Offload/Reload 的提交顺序并避免
   两条 FL copy stream 相互竞争；`--fl-use-comm-stream` 保持可选且默认关闭，开启后两种
   传输共同使用 combined schedule 的正常通信流。现有小型 smoke 模型不足以证明通信流
   应成为永久默认值。
6. combined smoke 当前默认使用 FlashAttention。此前考虑的 `AttentionSoftmax` Hook
   没有实现，因为 FlashAttention 不会物化完整注意力概率矩阵。

## 3. 已有提交序列

这些提交按实际开发流程组织，而不是把最终代码按模块拆分：

1. `8c2aa1fd4 feat(fl-offload): port explicit activation offload runtime`
2. `575f37c86 feat(fl-offload): wire the ordinary training schedule`
3. `dee7543ca fix(fl-offload): isolate command-line arguments`
4. `b906e08fc feat(fl-offload): schedule staged copies in combined 1F1B`
5. `e38d5c661 feat(fl-offload): trace copy lifecycles and schedule positions`
6. `302706a74 fix(fl-offload): ignore projected GPU semantic annotations`
7. `01f04a506 fix(fl-offload): harden runtime lifecycle invariants`
8. `c94d8f790 test(fl-offload): verify steady-state memory reduction`
9. `cf07db325 feat(fl-offload): validate swiglu transfer overlap`

用于重写提交元数据的脚本和 TSV 位于仓库外：

- `/home/lsy/zhiyuan/rewrite_commit_metadata.sh`
- `/home/lsy/zhiyuan/fl_commit_metadata.tsv`

## 4. 实现结构

### 4.1 运行时

主文件：`megatron/plugin/fl_offload/offload.py`

运行时生命周期如下：

```text
算子在 record(key) 上下文内执行 forward
  -> pack_hook(actual_tensor, op_name)
  -> 收集为 ActivationGroup 候选张量
  -> 排序、精确存储去重、按字节预算选择
  -> 分阶段 D2H，写入池化的 pinned CPU 缓冲区
  -> 清除已选中 TensorWrap.x 的引用
  -> 分阶段 H2D，写入持久 GPU 落地缓冲区
  -> 将恢复张量 clone 出落地缓冲区
  -> 算子 backward 调用 unpack_hook
```

核心对象：

- `TensorWrap`：保存活跃 GPU 张量以及 shape、dtype、device 元数据的可变槽位。
- `TensorPack`：由算子 autograd context 持有；`unpack_hook` 从其当前 `TensorWrap.x`
  读取张量。
- `ActivationGroup`：某个调度 key/microbatch 捕获到的全部有效张量。
- `CopyTaskGroup`：将选中的字节区间均分为多个复制阶段。
- `OffloadAsync`：D2H 的 prologue、分阶段 issue 和 epilogue。
- `OnloadAsync`：H2D 的 prologue、分阶段 issue 和 epilogue。

候选张量必须同时满足以下条件：

- 已启用 `--fl-patch-te`。
- 当前处于 `record(...)` 上下文中，并且梯度已启用。
- `op_name` 位于 `--fl-offload-modules` 中。
- 对象不是 `torch.nn.Parameter`。
- 对象未被识别为 RoPE frequency buffer。
- 实际运行时大小不小于 `--fl-min-offloaded-tensor-size`。

`captured` 由实际张量的 `tensor.numel() * tensor.element_size()` 计算，不是理论模型估算值。

选择逻辑：

- 候选张量先按是否连续排序，连续张量优先；之后按 `numel` 降序排列。
- 对完全相同的连续存储，即 `device`、`data_ptr` 和字节大小均相同的张量，只复制一次。
- 字节预算由 `--fl-per-batch-offload-size` 指定，单位为 MiB。
- 当 `captured >= budget` 时，精确选择 `budget` MiB。
- 当 `captured < budget` 时，选择量为 0，而不是把所有已捕获字节全部选中。
- 预算为 0 时仅执行捕获统计，不进行传输。
- 选中字节数必须能被复制阶段数整除。
- 预算允许切到张量中间。未选中的后缀会 clone 到 `partial_remainders` 并保留在 GPU 上。

重要参数状态：

- `--fl-activation-offload-ratio` 已解析，但运行时尚未使用。
- `--fl-activation-offload-threshold` 已解析，但运行时尚未使用。
- 当前有效选择只由模块名、最小张量大小和固定 MiB 预算控制。

缓冲区和 Stream 行为：

- pinned CPU 缓冲区按精确的 `(num_bytes, dtype)` 复用，并通过完成事件保护。
- 持久 GPU `onload` 字节缓冲区会扩展到历史最大请求大小，之后持续复用。
- D2H Offload 和 H2D Reload 默认共用一条专用 FL copy stream。
- 开启 `--fl-use-comm-stream` 后，两个方向都使用 combined 调度的通信 Stream。
- 当一个阶段跨越多个张量边界时，D2H 一个阶段可能发出多次复制。
- H2D 每个阶段向连续落地缓冲区发出一次复制。

### 4.2 显式算子 Hook

TE 运行时 Patch：`megatron/plugin/fl_offload/te_patch.py`

TE Patch 使用 `inspect.getsource`、精确源码锚点和动态编译。若受支持的 TE 版本不再包含
且仅包含一个预期锚点，它会主动失败。当前 Patch 包括：

- TE `_LayerNormLinear`：将 `inputmat` 以 `LayerNormLinear` 名称进行 pack，并从 TE 原有
  saved tensor 列表中移除；backward 时再 unpack。
- TE `_GroupedLinear`：将每个 grouped-linear 的 `inputmat` 以 `GroupedLinear` 名称进行
  pack；weight 和 bias 仍保留在 TE 原有 saved tensor 列表中。

Megatron 源码 Hook：`megatron/core/fusions/fused_bias_swiglu.py`

- `WeightedSwiGLUFunction` 将 backward 所需输入以 `swiglu` 名称进行 pack。
- Router weight 仍保留在 `ctx.save_for_backward` 中。
- 普通 `SwiGLUFunction` 和 `BiasSwiGLUFunction` 尚未接入 FL Offload。

### 4.3 调度接入

入口安装：`megatron/plugin/fl_offload/install.py` 和 `pretrain_gpt.py`

- `pretrain_gpt.py` 总是安装封装，但未开启 `--fl-patch-te` 时是无操作路径。
- 普通 forward/backward 调度可在 forward 后捕获并 Offload，根据输出匹配 backward，
  Reload 后再执行 backward。
- 无流水的 combined 路径使用以正确性为先的即时 Offload/Reload 往返，不提供较长的
  激活驻留空档。

分阶段重叠路径：

- `megatron/core/pipeline_parallel/combined_1f1b.py`
- `megatron/core/pipeline_parallel/schedules.py`
- `megatron/core/models/common/model_chunk_schedule_plan.py`

交错调度使用以下 key：

```text
(activation_group_id, model_chunk_id, microbatch_id_in_model_chunk)
```

每层暴露四个 issue 位置：

```text
stage 0: after_combine_bwd，在 attention forward 之前
stage 1: after_dispatch_fwd，在 MLP backward 之后
stage 2: after_dispatch_bwd
默认路径在 stage 2 后依次执行 forward MLP 和 forward combine
stage 3: after_combine_fwd
默认路径在 stage 3 后执行 backward attention
```

当 `ep_overlap_early_attn_memory_release` 开启时，backward attention 会提前到 stage 2 之前；
这属于框架原有的早释放路径，不是本次 FL 调度实验引入的行为。

每个 `issue_loads` 位置都先发出 Reload，再发出 Offload。配置的 `0 1 2 3` 分配会跨层重复。

最后一个 PP rank 的最后一个 model chunk 会被明确排除在 Offload 和 Reload 之外：

```text
pipeline_parallel_rank == pipeline_parallel_size - 1
and model_chunk_id == len(model) - 1
```

这解释了为什么当前 PP=2、VPP 配置下 rank 0/1 的完整序列数是 rank 2/3 的两倍。
这也意味着如果不扩展调度，vocabulary cross-entropy 等最后 chunk 的激活值无法 Offload。

### 4.4 Profiling 与显存检测

Profiling 插件：

- `megatron/plugin/profile/core.py`
- `megatron/plugin/profile/autograd_record.py`
- `megatron/core/models/gpt/fine_grained_callables.py`

语义标注覆盖 attention、dispatch、MoE、combine、backward 边界、Offload/Reload 生命周期，
以及四个 `issue_loads` 位置。只有启用 `--profile-pp-semantics` 时才会生成这些标注。

Trace 校验器：`examples/fl_offload/validate_trace.py`

它验证以下内容：

- prologue、stage 0..3、epilogue 生命周期完整；
- Offload/Reload 序列成对；
- issue 位置位于正确的 combined 调度 step 内；
- GPU 上存在 D2H/H2D 活动；
- 复制与 GPU 计算、通信 kernel 存在重叠。

Trace 限制：语义 issue 数由阶段数固定，不能证明字节量。一个 D2H 阶段可能包含多个张量
分片复制，而投影标注匹配器可能只归因其中一部分。因此当前 `copy_time_us` 只是下界，
不能据此推导 27 MiB 预算的有效带宽。原始 `gpu_d2h/gpu_h2d` 数量还包含框架的其他传输。
后续校验器应在 Trace 元数据可用时累计实际字节数。

显存检测：

- `megatron/plugin/fl_offload/memory.py`
- `examples/fl_offload/compare_memory.py`

它会在每个训练迭代附近重置 allocator peak、执行同步、确认 FL 运行时在 step 边界处于
空闲状态，并按 rank 输出 allocated/reserved 峰值。

当前比较策略：

- 默认由分布式全局峰值的下降量决定通过或失败。
- 本地 rank 显存回退默认只产生警告；只有明确设置 `--max-rank-regression-mib` 才会失败。
- combined smoke 要求全局峰值至少下降 1 MiB。

## 5. 当前支持的激活值规模

combined smoke 配置如下：

```text
layers=4, VPP layers per stage=1
hidden=512, FFN hidden=2048
heads=8, sequence length=1024
microbatch=1, global batch=8
PP=2, EP=2, TP=1, expert TP=1
experts=2, router top-k=2
BF16, TE grouped GEMM, all-to-all dispatcher
SwiGLU, no linear bias
```

对每个有效激活组/层，实际日志报告 27 MiB：

| Hook | 在当前 smoke 中的实际作用 | 运行时 shape 解释 | 捕获量 |
| --- | --- | --- | ---: |
| `LayerNormLinear` | QKV fused norm-linear 输入 | `1024 x 1 x 512` BF16 | 1 MiB |
| `GroupedLinear` | Expert FC1 输入 | 约 `2048 x 512` BF16 | 2 MiB |
| `GroupedLinear` | Expert FC2 输入 | 约 `2048 x 2048` BF16 | 8 MiB |
| `swiglu` | Weighted SwiGLU backward 输入 | 约 `2048 x 4096` BF16 | 16 MiB |
| | 合计 | | 27 MiB |

约 2048 行指的是 dispatch 后实际观察到的本地 routed token assignment 数量，不能笼统称为
通用的“EP group token 数”。该数量取决于运行时路由；在当前 top-k=2、两个 expert 的 smoke
配置中观察值保持稳定。

`GroupedLinear` 已经覆盖 Expert FC1 和 Expert FC2 的输入。除非底层算子路径发生变化，
否则不能将二者再次计为新的 Offload 机会。

## 6. FlashAttention 状态

在 `fl_env` 中观察到的包版本：

```text
flash-attn 2.7.3
Transformer Engine 2.14.0+3c34bb9a
```

combined smoke 当前默认使用：

```text
--attention-backend flash
```

可以设置 `ATTENTION_BACKEND=unfused` 进行对比。Megatron 会配置 TE 环境，使 `flash`
启用 FlashAttention 并关闭 fused/unfused attention fallback，因此运行成功即可确认 Flash
后端可用。

FlashAttention 不会物化 `[batch, heads, seq_q, seq_k]` 概率矩阵。因此：

- 该路径中不存在此前估算为 16 MiB 的 `AttentionSoftmax` 激活值。
- 尚未添加 `AttentionSoftmax` Hook。
- FlashAttention 内部会按 backward 需要保存 Q/K/V、输出、softmax LSE 和 RNG 状态。
- Flash 已经避免二次方 attention probability，但 Q/K/V/O 仍是可观的线性规模保存激活。
  它们应作为语义 scope 迁移的第二阶段目标；其中 `O` 必须由 `core_attn` 与 `attn_proj`
  联合处理，不能只增加普通 `_Linear` Hook。

与最近一次等价 unfused 运行相比，Flash 报告的迭代时间约下降 5.4%：

```text
iteration 1: 11381.0 ms -> 10761.1 ms
iteration 2:  5856.1 ms ->  5538.0 ms
iteration 3:   257.8 ms ->   243.9 ms
```

由于 kernel 数值计算顺序不同，不同 attention 后端的 loss 会略有差异。正确性对比必须在
相同后端下比较 baseline 和 Offload。

## 7. 已完成验证

### 7.1 单元测试与基本检查

此前记录的聚焦测试结果：

```text
7 passed, 9 warnings
```

这些测试覆盖独立 CLI 命名空间解析、分阶段往返、部分张量、引用释放、no-grad 行为、
跨层重复阶段分配，以及 Weighted SwiGLU backward 与 baseline 精确一致。当前代码树还包含
额外的生命周期、重复存储、显存解析和 Trace 测试；后续修改后应重新执行完整聚焦测试集。

TE Patch 的导入和应用已通过以下命令验证：

```bash
/home/lsy/miniconda3/envs/fl_env/bin/python -c \
  'from megatron.plugin.fl_offload.te_patch import apply_te_patches; apply_te_patches(); print("TE patch OK")'
```

### 7.2 Loss/梯度闭环

最初的简单 smoke 已通过 baseline/offload 的 loss 和梯度精确对比。

combined Flash smoke 在完整 27 MiB 预算下通过：

```text
captured=27.00 MiB
budget=27 MiB
selected=27.00 MiB

GroupedLinear:  10/10 MiB selected
LayerNormLinear: 1/1 MiB selected
swiglu:          16/16 MiB selected
```

三个迭代的 Flash baseline/offload loss 和输出的 grad norm 均完全一致：

```text
iteration 1 loss=9.126180, grad_norm=1.693
iteration 2 loss=9.023922, grad_norm=2.072
iteration 3 loss=8.693712, grad_norm=2.406
```

没有 skipped iteration、NaN iteration 或分阶段复制未完成警告。

完整预算下的短时性能结果存在波动，不足以得出性能结论：

```text
baseline iteration 3: 243.9 ms
offload  iteration 3: 235.4 ms
```

前两个迭代中 Offload 约慢 1.1%，第三个迭代约快 3.5%。在获得更长时间、更真实的负载前，
应将这些差异视为噪声。

### 7.3 完整预算 Trace 闭环

Flash + 27 MiB + 四阶段配置通过 Trace 校验器：

```text
rank 0: 8 complete offload, 8 complete reload, 8 paired
rank 1: 8 complete offload, 8 complete reload, 8 paired
rank 2: 4 complete offload, 4 complete reload, 4 paired
rank 3: 4 complete offload, 4 complete reload, 4 paired
boundary partial: 0 on every rank
```

重叠统计如下：

| Rank | 计算重叠 | 通信重叠 | 是否位于通信空隙 |
| ---: | ---: | ---: | --- |
| 0 | 11.02% | 25.12% | 否 |
| 1 | 9.38% | 41.30% | 否 |
| 2 | 2.82% | 0.00% | 是 |
| 3 | 3.67% | 77.14% | 否 |

每个 rank 都存在非零的实际计算重叠。这些比例只覆盖与投影 issue 标注匹配的 memcpy 事件，
需要结合前述 Trace 限制理解。

训练成功结束后出现的 `destroy_process_group()` 未调用警告属于进程清理警告，不是 FL
Offload 功能错误。

### 7.4 Attention backward 调序实验

曾实验性地将默认路径从：

```text
forward MLP -> forward combine -> stage 3 -> backward attention
```

调整为：

```text
forward MLP -> backward attention -> forward combine -> stage 3
```

Trace 表明该调整没有形成预期的 Attention 计算与 Combine 通信重叠，现已撤回。原因是当前
`ScheduleNode.backward()` 会在调用线程中同步执行完整的
`Variable._execution_engine.run_backward()`；主线程只有在完整 Attention backward graph
完成遍历和 kernel 提交后，才能继续调用 forward combine。不同 CUDA stream 只能重叠已经
提交的工作，不能让下一行 Python 调用提前执行。

后续若要获得稳定重叠，需要参考 DCU 的细粒度方案，将 Attention backward 至少拆分为
projection、core attention 和 QKV 节点：先完成 projection backward 并提交 combine，
再提交较大的 core attention/QKV backward。仅交换完整 Attention 节点与 Combine 节点的
顺序，只会改变串行方向。

### 7.5 显存结果

较早的小预算测试显示，分布式全局峰值下降约 16 MiB，即全局峰值 rank 从 759.59 MiB
降至 743.59 MiB。后续 SwiGLU/较大预算实验表明，rank 0/1 可能改善，而 rank 2/3 可能
小幅回退，原因包括：

- 最后一个 PP rank 的最后一个 VPP chunk 不参与 Offload；
- 持久落地缓冲区和恢复张量 clone 会增加 GPU 显存；
- 当前小模型在最后几个 rank 上的可 Offload 激活值较少。

## 8. 后续操作命令

进入仓库并激活环境：

```bash
cd /home/lsy/zhiyuan/Megatron-LM-FL-DCU-Offload
conda activate fl_env
```

聚焦测试：

```bash
python -m pytest \
  tests/unit_tests/test_fl_offload_arguments.py \
  tests/unit_tests/test_fl_offload_direct.py \
  tests/unit_tests/test_fl_offload_state.py \
  tests/unit_tests/test_fl_offload_memory.py \
  tests/unit_tests/test_fl_offload_trace.py \
  -q
```

单元测试 setup 期间可能出现缺少 `/opt/data/*.zip` 的 dataset/tokenizer fixture 警告；
这些警告在此前聚焦测试中不影响结果。

Flash 完整预算正确性测试：

```bash
ATTENTION_BACKEND=flash \
FL_OFFLOAD_MIB=27 \
LOG_DIR=/tmp/fl_offload_flash_full \
bash examples/fl_offload/run_combined_smoke.sh compare
```

Flash 完整预算 Trace 测试：

```bash
TRACE_DIR=/tmp/fl_offload_flash_full/trace_$(date +%s)

ATTENTION_BACKEND=flash \
FL_OFFLOAD_MIB=27 \
LOG_DIR=/tmp/fl_offload_flash_full \
TRACE_DIR="$TRACE_DIR" \
bash examples/fl_offload/run_combined_smoke.sh trace
```

Flash 完整预算显存测试：

```bash
ATTENTION_BACKEND=flash \
FL_OFFLOAD_MIB=27 \
LOG_DIR=/tmp/fl_offload_flash_full_memory \
bash examples/fl_offload/run_combined_smoke.sh memory
```

对比 unfused 和 Flash baseline：

```bash
ATTENTION_BACKEND=unfused \
LOG_DIR=/tmp/fl_offload_attn_unfused \
bash examples/fl_offload/run_combined_smoke.sh baseline

ATTENTION_BACKEND=flash \
LOG_DIR=/tmp/fl_offload_attn_flash \
bash examples/fl_offload/run_combined_smoke.sh baseline
```

## 9. 已知限制与风险

1. 最完整的重叠接入仅覆盖 combined interleaved 1F1B 调度。其他调度已有正确性封装，
   但尚未全部验证是否具备有效重叠和合理的显存驻留时间。
2. 跳过最后一个 PP rank/最后一个 model chunk 会造成各 rank Offload 次数不对称，并限制
   最后几个 rank 的显存收益。
3. 持久落地缓冲区加 clone 在功能上较稳健，但会增加瞬时和常驻 GPU 显存开销。
4. 基于精确源码的 TE Patch 有意保持版本敏感性。更新 TE 后必须重新执行 Patch 基本检查
   和算子梯度测试。
5. 部分张量选择会在 GPU 上保留 clone 后的后缀。功能正确，但可能降低实际显存收益。
6. `fl_activation_offload_ratio` 和 `fl_activation_offload_threshold` 当前没有运行时效果。
7. 当前 Trace 重叠统计不能证明选中字节量，也不能覆盖完整复制时间。
8. 当前性能测量使用很小的三迭代模型，结果主要受初始化、编译和随机波动影响。
9. FlashAttention 不生成完整 Softmax 概率张量，因此默认 Flash 配置下不能把仅适用于
   unfused 后端的 Offload 候选报告为可用功能。
10. 当前显存检查器默认使用分布式全局峰值作为通过标准。如需严格限制单 rank 行为，
    应设置 `--max-rank-regression-mib`。
11. FP8 下的 TE `GroupedLinear.inputmats` 可能是 `QuantizedTensorStorage`，不能直接套用
    当前面向普通 `torch.Tensor` 的打包和字节 view 逻辑；FP8 表格目前是规模分析而非
    已验证功能。
12. 当前显式 Patch 是直接移植阶段的验证基线，不是最终扩展架构。它依赖 TE 源码锚点，
    覆盖新保存激活时需要继续修改算子 backward，长期维护成本较高。
13. 早期曾尝试用一个 saved-tensor Hook 包围整个 microbatch。该方案会触及 weight，导致
    PyTorch 恢复的 Tensor 丢失 TE fused-wgrad 依赖的 Parameter 属性；为避免 view/weight
    又采用了过度保守过滤，漏掉大量合法 MoE 激活。这一失败不能直接等同于 Megatron
    原生的“窄作用域 + TE CPUOffloadEnabled/Parameter 保护”方案，但新 scope 路径必须专门
    回归 `gradient_accumulation_fusion` 和 Parameter 身份。

## 10. 建议的后续工作

### 10.1 架构结论

下一阶段不再以“继续增加 `_Linear` 等实现类 Patch”为主线，而是把用户配置和捕获边界提升
为语义 scope。底层仍然搬运具体 saved tensor，但对外按训练功能组织：

```text
qkv_linear
core_attn
attn_proj
attn_norm
mlp_norm
expert_fc1
moe_act
expert_fc2       # FL 需要保留的扩展；Megatron 原生当前没有独立此项
```

Megatron 原生方案可复用的思想是：

- 在上述窄范围前向外安装 saved-tensor Hook，而不是 Hook 整个 microbatch；
- 用 group start/commit 明确张量最后一次前向使用的位置；
- 对 residual、Q/K/V、FC1 输出等仍被 Python 引用的张量，在依赖结束后显式释放；
- `attn_proj` 不能脱离 `core_attn` 单独启用，因为二者都保存同一个 attention 输出 `O`；
- 将便宜的 norm/activation 重计算与大张量 Offload 组合，而不是所有张量一律搬运。

FL 不直接复用 Megatron 原生 `PipelineOffloadManager`、参数和传输策略。以下已经验证的 FL
行为应继续保留：

- 独立 `--fl-*` 参数命名空间；
- 单个 ActivationGroup 的固定 MiB budget；
- 连续 CPU byte buffer 和持久 H2D 落地 buffer；
- combined 1F1B 中四阶段 D2H/H2D issue 位置；
- 当前 Trace、loss/grad 和显存验证工具。

目标数据流为：

```text
语义 scope 内的 save_for_backward
  -> FL scoped pack/unpack
  -> storage/参数安全过滤与 scope 分项统计
  -> 当前 ActivationGroup 按预算选择
  -> 当前 combined schedule 分阶段 D2H/H2D
  -> backward 自动或显式恢复
```

### 10.2 分阶段迁移计划

#### 阶段 A：建立只观测的 Scope Profiler

1. 增加 FL 独立 scope 上下文，但先不进行 D2H/H2D。
2. 优先在 Megatron 原生已经验证的插入点标注 `qkv_linear`、`core_attn`、`attn_proj`、
   `expert_fc1` 和 `moe_act`。
3. 每个保存对象记录 scope、shape、dtype、字节数、Parameter 身份、data pointer、底层
   storage 范围、storage offset、stride、是否连续和是否为 TE 量化对象。
4. combined schedule 的现有 `attn` 节点还包含 pre-MLP norm、Router 和 dispatch preprocess，
   不能直接把节点名当作纯 self-attention scope；需要在具体 callable 内建立更窄边界。
5. 输出每个 scope 的逻辑保存量和物理 storage 去重量，为后续选择提供实测依据。

验收条件：默认关闭时无行为变化；开启 profiler 时 baseline loss/grad 不变；统计能解释
当前显式 Hook 的 27 MiB，并能识别 `O` 的跨 scope 共享和 Q/K/V view。

#### 阶段 B：迁移 `qkv_linear`，优先获得 `ln_out`

1. 以 `qkv_linear` scope 捕获 TE `_LayerNormLinear` 为 backward 保存的 activation。
2. 目标至少包括现有 `inputmat` 和尚未 Offload 的 `ln_out`；大模型 BF16 配置中
   `ln_out` 约 64 MiB/层，是当前实现风险最低、实际释放概率最高的新机会。
3. scoped 模式处理该范围时禁用对应显式 `_LayerNormLinear` 捕获，禁止重复 pack。
4. 参考 Megatron/TE 的 `CPUOffloadEnabled`、`mark_not_offload` 和 weight object 保存机制，
   确保 weight/bias 保持 Parameter 身份。

验收条件：BF16 round-trip、baseline/offload loss 和全部梯度一致；分别在开启和关闭
`gradient_accumulation_fusion` 时通过；`captured` 增量与实际 `ln_out` 大小一致；显存峰值
出现正向收益。

#### 阶段 C：迁移并保持 MoE 现有覆盖

1. 将现有 GroupedLinear FC1 输入映射到 `expert_fc1`。
2. 将 weighted SwiGLU 输入映射到 `moe_act`。
3. 增加 FL 专用 `expert_fc2` scope，继续覆盖当前 GroupedLinear FC2 输入。不能因为照搬
   Megatron 原生 group 列表而丢失已经验证的 FC2 激活值。
4. 保留运行时 routed assignment `R` 统计，不使用静态 EP token 公式替代实测 shape。
5. scoped MoE 路径通过后，再按 scope 逐项关闭 `_GroupedLinear` 和 weighted SwiGLU 显式
   Patch；迁移期间提供互斥 capture mode，而不是双重捕获。

验收条件：小型 combined smoke 的 scoped 路径仍能解释并覆盖原 27 MiB；完整预算下
loss/grad、四阶段 Trace 和显存结果不劣于显式路径。

#### 阶段 D：将 attention 作为联合功能迁移

1. `core_attn` 一次性处理 FlashAttention 保存的 Q/K/V/O/LSE/RNG context。
2. `attn_proj` 处理 projection 为 wgrad 保存的输入 `O`。
3. 强制配置依赖：启用 `attn_proj` 必须同时启用 `core_attn`。
4. 第一版可以参考 Megatron 原生方案，为两个 backward consumer 保存独立 CPU 副本，优先
   保证生命周期正确；确认带宽成为瓶颈后，再评估跨 scope 共用 CPU storage。
5. Q/K/V 可能是融合 QKV storage 的非连续 view。在启用实际搬运前，当前只支持精确
   `data_ptr + size` 的去重逻辑必须扩展为 storage range、offset 和 stride 感知的恢复。
6. 默认 Flash 路径不实现 `AttentionSoftmax`；只有维护 unfused 后端时才加入后端限定实现。

验收条件：MHA 与 GQA 分别测试；Flash baseline/offload loss 和梯度一致；`core_attn` 与
`attn_proj` 的 D2H/H2D 次数、顺序和 backward 恢复位置可由 Trace 证明；显存下降不能只由
`captured` 推断。

#### 阶段 E：扩大覆盖与精度支持

在前四阶段稳定后再评估：

1. `attn_norm/mlp_norm` 与 BDA residual 的联合生命周期；
2. Router 输入、routing metadata、dispatcher/combine buffer；
3. dense/shared expert 的普通 Linear、普通/带 bias SwiGLU；
4. vocabulary cross-entropy；该项需要先解除最后 PP rank/最后 chunk 排除限制；
5. TE `QuantizedTensorStorage`、scale metadata 和 FP8 DPA；
6. non-VPP、non-combined、PP=1 和 CUDA Graph 路径。

### 10.3 并行的验证与性能工作

以下任务不依赖 scope 迁移完成，可以并行推进：

1. 改进 Trace 校验器，使语义事件携带并累计实际计划/完成字节数，覆盖一个阶段包含多个
   D2H tensor fragment 的情况。
2. 在单个 activation group 可 Offload 数据超过 100 MiB、迭代足够长的配置上重新比较
   FL 统一专用 copy stream 与 combined 通信 stream。
3. 比较 Megatron 式每张量 reload buffer 与当前 DCU 式持久落地 buffer + clone；在获得
   稳定峰值和吞吐数据前保持当前默认。
4. 对每次新增 scope 都执行同后端 baseline/offload 对比，不使用不同 attention backend
   之间的 loss 差异判断正确性。

### 10.4 显式 Patch 的退出条件

显式 Patch 只有在以下条件全部满足后才能删除：

1. scoped 路径覆盖 `LayerNormLinear`、Grouped FC1/FC2 和 weighted SwiGLU 的现有功能；
2. 不再要求关闭 gradient accumulation fusion，且 TE fused-wgrad Parameter 属性完整；
3. BF16 loss、全部梯度、完整预算 round-trip、Trace 生命周期和显存检查全部通过；
4. 对同一张量不存在显式与 scoped 双重 pack；
5. 至少一个真实大模型配置证明 scoped 路径的显存和吞吐不差于当前显式路径；
6. FP8 未完成时必须明确保持 BF16-only 边界，不能用理论量化大小代替功能验证。

## 11. 快速解读指南

- `captured=X`：一个激活组内实际观察到的有效张量字节数。
- `budget=Y`：请求选择的固定字节前缀大小。
- `selected=Z`：计划传输的字节数，是功能预算是否生效的直接结果。
- loss/梯度一致：说明 Reload 后的数据正确。
- 四阶段 Trace 完整：说明生命周期和调度位置正确。
- 计算重叠非零：说明确有并发，但不代表传输已被完全隐藏。
- 显存下降：证明张量释放时间足够早并实际降低了峰值。
- 本地 rank 回退但全局改善：当前校验器允许这种情况，通常与 PP/VPP 不对称和落地缓冲区
  开销有关。
