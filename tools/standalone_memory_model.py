#!/usr/bin/env python3
"""Framework-independent estimator for peak accelerator memory during training."""

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass


MIB = 1024**2
GIB = 1024**3
OFFLOAD_MODULES = (
    "LayerNormLinear",
    "GroupedLinear",
    "swiglu",
    "Attention",
    "MLA",
    "SharedExpert",
    "MTP",
)
_ATTENTION_MODULE_ALIASES = {
    "attention": "Attention",
    "flashattention": "Attention",
    "fusedattention": "Attention",
    "unfusedattention": "Attention",
}


def _normalize_offload_module(module):
    return _ATTENTION_MODULE_ALIASES.get(str(module).lower(), module)


def _normalized_offload_modules(modules):
    return tuple(dict.fromkeys(_normalize_offload_module(module) for module in modules))


# Edit these values for a new capacity-planning run. Command-line arguments
# remain available and override the corresponding values temporarily.
MODEL_CONFIG = {
    "layers": 26,
    "hidden_size": 2048,
    "ffn_hidden_size": 512,
    "vocab_size": 129280,
    "sequence_length": 10240,
    "micro_batch_size": 1,
    "attention_heads": 16,
    # train_numa.sh NUM_QUERY_GROUPS=0 disables GQA, so Megatron resolves the
    # effective KV head count to NUM_ATTENTION_HEADS=16.
    "kv_heads": 16,
    "attention_backend": "fused",
    "multi_latent_attention": True,
    "q_lora_rank": 0,
    "kv_lora_rank": 512,
    "qk_head_dim": 128,
    "qk_pos_emb_head_dim": 64,
    "v_head_dim": 128,
    "qk_layernorm": True,
    "mla_down_proj_fusion": False,
    "experts": 64,
    "topk": 6,
    "moe_layers": 26,
    # Zero preserves the historical behavior of using ffn_hidden_size for
    # routed experts.
    "moe_ffn_hidden_size": 1408,
    "moe_shared_expert_intermediate_size": 2816,
    "moe_shared_expert_gate": False,
    # MTP is placed on the last PP stage by Megatron when no custom pipeline
    # layout is supplied. The inner layer follows the final decoder layer type.
    "mtp_num_layers": 1,
    "mtp_layer_is_moe": True,
    "capacity_factor": 1.0,
    "gated_mlp": True,
    "untied_embeddings": True,
}

PARALLEL_CONFIG = {
    "world_size": 16,
    "tensor_parallel": 1,
    "pipeline_parallel": 2,
    "expert_parallel": 4,
    "expert_tensor_parallel": 1,
    "context_parallel": 1,
    # train_numa.sh only enables sequence parallel when TP > 1.
    "sequence_parallel": False,
    "pipeline_schedule": "interleaved-1f1b",
    # Same meaning as --num-layers-per-virtual-pipeline-stage.
    "layers_per_virtual_pipeline_stage": 1,
    # Zero uses PP, matching Megatron's default VP microbatch group size.
    "microbatch_group_size_per_vp_stage": 0,
    # Combined MoE 1F1B inserts one additional warmup forward.
    "overlap_moe_expert_parallel_comm": True,
    # Set overrides only for an uneven pipeline partition; zero/-1 means auto.
    "local_layers": 0,
    "local_moe_layers": -1,
    # Zero derives residency from the schedule; a positive value overrides it.
    "inflight_microbatches": 0,
}

PRECISION_CONFIG = {
    "parameter_bytes": 2.0,
    "gradient_bytes": 4.0,
    "master_parameter_bytes": 4.0,
    "optimizer_state_bytes": 8.0,
    # Match Megatron's --use-distributed-optimizer. Shared optimizer state is
    # sharded over DP, while expert state uses the independent expert-DP grid.
    "use_distributed_optimizer": True,
    "activation_bytes": 2.0,
    "lse_bytes": 4.0,
    "logit_bytes": 4.0,
    "parameter_shards": 1,
    "gradient_shards": 1,
    "optimizer_shards": 1,
    "expert_activation_shards": 1,
}

PLANNING_CONFIG = {
    "global_batch_size": 256,
    "attention_score_buffers": 0.0,
    # Peak-resident equivalent probability buffers. TE exposes two logical
    # saved tensors to the FL hook, but they do not contribute two independent
    # full-residency allocations in the tested zero-dropout path. Keep the
    # factor configurable for other stacks.
    "unfused_attention_probability_buffers": 1.0,
    # Optional module names accepted by --offload-modules:
    #   LayerNormLinear, GroupedLinear, swiglu, Attention, MLA, SharedExpert,
    #   MTP. Attention follows attention_backend and covers flash, fused, and
    #   unfused implementations. MTP covers its 2H -> H input;
    # its inner decoder layer reuses the MLA/attention/expert module names.
    # An empty tuple models baseline training with FL offload disabled.
    # "offload_modules": (),
    "offload_modules": ("Attention", "GroupedLinear", "swiglu"),
    "offload_min_tensor_bytes": 1 << 20,
    # Runtime budget for one activation group/layer. Runtime selects nothing
    # when the eligible captured size is smaller than this complete budget.
    "offload_mib_per_layer": 900,
    # An explicit reserve can exceed the automatically inferred landing buffer.
    "offload_gpu_buffer_mib": 0.0,
    "communication_buffer_factor": 0.0,
    "communication_buffer_mib": 0.0,
    # Calibrated against the reference 16-GPU H800 MLA/unfused run. Workspace
    # closes the analytical-to-max-allocated gap; overhead models the observed
    # CUDA allocator reserved/allocated gap.
    "workspace_mib": 2560.0,
    "overhead_percent": 14.0,
    "device_memory_gib": 0.0,
}

SCRIPT_CONFIG = {
    **MODEL_CONFIG,
    **PARALLEL_CONFIG,
    **PRECISION_CONFIG,
    **PLANNING_CONFIG,
}


@dataclass(frozen=True)
class MemoryModelConfig:
    layers: int
    hidden_size: int
    ffn_hidden_size: int
    vocab_size: int
    sequence_length: int
    micro_batch_size: int
    attention_heads: int
    kv_heads: int
    attention_backend: str = "flash"
    multi_latent_attention: bool = False
    q_lora_rank: int = 0
    kv_lora_rank: int = 0
    qk_head_dim: int = 0
    qk_pos_emb_head_dim: int = 0
    v_head_dim: int = 0
    qk_layernorm: bool = False
    mla_down_proj_fusion: bool = False
    experts: int = 1
    topk: int = 1
    moe_layers: int = 0
    moe_ffn_hidden_size: int = 0
    moe_shared_expert_intermediate_size: int = 0
    moe_shared_expert_gate: bool = False
    mtp_num_layers: int = 0
    mtp_layer_is_moe: bool = True
    capacity_factor: float = 1.0
    gated_mlp: bool = False
    untied_embeddings: bool = False
    world_size: int = 0
    tensor_parallel: int = 1
    pipeline_parallel: int = 1
    expert_parallel: int = 1
    expert_tensor_parallel: int = 1
    context_parallel: int = 1
    sequence_parallel: bool = False
    pipeline_schedule: str = "manual"
    layers_per_virtual_pipeline_stage: int = 0
    microbatch_group_size_per_vp_stage: int = 0
    overlap_moe_expert_parallel_comm: bool = False
    local_layers: int = 0
    local_moe_layers: int = -1
    inflight_microbatches: int = 1
    parameter_bytes: float = 2.0
    gradient_bytes: float = 4.0
    master_parameter_bytes: float = 4.0
    optimizer_state_bytes: float = 8.0
    use_distributed_optimizer: bool = False
    activation_bytes: float = 2.0
    lse_bytes: float = 4.0
    logit_bytes: float = 4.0
    parameter_shards: int = 1
    gradient_shards: int = 1
    optimizer_shards: int = 1
    expert_activation_shards: int = 1
    global_batch_size: int = 0
    attention_score_buffers: float = 0.0
    unfused_attention_probability_buffers: float = 2.0
    offload_modules: tuple = ()
    offload_min_tensor_bytes: int = 1 << 20
    offload_mib_per_layer: float = 0.0
    offload_gpu_buffer_mib: float = 0.0
    communication_buffer_factor: float = 0.0
    communication_buffer_mib: float = 0.0
    workspace_mib: float = 0.0
    overhead_percent: float = 10.0
    device_memory_gib: float = 0.0


def _positive(name, value):
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _parallel_layout(config):
    normal_model_parallel = (
        config.tensor_parallel * config.pipeline_parallel * config.context_parallel
    )
    expert_model_parallel = (
        config.expert_tensor_parallel
        * config.expert_parallel
        * config.pipeline_parallel
    )
    if normal_model_parallel > config.world_size:
        raise ValueError("TP * PP * CP cannot exceed world_size")
    if config.world_size % normal_model_parallel:
        raise ValueError("world_size must be divisible by TP * PP * CP")
    if expert_model_parallel > config.world_size:
        raise ValueError("ETP * EP * PP cannot exceed world_size")
    if config.world_size % expert_model_parallel:
        raise ValueError("world_size must be divisible by ETP * EP * PP")
    return {
        "data_parallel": config.world_size // normal_model_parallel,
        "expert_data_parallel": config.world_size // expert_model_parallel,
    }


def validate_config(config):
    for name in (
        "layers",
        "hidden_size",
        "ffn_hidden_size",
        "vocab_size",
        "sequence_length",
        "micro_batch_size",
        "attention_heads",
        "kv_heads",
        "experts",
        "topk",
        "tensor_parallel",
        "pipeline_parallel",
        "expert_parallel",
        "expert_tensor_parallel",
        "context_parallel",
        "parameter_shards",
        "gradient_shards",
        "optimizer_shards",
        "expert_activation_shards",
    ):
        _positive(name, getattr(config, name))
    if config.hidden_size % config.attention_heads:
        raise ValueError("hidden_size must be divisible by attention_heads")
    if config.attention_heads % config.tensor_parallel:
        raise ValueError("attention_heads must be divisible by tensor_parallel")
    if config.multi_latent_attention:
        for name in (
            "kv_lora_rank",
            "qk_head_dim",
            "qk_pos_emb_head_dim",
            "v_head_dim",
        ):
            _positive(name, getattr(config, name))
    else:
        if config.attention_heads % config.kv_heads:
            raise ValueError("attention_heads must be divisible by kv_heads")
        if config.kv_heads % config.tensor_parallel:
            raise ValueError("kv_heads must be divisible by tensor_parallel")
    if config.attention_backend not in {"flash", "fused", "unfused"}:
        raise ValueError("attention_backend must be flash, fused, or unfused")
    if config.attention_backend == "unfused" and config.context_parallel != 1:
        raise ValueError("unfused attention modeling currently requires context_parallel=1")
    if config.experts % config.expert_parallel:
        raise ValueError("experts must be divisible by expert_parallel")
    if config.topk > config.experts:
        raise ValueError("topk cannot exceed experts")
    if config.q_lora_rank < 0:
        raise ValueError("q_lora_rank cannot be negative")
    if config.moe_ffn_hidden_size < 0:
        raise ValueError("moe_ffn_hidden_size cannot be negative")
    if config.moe_shared_expert_intermediate_size < 0:
        raise ValueError("moe_shared_expert_intermediate_size cannot be negative")
    if config.mtp_num_layers < 0:
        raise ValueError("mtp_num_layers cannot be negative")
    if not 0 <= config.moe_layers <= config.layers:
        raise ValueError("moe_layers must be in [0, layers]")
    if config.local_layers < 0 or config.local_moe_layers < -1:
        raise ValueError("local layer overrides cannot be negative")
    if config.inflight_microbatches < 0:
        raise ValueError("inflight_microbatches cannot be negative")
    if config.layers_per_virtual_pipeline_stage < 0:
        raise ValueError("layers_per_virtual_pipeline_stage cannot be negative")
    if config.microbatch_group_size_per_vp_stage < 0:
        raise ValueError("microbatch_group_size_per_vp_stage cannot be negative")
    unknown_offload_modules = set(_normalized_offload_modules(config.offload_modules)) - set(
        OFFLOAD_MODULES
    )
    if unknown_offload_modules:
        raise ValueError(
            "unsupported offload modules: "
            + ", ".join(sorted(unknown_offload_modules))
        )
    schedules = {"manual", "1f1b", "interleaved-1f1b", "gpipe"}
    if config.pipeline_schedule not in schedules:
        raise ValueError(
            f"pipeline_schedule must be one of {sorted(schedules)}, "
            f"got {config.pipeline_schedule!r}"
        )
    needs_world = (
        config.pipeline_schedule != "manual" or config.use_distributed_optimizer
    )
    parallel_layout = None
    if needs_world:
        _positive("world_size", config.world_size)
        parallel_layout = _parallel_layout(config)
    elif config.world_size:
        parallel_layout = _parallel_layout(config)

    if config.pipeline_schedule == "manual":
        if config.inflight_microbatches == 0:
            raise ValueError("manual schedule requires inflight_microbatches > 0")
    else:
        _positive("global_batch_size", config.global_batch_size)
        data_parallel = parallel_layout["data_parallel"]
        batch_divisor = config.micro_batch_size * data_parallel
        if config.global_batch_size % batch_divisor:
            raise ValueError("global_batch_size must be divisible by MBS * DP")
    for name in (
        "capacity_factor",
        "parameter_bytes",
        "gradient_bytes",
        "master_parameter_bytes",
        "optimizer_state_bytes",
        "activation_bytes",
        "lse_bytes",
        "logit_bytes",
        "attention_score_buffers",
        "unfused_attention_probability_buffers",
        "offload_min_tensor_bytes",
        "offload_mib_per_layer",
        "offload_gpu_buffer_mib",
        "communication_buffer_factor",
        "communication_buffer_mib",
        "workspace_mib",
        "overhead_percent",
        "device_memory_gib",
    ):
        if getattr(config, name) < 0:
            raise ValueError(f"{name} cannot be negative")


def parameter_counts(config):
    head_dim = config.hidden_size // config.attention_heads
    kv_width = config.kv_heads * head_dim
    mlp_multiplier = 3 if config.gated_mlp else 2
    if config.multi_latent_attention:
        q_width = config.attention_heads * (
            config.qk_head_dim + config.qk_pos_emb_head_dim
        )
        kv_up_width = config.attention_heads * (
            config.qk_head_dim + config.v_head_dim
        )
        if config.q_lora_rank:
            q_projection = (
                config.hidden_size * config.q_lora_rank
                + config.q_lora_rank * q_width
            )
        else:
            q_projection = config.hidden_size * q_width
        kv_projection = (
            config.hidden_size
            * (config.kv_lora_rank + config.qk_pos_emb_head_dim)
            + config.kv_lora_rank * kv_up_width
        )
        output_projection = (
            config.attention_heads
            * config.v_head_dim
            * config.hidden_size
        )
        attention_per_layer = q_projection + kv_projection + output_projection
        mla_norm_per_layer = 0
        if config.qk_layernorm:
            mla_norm_per_layer = config.kv_lora_rank
            if config.q_lora_rank:
                mla_norm_per_layer += config.q_lora_rank
    else:
        attention_per_layer = config.hidden_size * (
            config.hidden_size + 2 * kv_width
        ) + config.hidden_size**2
        mla_norm_per_layer = 0
    norm_per_layer = 2 * config.hidden_size + mla_norm_per_layer
    dense_mlp_per_layer = mlp_multiplier * config.hidden_size * config.ffn_hidden_size
    moe_ffn_hidden_size = config.moe_ffn_hidden_size or config.ffn_hidden_size
    routed_expert_per_layer = (
        mlp_multiplier * config.hidden_size * moe_ffn_hidden_size
    )
    expert_mlp_per_layer = config.experts * routed_expert_per_layer
    shared_expert_per_layer = (
        mlp_multiplier
        * config.hidden_size
        * config.moe_shared_expert_intermediate_size
    )
    if config.moe_shared_expert_gate:
        shared_expert_per_layer += config.hidden_size
    router_per_moe_layer = config.hidden_size * config.experts
    mtp_projection_per_layer = 2 * config.hidden_size**2
    mtp_extra_norm_per_layer = 3 * config.hidden_size
    mtp_shared_per_layer = (
        attention_per_layer
        + norm_per_layer
        + mtp_projection_per_layer
        + mtp_extra_norm_per_layer
    )
    mtp_expert_per_layer = 0
    mtp_active_expert_per_layer = 0
    if config.mtp_layer_is_moe:
        mtp_shared_per_layer += shared_expert_per_layer + router_per_moe_layer
        mtp_expert_per_layer = expert_mlp_per_layer
        mtp_active_expert_per_layer = config.topk * routed_expert_per_layer
    else:
        mtp_shared_per_layer += dense_mlp_per_layer
    dense_layers = config.layers - config.moe_layers
    embeddings = config.hidden_size * config.vocab_size
    output = embeddings if config.untied_embeddings else 0
    final_norm = config.hidden_size
    components = {
        "embeddings": embeddings,
        "output_projection": output,
        "attention": attention_per_layer * config.layers,
        "normalization": norm_per_layer * config.layers + final_norm,
        "dense_mlp": dense_mlp_per_layer * dense_layers,
        "expert_mlp": expert_mlp_per_layer * config.moe_layers,
        "shared_expert_mlp": shared_expert_per_layer * config.moe_layers,
        "router": router_per_moe_layer * config.moe_layers,
        "mtp_shared": mtp_shared_per_layer * config.mtp_num_layers,
        "mtp_expert": mtp_expert_per_layer * config.mtp_num_layers,
    }
    active_components = dict(components)
    active_expert_mlp_per_layer = config.topk * routed_expert_per_layer
    active_components["expert_mlp"] = (
        active_expert_mlp_per_layer * config.moe_layers
    )
    active_components["mtp_expert"] = (
        mtp_active_expert_per_layer * config.mtp_num_layers
    )
    return {
        "components": components,
        "total": sum(components.values()),
        "active_components": active_components,
        "active_total": sum(active_components.values()),
        "per_layer": {
            "attention": attention_per_layer,
            "normalization": norm_per_layer,
            "dense_mlp": dense_mlp_per_layer,
            "routed_expert_mlp": expert_mlp_per_layer,
            "expert_mlp": expert_mlp_per_layer,
            "active_expert_mlp": active_expert_mlp_per_layer,
            "shared_expert_mlp": shared_expert_per_layer,
            "router": router_per_moe_layer,
            "mtp_shared": mtp_shared_per_layer,
            "mtp_expert": mtp_expert_per_layer,
            "mtp_active_expert": mtp_active_expert_per_layer,
            "mtp_projection": mtp_projection_per_layer,
            "mtp_extra_normalization": mtp_extra_norm_per_layer,
        },
    }


def local_layer_counts(config):
    local_layers = config.local_layers or math.ceil(config.layers / config.pipeline_parallel)
    if local_layers > config.layers:
        raise ValueError("local_layers cannot exceed layers")
    if config.local_moe_layers >= 0:
        local_moe_layers = config.local_moe_layers
    elif config.moe_layers == 0:
        local_moe_layers = 0
    elif config.moe_layers == config.layers:
        local_moe_layers = local_layers
    else:
        local_moe_layers = math.ceil(local_layers * config.moe_layers / config.layers)
    if not 0 <= local_moe_layers <= local_layers:
        raise ValueError("local_moe_layers must be in [0, local_layers]")
    return local_layers, local_moe_layers


def _schedule_table(num_microbatches, num_model_chunks, group_size):
    table = []
    for first_microbatch in range(0, num_microbatches, group_size):
        last_microbatch = min(first_microbatch + group_size, num_microbatches)
        table.extend(
            chunk_id
            for chunk_id in range(num_model_chunks)
            for _microbatch_id in range(first_microbatch, last_microbatch)
        )
    return table


def _maximum_residency(forward_chunks, warmup, chunk_layers):
    num_chunks = len(chunk_layers)
    resident = [0] * num_chunks
    peak_resident = list(resident)
    peak_layers = 0
    peak_virtual = 0

    def update_peak():
        nonlocal peak_layers, peak_resident, peak_virtual
        resident_layers = sum(
            count * layers for count, layers in zip(resident, chunk_layers)
        )
        resident_virtual = sum(resident)
        if resident_layers > peak_layers:
            peak_layers = resident_layers
            peak_virtual = resident_virtual
            peak_resident = list(resident)

    total = len(forward_chunks)
    for virtual_microbatch_id in range(warmup):
        resident[forward_chunks[virtual_microbatch_id]] += 1
        update_peak()

    remaining = total - warmup
    for step in range(remaining):
        forward_id = warmup + step
        resident[forward_chunks[forward_id]] += 1
        update_peak()

        backward_chunk = num_chunks - forward_chunks[step] - 1
        resident[backward_chunk] -= 1
        if resident[backward_chunk] < 0:
            raise RuntimeError("invalid pipeline residency simulation")

    for backward_id in range(remaining, total):
        backward_chunk = num_chunks - forward_chunks[backward_id] - 1
        resident[backward_chunk] -= 1
        if resident[backward_chunk] < 0:
            raise RuntimeError("invalid pipeline cooldown simulation")
    if any(resident):
        raise RuntimeError("pipeline residency simulation did not drain")
    return peak_virtual, peak_layers, peak_resident


def schedule_residency(config, local_layers):
    """Infer the maximum saved-activation residency for every PP rank."""
    if config.inflight_microbatches > 0:
        layer_equivalents = local_layers * config.inflight_microbatches
        ranks = [
            {
                "pp_rank": rank,
                "warmup_virtual_microbatches": None,
                "max_outstanding_virtual_microbatches": None,
                "max_resident_by_chunk": None,
                "max_resident_layer_equivalents": layer_equivalents,
                "activation_batches": float(config.inflight_microbatches),
            }
            for rank in range(config.pipeline_parallel)
        ]
        return {
            "schedule": "manual",
            "manual_override": True,
            "data_parallel": None,
            "num_microbatches": None,
            "virtual_pipeline_chunks": 1,
            "microbatch_group_size_per_vp_stage": None,
            "ranks": ranks,
        }

    data_parallel = _parallel_layout(config)["data_parallel"]
    num_microbatches = config.global_batch_size // (
        config.micro_batch_size * data_parallel
    )

    if config.pipeline_schedule == "interleaved-1f1b":
        _positive(
            "layers_per_virtual_pipeline_stage",
            config.layers_per_virtual_pipeline_stage,
        )
        if local_layers % config.layers_per_virtual_pipeline_stage:
            raise ValueError(
                "local_layers must be divisible by layers_per_virtual_pipeline_stage"
            )
        num_chunks = local_layers // config.layers_per_virtual_pipeline_stage
        if num_chunks < 2:
            raise ValueError("interleaved-1f1b requires at least two virtual chunks")
        group_size = config.microbatch_group_size_per_vp_stage or config.pipeline_parallel
        if not config.pipeline_parallel <= group_size <= num_microbatches:
            raise ValueError("VP microbatch group size must be in [PP, num_microbatches]")
        final_group = num_microbatches % group_size
        if 0 < final_group < config.pipeline_parallel:
            raise ValueError("final VP microbatch group must be zero or at least PP")
        chunk_layers = [config.layers_per_virtual_pipeline_stage] * num_chunks
        forward_chunks = _schedule_table(num_microbatches, num_chunks, group_size)
    else:
        num_chunks = 1
        group_size = None
        chunk_layers = [local_layers]
        forward_chunks = [0] * num_microbatches

    total_virtual_microbatches = len(forward_chunks)
    ranks = []
    for rank in range(config.pipeline_parallel):
        if config.pipeline_schedule == "gpipe":
            warmup = total_virtual_microbatches
        elif config.pipeline_schedule == "interleaved-1f1b":
            warmup = (config.pipeline_parallel - rank - 1) * 2
            warmup += (num_chunks - 1) * group_size
            if config.overlap_moe_expert_parallel_comm:
                warmup += 1
            warmup = min(warmup, total_virtual_microbatches)
        else:
            warmup = min(
                config.pipeline_parallel - rank - 1,
                total_virtual_microbatches,
            )
        peak_virtual, peak_layers, peak_by_chunk = _maximum_residency(
            forward_chunks, warmup, chunk_layers
        )
        ranks.append(
            {
                "pp_rank": rank,
                "warmup_virtual_microbatches": warmup,
                "max_outstanding_virtual_microbatches": peak_virtual,
                "max_resident_by_chunk": peak_by_chunk,
                "max_resident_layer_equivalents": peak_layers,
                "activation_batches": peak_layers / local_layers,
            }
        )
    return {
        "schedule": config.pipeline_schedule,
        "manual_override": False,
        "data_parallel": data_parallel,
        "num_microbatches": num_microbatches,
        "virtual_pipeline_chunks": num_chunks,
        "microbatch_group_size_per_vp_stage": group_size,
        "ranks": ranks,
    }


def _tensor_record(module, tensor, size, offload_supported=False):
    return {
        "module": module,
        "tensor": tensor,
        "bytes": size,
        "offload_supported": offload_supported,
    }


def _module_totals(records):
    totals = {}
    for record in records:
        totals[record["module"]] = totals.get(record["module"], 0) + record["bytes"]
    return totals


def _offload_selection(config, records):
    requested_modules = set(_normalized_offload_modules(config.offload_modules))
    eligible = [
        record
        for record in records
        if record["offload_supported"]
        and record["module"] in requested_modules
        and record["bytes"] >= config.offload_min_tensor_bytes
    ]
    # ActivationGroup sorts contiguous tensors largest-first before applying
    # its single byte budget. Model tensors are all treated as contiguous.
    eligible.sort(key=lambda record: -record["bytes"])
    captured_by_module = _module_totals(eligible)
    captured = sum(record["bytes"] for record in eligible)
    budget = config.offload_mib_per_layer * MIB
    selected = budget if budget > 0 and captured >= budget else 0
    remaining = selected
    selected_by_module = {module: 0 for module in captured_by_module}
    for record in eligible:
        amount = min(record["bytes"], remaining)
        selected_by_module[record["module"]] += amount
        remaining -= amount
        if remaining <= 0:
            break
    return {
        "captured": captured,
        "selected": selected,
        "captured_by_module": captured_by_module,
        "selected_by_module": selected_by_module,
    }


def activation_bytes(config):
    local_layers, local_moe_layers = local_layer_counts(config)
    local_dense_layers = local_layers - local_moe_layers
    head_dim = config.hidden_size // config.attention_heads
    kv_width = config.kv_heads * head_dim
    moe_ffn_hidden_size = config.moe_ffn_hidden_size or config.ffn_hidden_size
    tokens = math.ceil(config.sequence_length / config.context_parallel) * config.micro_batch_size
    sequence_token_shards = config.tensor_parallel if config.sequence_parallel else 1
    sequence_tokens = math.ceil(tokens / sequence_token_shards)

    layernorm_tensor = sequence_tokens * config.hidden_size * config.activation_bytes
    if config.multi_latent_attention:
        heads_per_rank = config.attention_heads / config.tensor_parallel
        qk_width_per_rank = heads_per_rank * (
            config.qk_head_dim + config.qk_pos_emb_head_dim
        )
        value_width_per_rank = heads_per_rank * config.v_head_dim
        projection_records = [
            _tensor_record(
                "MLA",
                "projection_input",
                layernorm_tensor,
                offload_supported=True,
            ),
            _tensor_record(
                "MLA",
                "kv_up_input",
                sequence_tokens
                * config.kv_lora_rank
                * config.activation_bytes,
                offload_supported=True,
            ),
        ]
        if config.mla_down_proj_fusion:
            projection_records.append(
                _tensor_record(
                    "MLA",
                    "projection_input.ln_out",
                    layernorm_tensor,
                    offload_supported=True,
                )
            )
        if config.q_lora_rank:
            projection_records.append(
                _tensor_record(
                    "MLA",
                    "q_up_input",
                    sequence_tokens
                    * config.q_lora_rank
                    * config.activation_bytes,
                    offload_supported=True,
                )
            )
        if config.qk_layernorm:
            projection_records.append(
                _tensor_record(
                    "MLANorm",
                    "kv_input",
                    sequence_tokens
                    * config.kv_lora_rank
                    * config.activation_bytes,
                )
            )
            if config.q_lora_rank:
                projection_records.append(
                    _tensor_record(
                        "MLANorm",
                        "q_input",
                        sequence_tokens
                        * config.q_lora_rank
                        * config.activation_bytes,
                    )
                )
        attention_records = [
            _tensor_record(
                "Attention",
                "q",
                tokens * qk_width_per_rank * config.activation_bytes,
                offload_supported=True,
            ),
            _tensor_record(
                "Attention",
                "k",
                tokens * qk_width_per_rank * config.activation_bytes,
                offload_supported=True,
            ),
            _tensor_record(
                "Attention",
                "v",
                tokens * value_width_per_rank * config.activation_bytes,
                offload_supported=True,
            ),
            _tensor_record(
                "Attention",
                "output",
                tokens * value_width_per_rank * config.activation_bytes,
                offload_supported=True,
            ),
            _tensor_record(
                "Attention",
                "softmax_lse",
                tokens
                * config.attention_heads
                * config.lse_bytes
                / config.tensor_parallel,
                offload_supported=True,
            ),
        ]
        if config.attention_backend == "fused":
            attention_records = [
                _tensor_record(
                    "Attention",
                    "q",
                    tokens * qk_width_per_rank * config.activation_bytes,
                    offload_supported=True,
                ),
                _tensor_record(
                    "Attention",
                    "k",
                    tokens * qk_width_per_rank * config.activation_bytes,
                    offload_supported=True,
                ),
                _tensor_record(
                    "Attention",
                    "v",
                    tokens * value_width_per_rank * config.activation_bytes,
                    offload_supported=True,
                ),
                _tensor_record(
                    "Attention",
                    "output",
                    tokens * value_width_per_rank * config.activation_bytes,
                    offload_supported=True,
                ),
                _tensor_record(
                    "Attention",
                    "softmax_stats",
                    tokens
                    * config.attention_heads
                    * config.lse_bytes
                    / config.tensor_parallel,
                    offload_supported=True,
                ),
            ]
        elif config.attention_backend == "unfused":
            attention_records = [
                _tensor_record(
                    "Attention",
                    "q",
                    tokens * qk_width_per_rank * config.activation_bytes,
                    offload_supported=True,
                ),
                _tensor_record(
                    "Attention",
                    "k",
                    tokens * qk_width_per_rank * config.activation_bytes,
                    offload_supported=True,
                ),
                _tensor_record(
                    "Attention",
                    "v",
                    tokens * value_width_per_rank * config.activation_bytes,
                    offload_supported=True,
                ),
                _tensor_record(
                    "Attention",
                    "output",
                    tokens * value_width_per_rank * config.activation_bytes,
                    offload_supported=True,
                ),
                _tensor_record(
                    "Attention",
                    "attention_probs",
                    config.unfused_attention_probability_buffers
                    * config.micro_batch_size
                    * heads_per_rank
                    * math.ceil(config.sequence_length / config.context_parallel) ** 2
                    * config.activation_bytes,
                    offload_supported=True,
                ),
            ]
    else:
        projection_records = [
            _tensor_record(
                "LayerNormLinear", "inputmat", layernorm_tensor, offload_supported=True
            ),
            _tensor_record(
                "LayerNormLinear", "ln_out", layernorm_tensor, offload_supported=True
            ),
        ]
        attention_records = [
            _tensor_record(
                "Attention",
                "q",
                tokens * config.hidden_size * config.activation_bytes / config.tensor_parallel,
                offload_supported=True,
            ),
            _tensor_record(
                "Attention",
                "k",
                tokens * kv_width * config.activation_bytes / config.tensor_parallel,
                offload_supported=True,
            ),
            _tensor_record(
                "Attention",
                "v",
                tokens * kv_width * config.activation_bytes / config.tensor_parallel,
                offload_supported=True,
            ),
            _tensor_record(
                "Attention",
                "output",
                tokens * config.hidden_size * config.activation_bytes / config.tensor_parallel,
                offload_supported=True,
            ),
            _tensor_record(
                "Attention",
                "softmax_lse",
                tokens
                * config.attention_heads
                * config.lse_bytes
                / config.tensor_parallel,
                offload_supported=True,
            ),
        ]
        if config.attention_backend == "fused":
            attention_records = [
                _tensor_record(
                    "Attention",
                    "q",
                    tokens * config.hidden_size * config.activation_bytes / config.tensor_parallel,
                    offload_supported=True,
                ),
                _tensor_record(
                    "Attention",
                    "k",
                    tokens * kv_width * config.activation_bytes / config.tensor_parallel,
                    offload_supported=True,
                ),
                _tensor_record(
                    "Attention",
                    "v",
                    tokens * kv_width * config.activation_bytes / config.tensor_parallel,
                    offload_supported=True,
                ),
                _tensor_record(
                    "Attention",
                    "output",
                    tokens * config.hidden_size * config.activation_bytes / config.tensor_parallel,
                    offload_supported=True,
                ),
                _tensor_record(
                    "Attention",
                    "softmax_stats",
                    tokens
                    * config.attention_heads
                    * config.lse_bytes
                    / config.tensor_parallel,
                    offload_supported=True,
                ),
            ]
        elif config.attention_backend == "unfused":
            attention_records = [
                _tensor_record(
                    "Attention",
                    "q",
                    tokens * config.hidden_size * config.activation_bytes / config.tensor_parallel,
                    offload_supported=True,
                ),
                _tensor_record(
                    "Attention",
                    "k",
                    tokens * config.hidden_size * config.activation_bytes / config.tensor_parallel,
                    offload_supported=True,
                ),
                _tensor_record(
                    "Attention",
                    "v",
                    tokens * config.hidden_size * config.activation_bytes / config.tensor_parallel,
                    offload_supported=True,
                ),
                _tensor_record(
                    "Attention",
                    "output",
                    tokens * config.hidden_size * config.activation_bytes / config.tensor_parallel,
                    offload_supported=True,
                ),
                _tensor_record(
                    "Attention",
                    "attention_probs",
                    config.unfused_attention_probability_buffers
                    * config.micro_batch_size
                    * (config.attention_heads / config.tensor_parallel)
                    * math.ceil(config.sequence_length / config.context_parallel) ** 2
                    * config.activation_bytes,
                    offload_supported=True,
                ),
            ]
    common_records = projection_records + attention_records
    qkv_linear = sum(record["bytes"] for record in projection_records)
    core_attention = sum(
        record["bytes"]
        for record in attention_records
    )
    attention_scores = (
        config.attention_score_buffers
        * config.micro_batch_size
        * config.attention_heads
        * math.ceil(config.sequence_length / config.context_parallel) ** 2
        * config.activation_bytes
        / config.tensor_parallel
    )
    attention_per_layer = qkv_linear + core_attention + attention_scores

    # The All-to-All dispatcher receives S/TP tokens when sequence parallelism
    # is enabled. In padded-capacity mode, each local expert processes
    # capacity * ETP * EP tokens and each rank owns experts / EP experts.
    expert_capacity = math.ceil(
        sequence_tokens * config.topk * config.capacity_factor / config.experts
    )
    routed_tokens = expert_capacity * config.expert_tensor_parallel * config.experts
    dense_records = [
        _tensor_record(
            "DenseMLP",
            "fc1_input",
            tokens
            * config.hidden_size
            * config.activation_bytes
            / sequence_token_shards,
        ),
        _tensor_record(
            "DenseMLP",
            "activation_input",
            tokens
            * (2 if config.gated_mlp else 1)
            * config.ffn_hidden_size
            * config.activation_bytes
            / config.tensor_parallel,
        ),
        _tensor_record(
            "DenseMLP",
            "fc2_input",
            tokens
            * config.ffn_hidden_size
            * config.activation_bytes
            / config.tensor_parallel,
        ),
    ]
    expert_records = [
        _tensor_record(
            "GroupedLinear",
            "expert_fc1_input",
            routed_tokens
            * config.hidden_size
            * config.activation_bytes
            / config.expert_activation_shards,
            offload_supported=True,
        ),
        _tensor_record(
            "swiglu" if config.gated_mlp else "ExpertActivation",
            "input",
            routed_tokens
            * (2 if config.gated_mlp else 1)
            * moe_ffn_hidden_size
            * config.activation_bytes
            / config.expert_activation_shards,
            offload_supported=config.gated_mlp,
        ),
        _tensor_record(
            "GroupedLinear",
            "expert_fc2_input",
            routed_tokens
            * moe_ffn_hidden_size
            * config.activation_bytes
            / config.expert_activation_shards,
            offload_supported=True,
        ),
    ]
    shared_expert_records = []
    if config.moe_shared_expert_intermediate_size:
        shared_expert_records = [
            _tensor_record(
                "SharedExpert",
                "fc1_input",
                sequence_tokens
                * config.hidden_size
                * config.activation_bytes,
                offload_supported=True,
            ),
            _tensor_record(
                "SharedExpert",
                "swiglu_input" if config.gated_mlp else "activation_input",
                tokens
                * (2 if config.gated_mlp else 1)
                * config.moe_shared_expert_intermediate_size
                * config.activation_bytes
                / config.tensor_parallel,
                offload_supported=config.gated_mlp,
            ),
            _tensor_record(
                "SharedExpert",
                "fc2_input",
                tokens
                * config.moe_shared_expert_intermediate_size
                * config.activation_bytes
                / config.tensor_parallel,
                offload_supported=True,
            ),
        ]
    dense_mlp_per_layer = sum(record["bytes"] for record in dense_records)
    routed_expert_mlp_per_layer = sum(record["bytes"] for record in expert_records)
    shared_expert_mlp_per_layer = sum(
        record["bytes"] for record in shared_expert_records
    )
    expert_mlp_per_layer = (
        routed_expert_mlp_per_layer + shared_expert_mlp_per_layer
    )
    mtp_records = [
        _tensor_record(
            "MTP",
            "enorm_input",
            layernorm_tensor,
        ),
        _tensor_record(
            "MTP",
            "hnorm_input",
            layernorm_tensor,
        ),
        _tensor_record(
            "MTP",
            "eh_proj_input",
            2 * layernorm_tensor,
            offload_supported=True,
        ),
        _tensor_record(
            "MTP",
            "final_norm_input",
            layernorm_tensor,
        ),
    ]
    mtp_inner_records = (
        expert_records + shared_expert_records
        if config.mtp_layer_is_moe
        else dense_records
    )
    mtp_inner_mlp_per_layer = (
        expert_mlp_per_layer if config.mtp_layer_is_moe else dense_mlp_per_layer
    )
    mtp_per_layer = (
        sum(record["bytes"] for record in mtp_records)
        + attention_per_layer
        + mtp_inner_mlp_per_layer
    )

    raw_per_microbatch = (
        attention_per_layer * local_layers
        + dense_mlp_per_layer * local_dense_layers
        + expert_mlp_per_layer * local_moe_layers
    )
    dense_selection = _offload_selection(config, common_records + dense_records)
    expert_selection = _offload_selection(
        config, common_records + expert_records + shared_expert_records
    )
    mtp_layer_records = mtp_records + common_records + mtp_inner_records
    mtp_selection = _offload_selection(config, mtp_layer_records)
    final_decoder_records = common_records + (
        expert_records + shared_expert_records
        if config.mtp_layer_is_moe
        else dense_records
    )
    # The last decoder chunk and every MTP depth execute inside one scheduler
    # record. They therefore share one fixed runtime budget rather than each
    # receiving an independent budget.
    mtp_group_records = final_decoder_records + (
        mtp_layer_records * config.mtp_num_layers
    )
    mtp_group_candidate = _offload_selection(config, mtp_group_records)
    requested_modules = set(_normalized_offload_modules(config.offload_modules))
    mtp_group_scheduled = bool(config.mtp_num_layers and "MTP" in requested_modules)
    mtp_group_selection = {
        **mtp_group_candidate,
        "scheduled": mtp_group_scheduled,
        "candidate_captured": mtp_group_candidate["captured"],
        "candidate_selected": mtp_group_candidate["selected"],
    }
    if not mtp_group_scheduled:
        # combined_1f1b deliberately skips the final PP/VPP chunk unless MTP
        # is selected as the schedule-enabling module. Keep candidate values
        # for diagnostics, but report no runtime capture or selection.
        mtp_group_selection.update(
            captured=0,
            selected=0,
            captured_by_module={},
            selected_by_module={},
        )
    effective_offload = (
        dense_selection["selected"] * local_dense_layers
        + expert_selection["selected"] * local_moe_layers
    )
    active_selections = []
    if local_dense_layers:
        active_selections.append(dense_selection["selected"])
    if local_moe_layers:
        active_selections.append(expert_selection["selected"])
    if mtp_group_scheduled:
        active_selections.append(mtp_group_selection["selected"])
    inferred_landing_buffer = max(active_selections, default=0)

    embedding_misc = tokens * (8 + config.hidden_size)
    output_depths = 1 + config.mtp_num_layers
    final_output = (
        tokens
        * output_depths
        * (
            config.hidden_size * config.activation_bytes
            + config.vocab_size * config.logit_bytes / config.tensor_parallel
        )
    )
    return {
        "local_layers": local_layers,
        "local_moe_layers": local_moe_layers,
        "tokens_per_rank": tokens,
        "router_tokens_per_rank": sequence_tokens,
        "expert_capacity": expert_capacity,
        "routed_tokens_per_rank": routed_tokens,
        "module_tensors": {
            "common": common_records,
            "dense_mlp": dense_records,
            "expert_mlp": expert_records + shared_expert_records,
            "mtp": mtp_layer_records,
        },
        "module_totals_per_layer": {
            "common": _module_totals(common_records),
            "dense_mlp": _module_totals(dense_records),
            "expert_mlp": _module_totals(expert_records + shared_expert_records),
            "mtp": _module_totals(mtp_layer_records),
        },
        "offload": {
            "requested_modules": list(_normalized_offload_modules(config.offload_modules)),
            "budget_per_layer": config.offload_mib_per_layer * MIB,
            "minimum_tensor_bytes": config.offload_min_tensor_bytes,
            "inferred_landing_buffer": inferred_landing_buffer,
            "dense_layer": dense_selection,
            "expert_layer": expert_selection,
            "mtp_layer": mtp_selection,
            "final_decoder_and_mtp_group": mtp_group_selection,
        },
        "per_layer": {
            "qkv_linear": qkv_linear,
            "core_attention": core_attention,
            "attention_scores": attention_scores,
            "dense_mlp": dense_mlp_per_layer,
            "routed_expert_mlp": routed_expert_mlp_per_layer,
            "shared_expert_mlp": shared_expert_mlp_per_layer,
            "expert_mlp": expert_mlp_per_layer,
            "mtp": mtp_per_layer,
        },
        "raw_per_microbatch": raw_per_microbatch,
        "effective_offload_per_microbatch": effective_offload,
        "mtp_raw_per_microbatch": mtp_per_layer * config.mtp_num_layers,
        "embedding_misc": embedding_misc,
        "final_output": final_output,
    }


def _base_local_parameters(config, counts, local_layers, local_moe_layers):
    per_layer = counts["per_layer"]
    local_dense_layers = local_layers - local_moe_layers
    shared = local_layers * (
        per_layer["attention"] / config.tensor_parallel + per_layer["normalization"]
    )
    shared += local_moe_layers * per_layer["router"] / config.tensor_parallel
    shared += (
        local_moe_layers
        * per_layer["shared_expert_mlp"]
        / config.tensor_parallel
    )
    dense = local_dense_layers * per_layer["dense_mlp"] / config.tensor_parallel
    experts = (
        local_moe_layers
        * per_layer["routed_expert_mlp"]
        / (config.expert_parallel * config.expert_tensor_parallel)
    )
    return {"shared": shared + dense, "experts": experts}


def _optimizer_parallelism(config):
    if not config.use_distributed_optimizer:
        return {
            "enabled": False,
            "data_parallel": None,
            "expert_data_parallel": None,
            "shared_optimizer_shards": config.optimizer_shards,
            "expert_optimizer_shards": config.optimizer_shards,
        }

    layout = _parallel_layout(config)
    data_parallel = layout["data_parallel"]
    expert_data_parallel = layout["expert_data_parallel"]
    return {
        "enabled": True,
        "data_parallel": data_parallel,
        "expert_data_parallel": expert_data_parallel,
        "shared_optimizer_shards": data_parallel,
        "expert_optimizer_shards": expert_data_parallel,
    }


def _state_components(config, shared_parameters, expert_parameters, optimizer_parallelism):
    local_parameters = shared_parameters + expert_parameters
    shared_optimizer_shards = optimizer_parallelism["shared_optimizer_shards"]
    expert_optimizer_shards = optimizer_parallelism["expert_optimizer_shards"]
    sharded_optimizer_parameters = (
        shared_parameters / shared_optimizer_shards
        + expert_parameters / expert_optimizer_shards
    )
    return {
        "model_parameters": local_parameters * config.parameter_bytes / config.parameter_shards,
        "gradients": local_parameters * config.gradient_bytes / config.gradient_shards,
        "master_parameters": sharded_optimizer_parameters * config.master_parameter_bytes,
        "optimizer_states": sharded_optimizer_parameters * config.optimizer_state_bytes,
    }


def estimate_memory(config):
    validate_config(config)
    counts = parameter_counts(config)
    activations = activation_bytes(config)
    local_layers = activations["local_layers"]
    local_moe_layers = activations["local_moe_layers"]
    schedule = schedule_residency(config, local_layers)
    schedule_by_rank = {item["pp_rank"]: item for item in schedule["ranks"]}
    base_parameters = _base_local_parameters(
        config, counts, local_layers, local_moe_layers
    )
    optimizer_parallelism = _optimizer_parallelism(config)
    embedding_parameters = config.hidden_size * config.vocab_size / config.tensor_parallel
    final_norm_parameters = config.hidden_size

    if config.pipeline_parallel == 1:
        endpoint_parameters = embedding_parameters * (2 if config.untied_embeddings else 1)
        roles = [
            (
                "single",
                0,
                endpoint_parameters + final_norm_parameters,
                activations["final_output"],
            )
        ]
    else:
        roles = [
            ("first", 0, embedding_parameters, activations["embedding_misc"]),
            (
                "last",
                config.pipeline_parallel - 1,
                embedding_parameters + final_norm_parameters,
                activations["final_output"],
            ),
        ]
        if config.pipeline_parallel > 2:
            # Rank 1 has the largest warmup among homogeneous middle stages.
            roles.append(("middle", 1, 0, 0))

    candidates = []
    for role, pp_rank, endpoint_parameters, endpoint_activations in roles:
        rank_schedule = schedule_by_rank[pp_rank]
        has_mtp = role in {"single", "last"} and config.mtp_num_layers > 0
        mtp_shared_parameters = 0
        mtp_expert_parameters = 0
        if has_mtp:
            mtp_shared_parameters = (
                counts["per_layer"]["mtp_shared"]
                * config.mtp_num_layers
                / config.tensor_parallel
            )
            mtp_expert_parameters = (
                counts["per_layer"]["mtp_expert"]
                * config.mtp_num_layers
                / (config.expert_parallel * config.expert_tensor_parallel)
            )
        shared_parameters = (
            base_parameters["shared"] + endpoint_parameters + mtp_shared_parameters
        )
        expert_parameters = base_parameters["experts"] + mtp_expert_parameters
        local_parameters = shared_parameters + expert_parameters
        states = _state_components(
            config,
            shared_parameters,
            expert_parameters,
            optimizer_parallelism,
        )
        communication = (
            local_parameters * config.parameter_bytes * config.communication_buffer_factor
            + config.communication_buffer_mib * MIB
        )
        selected_per_layer = (
            activations["effective_offload_per_microbatch"] / local_layers
        )
        offload_layer_equivalents = rank_schedule["max_resident_layer_equivalents"]
        # combined_1f1b normally leaves the final model chunk on the last PP
        # rank resident. Selecting MTP explicitly enables that group, which
        # contains the final decoder chunk and every MTP depth under one budget.
        resident_by_chunk = rank_schedule["max_resident_by_chunk"]
        mtp_offload_enabled = (
            has_mtp
            and activations["offload"]["final_decoder_and_mtp_group"]["scheduled"]
        )
        final_chunk_residency = 0
        if (
            config.pipeline_parallel > 1
            and pp_rank == config.pipeline_parallel - 1
            and resident_by_chunk
        ):
            last_chunk_layers = (
                config.layers_per_virtual_pipeline_stage
                if config.pipeline_schedule == "interleaved-1f1b"
                else local_layers
            )
            final_chunk_residency = resident_by_chunk[-1]
            offload_layer_equivalents -= final_chunk_residency * last_chunk_layers
        offload_layer_equivalents = max(0, offload_layer_equivalents)
        decoder_offload_savings = selected_per_layer * offload_layer_equivalents
        mtp_offload_savings = 0
        if mtp_offload_enabled and final_chunk_residency:
            mtp_offload_savings = (
                activations["offload"]["final_decoder_and_mtp_group"]["selected"]
                * final_chunk_residency
            )
        offload_savings = decoder_offload_savings + mtp_offload_savings
        offload_gpu_buffer = config.offload_gpu_buffer_mib * MIB
        if offload_savings:
            offload_gpu_buffer = max(
                offload_gpu_buffer,
                activations["offload"]["inferred_landing_buffer"],
            )
        baseline_saved_activations = (
            activations["raw_per_microbatch"]
            * rank_schedule["activation_batches"]
        )
        if has_mtp:
            baseline_saved_activations += (
                activations["mtp_raw_per_microbatch"]
                * rank_schedule["activation_batches"]
            )
        components = {
            **states,
            "saved_activations": baseline_saved_activations - offload_savings,
            "endpoint_activations": endpoint_activations,
            "communication_buffers": communication,
            "operator_workspace": config.workspace_mib * MIB,
            "offload_gpu_buffer": offload_gpu_buffer,
        }
        subtotal = sum(components.values())
        baseline_subtotal = (
            subtotal
            - components["saved_activations"]
            - components["offload_gpu_buffer"]
            + baseline_saved_activations
        )
        components["allocator_and_unmodeled_overhead"] = (
            baseline_subtotal * config.overhead_percent / 100
        )
        total = sum(components.values())
        candidates.append(
            {
                "role": role,
                "pp_rank": pp_rank,
                "local_parameters": local_parameters,
                "local_shared_parameters": shared_parameters,
                "local_expert_parameters": expert_parameters,
                "offload_layer_equivalents": offload_layer_equivalents,
                "offload_savings": offload_savings,
                "schedule": rank_schedule,
                "components": components,
                "total": total,
            }
        )

    peak = max(candidates, key=lambda item: item["total"])
    activations["resident_peak"] = max(
        item["components"]["saved_activations"] for item in candidates
    )
    device_bytes = config.device_memory_gib * GIB
    return {
        "config": asdict(config),
        "global_parameters": counts,
        "activation_model": activations,
        "schedule_model": schedule,
        "optimizer_model": optimizer_parallelism,
        "stage_candidates": candidates,
        "peak_stage": peak,
        "device_memory_bytes": device_bytes,
        "headroom_bytes": device_bytes - peak["total"] if device_bytes else None,
    }


def _format_count(value):
    return f"{value / 1e9:.3f} B"


def _format_memory(value):
    return f"{value / MIB:.2f} MiB"


def _format_activation_memory(value):
    return f"{value / MIB:.2f} MiB"


def print_report(estimate):
    config = estimate["config"]
    peak = estimate["peak_stage"]
    activation = estimate["activation_model"]
    schedule = estimate["schedule_model"]
    optimizer = estimate["optimizer_model"]
    print("Framework-independent peak memory model")
    print(
        f"model: layers={config['layers']} hidden={config['hidden_size']} "
        f"ffn={config['ffn_hidden_size']} vocab={config['vocab_size']} "
        f"attention={config['attention_backend']}"
    )
    print(
        f"parallel: TP={config['tensor_parallel']} PP={config['pipeline_parallel']} "
        f"EP={config['expert_parallel']} ETP={config['expert_tensor_parallel']} "
        f"CP={config['context_parallel']}"
    )
    print(
        f"schedule: {schedule['schedule']} DP={schedule['data_parallel']} "
        f"microbatches={schedule['num_microbatches']} "
        f"VPP_chunks={schedule['virtual_pipeline_chunks']} "
        f"VP_group={schedule['microbatch_group_size_per_vp_stage']}"
    )
    parameters = estimate["global_parameters"]
    print(f"global parameters: {_format_count(parameters['total'])}")
    active_suffix = ""
    if config["moe_layers"]:
        active_suffix = f" (TopK={config['topk']}/{config['experts']} experts)"
    print(
        "active parameters per token: "
        f"{_format_count(parameters['active_total'])}{active_suffix}"
    )
    print(
        f"peak stage: {peak['role']} rank={peak['pp_rank']} "
        f"local_parameters={_format_count(peak['local_parameters'])}"
    )
    if optimizer["enabled"]:
        print(
            "distributed optimizer: "
            f"shared={_format_count(peak['local_shared_parameters'])}/"
            f"{optimizer['shared_optimizer_shards']} "
            f"experts={_format_count(peak['local_expert_parameters'])}/"
            f"{optimizer['expert_optimizer_shards']} "
            f"expert-DP={optimizer['expert_data_parallel']}"
        )
    else:
        print(f"optimizer shards: {optimizer['shared_optimizer_shards']}")
    print(
        f"activation: local_layers={activation['local_layers']} "
        f"local_moe_layers={activation['local_moe_layers']} "
        f"peak_batches={peak['schedule']['activation_batches']:.2f} "
        f"peak_layer_equivalents="
        f"{peak['schedule']['max_resident_layer_equivalents']}"
    )
    if activation["local_moe_layers"]:
        print(
            "moe routing: "
            f"input_tokens={activation['router_tokens_per_rank']} "
            f"capacity_per_expert={activation['expert_capacity']} "
            f"local_expert_tokens={activation['routed_tokens_per_rank']}"
        )
    print("\nPP activation residency")
    for rank in schedule["ranks"]:
        print(
            f"  rank {rank['pp_rank']:2d}: "
            f"warmup={rank['warmup_virtual_microbatches']} "
            f"virtual={rank['max_outstanding_virtual_microbatches']} "
            f"batches={rank['activation_batches']:.2f} "
            f"by_chunk={rank['max_resident_by_chunk']}"
        )
    print("\nPer-layer saved activations")
    for name, value in activation["per_layer"].items():
        if name == "mtp" and not config["mtp_num_layers"]:
            continue
        print(f"  {name:26s} {_format_activation_memory(value)}")
    layer_type = "expert_mlp" if activation["local_moe_layers"] else "dense_mlp"
    print(f"\nModule tensors for one {layer_type} layer")
    for record in (
        activation["module_tensors"]["common"]
        + activation["module_tensors"][layer_type]
    ):
        support = "yes" if record["offload_supported"] else "no"
        label = f"{record['module']}.{record['tensor']}"
        print(
            f"  {label:40s} "
            f"{_format_activation_memory(record['bytes']):>10s} offload={support}"
        )
    if config["mtp_num_layers"]:
        print("\nModule tensors for one mtp layer")
        for record in activation["module_tensors"]["mtp"]:
            support = "yes" if record["offload_supported"] else "no"
            label = f"{record['module']}.{record['tensor']}"
            print(
                f"  {label:40s} "
                f"{_format_activation_memory(record['bytes']):>10s} offload={support}"
            )
    offload = activation["offload"]
    selection = offload["expert_layer" if layer_type == "expert_mlp" else "dense_layer"]
    print("\nFL offload estimate per activation group/layer")
    print(
        f"  modules={offload['requested_modules']} "
        f"budget={_format_activation_memory(offload['budget_per_layer'])} "
        f"captured={_format_activation_memory(selection['captured'])} "
        f"selected={_format_activation_memory(selection['selected'])}"
    )
    module_names = sorted(
        set(selection["captured_by_module"]) | set(selection["selected_by_module"])
    )
    for module in module_names:
        print(
            f"  {module:26s} "
            f"captured={_format_activation_memory(selection['captured_by_module'].get(module, 0))} "
            f"selected={_format_activation_memory(selection['selected_by_module'].get(module, 0))}"
        )
    if config["mtp_num_layers"]:
        mtp_selection = offload["final_decoder_and_mtp_group"]
        if mtp_selection["scheduled"]:
            print(
                "  final decoder + MTP group: scheduled=yes "
                f"captured={_format_activation_memory(mtp_selection['captured'])} "
                f"selected={_format_activation_memory(mtp_selection['selected'])}"
            )
        else:
            print(
                "  final decoder + MTP group: scheduled=no captured=0.00 MiB "
                "selected=0.00 MiB "
                f"(eligible_if_MTP_enabled="
                f"{_format_activation_memory(mtp_selection['candidate_captured'])}, "
                f"selectable={_format_activation_memory(mtp_selection['candidate_selected'])})"
            )
    print(
        f"  inferred H2D landing buffer: "
        f"{_format_activation_memory(offload['inferred_landing_buffer'])}"
    )
    print(
        f"  peak-stage eligible layer-equivalents: "
        f"{peak['offload_layer_equivalents']} "
        f"saved={_format_activation_memory(peak['offload_savings'])}"
    )
    print("\nPeak device memory")
    for name, value in peak["components"].items():
        print(f"  {name:34s} {_format_memory(value)}")
    print(f"  {'total':34s} {_format_memory(peak['total'])}")
    if estimate["headroom_bytes"] is not None:
        print(f"  {'device_headroom':34s} {_format_memory(estimate['headroom_bytes'])}")
    print("\nAssumptions")
    print("  - Values are analytical estimates, not allocator measurements.")
    if optimizer["enabled"]:
        print("  - Distributed optimizer shards shared state by DP and expert state by expert-DP.")
    else:
        print("  - Parameter, gradient, and optimizer shard factors are explicit inputs.")
    print("  - Attention output shared by attention core/projection is counted once.")
    print("  - Workspace, communication buffers, and allocator overhead are separate terms.")


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--layers", type=int, default=SCRIPT_CONFIG["layers"])
    parser.add_argument("--hidden-size", type=int, default=SCRIPT_CONFIG["hidden_size"])
    parser.add_argument(
        "--ffn-hidden-size", type=int, default=SCRIPT_CONFIG["ffn_hidden_size"]
    )
    parser.add_argument("--vocab-size", type=int, default=SCRIPT_CONFIG["vocab_size"])
    parser.add_argument(
        "--sequence-length", type=int, default=SCRIPT_CONFIG["sequence_length"]
    )
    parser.add_argument(
        "--micro-batch-size", type=int, default=SCRIPT_CONFIG["micro_batch_size"]
    )
    parser.add_argument(
        "--attention-heads", type=int, default=SCRIPT_CONFIG["attention_heads"]
    )
    parser.add_argument("--kv-heads", type=int, default=SCRIPT_CONFIG["kv_heads"])
    parser.add_argument(
        "--attention-backend",
        choices=("flash", "fused", "unfused"),
        default=SCRIPT_CONFIG["attention_backend"],
    )
    parser.add_argument(
        "--multi-latent-attention",
        action=argparse.BooleanOptionalAction,
        default=SCRIPT_CONFIG["multi_latent_attention"],
    )
    parser.add_argument("--q-lora-rank", type=int, default=SCRIPT_CONFIG["q_lora_rank"])
    parser.add_argument("--kv-lora-rank", type=int, default=SCRIPT_CONFIG["kv_lora_rank"])
    parser.add_argument("--qk-head-dim", type=int, default=SCRIPT_CONFIG["qk_head_dim"])
    parser.add_argument(
        "--qk-pos-emb-head-dim",
        type=int,
        default=SCRIPT_CONFIG["qk_pos_emb_head_dim"],
    )
    parser.add_argument("--v-head-dim", type=int, default=SCRIPT_CONFIG["v_head_dim"])
    parser.add_argument(
        "--qk-layernorm",
        action=argparse.BooleanOptionalAction,
        default=SCRIPT_CONFIG["qk_layernorm"],
    )
    parser.add_argument(
        "--mla-down-proj-fusion",
        action=argparse.BooleanOptionalAction,
        default=SCRIPT_CONFIG["mla_down_proj_fusion"],
    )
    parser.add_argument("--experts", type=int, default=SCRIPT_CONFIG["experts"])
    parser.add_argument("--topk", type=int, default=SCRIPT_CONFIG["topk"])
    parser.add_argument("--moe-layers", type=int, default=SCRIPT_CONFIG["moe_layers"])
    parser.add_argument(
        "--moe-ffn-hidden-size",
        type=int,
        default=SCRIPT_CONFIG["moe_ffn_hidden_size"],
    )
    parser.add_argument(
        "--moe-shared-expert-intermediate-size",
        type=int,
        default=SCRIPT_CONFIG["moe_shared_expert_intermediate_size"],
    )
    parser.add_argument(
        "--moe-shared-expert-gate",
        action=argparse.BooleanOptionalAction,
        default=SCRIPT_CONFIG["moe_shared_expert_gate"],
    )
    parser.add_argument(
        "--mtp-num-layers", type=int, default=SCRIPT_CONFIG["mtp_num_layers"]
    )
    parser.add_argument(
        "--mtp-layer-is-moe",
        action=argparse.BooleanOptionalAction,
        default=SCRIPT_CONFIG["mtp_layer_is_moe"],
    )
    parser.add_argument(
        "--capacity-factor", type=float, default=SCRIPT_CONFIG["capacity_factor"]
    )
    parser.add_argument(
        "--gated-mlp",
        action=argparse.BooleanOptionalAction,
        default=SCRIPT_CONFIG["gated_mlp"],
    )
    parser.add_argument(
        "--untied-embeddings",
        action=argparse.BooleanOptionalAction,
        default=SCRIPT_CONFIG["untied_embeddings"],
    )
    parser.add_argument("--world-size", type=int, default=SCRIPT_CONFIG["world_size"])
    parser.add_argument(
        "--global-batch-size", type=int, default=SCRIPT_CONFIG["global_batch_size"]
    )
    parser.add_argument(
        "--tensor-parallel", type=int, default=SCRIPT_CONFIG["tensor_parallel"]
    )
    parser.add_argument(
        "--pipeline-parallel", type=int, default=SCRIPT_CONFIG["pipeline_parallel"]
    )
    parser.add_argument(
        "--expert-parallel", type=int, default=SCRIPT_CONFIG["expert_parallel"]
    )
    parser.add_argument(
        "--expert-tensor-parallel",
        type=int,
        default=SCRIPT_CONFIG["expert_tensor_parallel"],
    )
    parser.add_argument(
        "--context-parallel", type=int, default=SCRIPT_CONFIG["context_parallel"]
    )
    parser.add_argument(
        "--sequence-parallel",
        action=argparse.BooleanOptionalAction,
        default=SCRIPT_CONFIG["sequence_parallel"],
    )
    parser.add_argument(
        "--use-distributed-optimizer",
        action=argparse.BooleanOptionalAction,
        default=SCRIPT_CONFIG["use_distributed_optimizer"],
    )
    parser.add_argument(
        "--pipeline-schedule",
        choices=("manual", "1f1b", "interleaved-1f1b", "gpipe"),
        default=SCRIPT_CONFIG["pipeline_schedule"],
    )
    parser.add_argument(
        "--layers-per-virtual-pipeline-stage",
        type=int,
        default=SCRIPT_CONFIG["layers_per_virtual_pipeline_stage"],
    )
    parser.add_argument(
        "--microbatch-group-size-per-vp-stage",
        type=int,
        default=SCRIPT_CONFIG["microbatch_group_size_per_vp_stage"],
    )
    parser.add_argument(
        "--overlap-moe-expert-parallel-comm",
        action=argparse.BooleanOptionalAction,
        default=SCRIPT_CONFIG["overlap_moe_expert_parallel_comm"],
    )
    parser.add_argument("--local-layers", type=int, default=SCRIPT_CONFIG["local_layers"])
    parser.add_argument(
        "--local-moe-layers", type=int, default=SCRIPT_CONFIG["local_moe_layers"]
    )
    parser.add_argument(
        "--inflight-microbatches",
        type=int,
        default=SCRIPT_CONFIG["inflight_microbatches"],
    )
    parser.add_argument(
        "--offload-modules",
        nargs="*",
        type=_normalize_offload_module,
        choices=OFFLOAD_MODULES,
        default=SCRIPT_CONFIG["offload_modules"],
    )
    for option, value_type in (
        ("parameter_bytes", float),
        ("gradient_bytes", float),
        ("master_parameter_bytes", float),
        ("optimizer_state_bytes", float),
        ("activation_bytes", float),
        ("lse_bytes", float),
        ("logit_bytes", float),
        ("parameter_shards", int),
        ("gradient_shards", int),
        ("optimizer_shards", int),
        ("expert_activation_shards", int),
        ("attention_score_buffers", float),
        ("unfused_attention_probability_buffers", float),
        ("offload_min_tensor_bytes", int),
        ("offload_mib_per_layer", float),
        ("offload_gpu_buffer_mib", float),
        ("communication_buffer_factor", float),
        ("communication_buffer_mib", float),
        ("workspace_mib", float),
        ("overhead_percent", float),
        ("device_memory_gib", float),
    ):
        parser.add_argument(
            f"--{option.replace('_', '-')}",
            type=value_type,
            default=SCRIPT_CONFIG[option],
        )
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser.parse_args()


def config_from_args(args):
    values = vars(args).copy()
    values.pop("as_json")
    cli_args = sys.argv[1:]

    def option_was_set(option):
        return any(arg == option or arg.startswith(f"{option}=") for arg in cli_args)

    if option_was_set("--attention-heads") and not option_was_set("--kv-heads"):
        values["kv_heads"] = args.attention_heads
    if option_was_set("--experts") and not option_was_set("--moe-layers"):
        values["moe_layers"] = args.layers if args.experts > 1 else 0
    elif (
        option_was_set("--layers")
        and not option_was_set("--moe-layers")
        and SCRIPT_CONFIG["moe_layers"] == SCRIPT_CONFIG["layers"]
    ):
        values["moe_layers"] = args.layers
    return MemoryModelConfig(**values)


def main():
    args = parse_args()
    estimate = estimate_memory(config_from_args(args))
    if args.as_json:
        print(json.dumps(estimate, indent=2, sort_keys=True))
    else:
        print_report(estimate)


if __name__ == "__main__":
    main()
