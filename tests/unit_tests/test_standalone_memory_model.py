import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.standalone_memory_model import (
    GIB,
    MIB,
    SCRIPT_CONFIG,
    MemoryModelConfig,
    estimate_memory,
    parameter_counts,
    print_report,
    schedule_residency,
)


def _balanced_moe_config(**overrides):
    values = {
        "layers": 8,
        "hidden_size": 4096,
        "ffn_hidden_size": 2048,
        "vocab_size": 8192,
        "sequence_length": 1024,
        "micro_batch_size": 1,
        "attention_heads": 64,
        "kv_heads": 8,
        "experts": 8,
        "topk": 2,
        "moe_layers": 8,
        "capacity_factor": 1.0,
        "gated_mlp": True,
        "tensor_parallel": 1,
        "pipeline_parallel": 2,
        "expert_parallel": 2,
        "optimizer_shards": 2,
        "inflight_microbatches": 1,
        "overhead_percent": 0,
    }
    values.update(overrides)
    return MemoryModelConfig(**values)


def test_balanced_moe_activation_matches_observed_scope_sizes():
    estimate = estimate_memory(_balanced_moe_config())
    per_layer = estimate["activation_model"]["per_layer"]

    assert per_layer["qkv_linear"] / MIB == pytest.approx(16.0)
    assert per_layer["core_attention"] / MIB == pytest.approx(18.25)
    assert per_layer["expert_mlp"] / MIB == pytest.approx(40.0)
    assert sum(
        per_layer[name] for name in ("qkv_linear", "core_attention", "expert_mlp")
    ) / MIB == pytest.approx(74.25)

    modules = estimate["activation_model"]["module_totals_per_layer"]
    assert modules["common"]["LayerNormLinear"] / MIB == pytest.approx(16.0)
    assert modules["common"]["Attention"] / MIB == pytest.approx(18.25)
    assert modules["expert_mlp"]["GroupedLinear"] / MIB == pytest.approx(24.0)
    assert modules["expert_mlp"]["swiglu"] / MIB == pytest.approx(16.0)


def test_active_parameters_use_only_topk_experts_per_moe_layer():
    config = _balanced_moe_config()
    counts = parameter_counts(config)
    expert_parameters = 3 * config.hidden_size * config.ffn_hidden_size

    assert counts["per_layer"]["active_expert_mlp"] == (
        config.topk * expert_parameters
    )
    assert counts["total"] - counts["active_total"] == (
        (config.experts - config.topk) * expert_parameters * config.moe_layers
    )
    assert counts["active_components"]["expert_mlp"] == (
        config.topk * expert_parameters * config.moe_layers
    )


def test_all_experts_active_matches_total_parameter_count():
    config = _balanced_moe_config(topk=8)
    counts = parameter_counts(config)

    assert counts["active_total"] == counts["total"]


def test_mtp_parameters_include_projection_and_one_active_expert_set_per_depth():
    config = _balanced_moe_config(mtp_num_layers=1)
    baseline = parameter_counts(_balanced_moe_config())
    counts = parameter_counts(config)
    routed_expert = 3 * config.hidden_size * config.ffn_hidden_size
    expected_shared = (
        baseline["per_layer"]["attention"]
        + baseline["per_layer"]["normalization"]
        + 2 * config.hidden_size**2
        + 3 * config.hidden_size
        + config.hidden_size * config.experts
    )

    assert counts["per_layer"]["mtp_shared"] == expected_shared
    assert counts["per_layer"]["mtp_expert"] == config.experts * routed_expert
    assert counts["per_layer"]["mtp_active_expert"] == config.topk * routed_expert
    assert counts["total"] - baseline["total"] == (
        expected_shared + config.experts * routed_expert
    )
    assert counts["active_total"] - baseline["active_total"] == (
        expected_shared + config.topk * routed_expert
    )


def test_mla_and_shared_expert_parameter_counts_use_distinct_widths():
    config = _balanced_moe_config(
        hidden_size=2048,
        ffn_hidden_size=11264,
        attention_heads=16,
        kv_heads=16,
        multi_latent_attention=True,
        kv_lora_rank=512,
        qk_head_dim=128,
        qk_pos_emb_head_dim=64,
        v_head_dim=128,
        experts=64,
        topk=2,
        moe_ffn_hidden_size=1408,
        moe_shared_expert_intermediate_size=2816,
    )
    counts = parameter_counts(config)
    q_projection = 2048 * (16 * (128 + 64))
    kv_projection = 2048 * (512 + 64) + 512 * (16 * (128 + 128))
    output_projection = 16 * 128 * 2048
    routed_expert = 3 * 2048 * 1408
    shared_expert = 3 * 2048 * 2816

    assert counts["per_layer"]["attention"] == (
        q_projection + kv_projection + output_projection
    )
    assert counts["per_layer"]["expert_mlp"] == 64 * routed_expert
    assert counts["per_layer"]["routed_expert_mlp"] == 64 * routed_expert
    assert counts["per_layer"]["active_expert_mlp"] == 2 * routed_expert
    assert counts["per_layer"]["shared_expert_mlp"] == shared_expert
    assert counts["active_components"]["shared_expert_mlp"] == (
        shared_expert * config.moe_layers
    )

    peak = estimate_memory(config)["peak_stage"]
    expected_routed = (
        config.layers
        // config.pipeline_parallel
        * (64 * routed_expert)
        / (config.expert_parallel * config.expert_tensor_parallel)
    )
    assert peak["local_expert_parameters"] == pytest.approx(expected_routed)


def test_mla_and_shared_expert_saved_activation_breakdown():
    estimate = estimate_memory(
        _balanced_moe_config(
            hidden_size=2048,
            ffn_hidden_size=11264,
            sequence_length=1024,
            attention_heads=16,
            kv_heads=16,
            tensor_parallel=2,
            sequence_parallel=True,
            experts=64,
            expert_parallel=4,
            topk=2,
            multi_latent_attention=True,
            kv_lora_rank=512,
            qk_head_dim=128,
            qk_pos_emb_head_dim=64,
            v_head_dim=128,
            moe_ffn_hidden_size=1408,
            moe_shared_expert_intermediate_size=2816,
            offload_modules=("MLA", "Attention", "SharedExpert"),
            offload_mib_per_layer=22.78125,
            offload_min_tensor_bytes=0,
        )
    )
    activation = estimate["activation_model"]
    modules = activation["module_totals_per_layer"]

    assert modules["common"]["MLA"] / MIB == pytest.approx(2.5)
    assert modules["common"]["Attention"] / MIB == pytest.approx(10.03125)
    assert modules["expert_mlp"]["SharedExpert"] / MIB == pytest.approx(10.25)
    assert activation["per_layer"]["routed_expert_mlp"] / MIB == pytest.approx(12.25)
    assert activation["per_layer"]["shared_expert_mlp"] / MIB == pytest.approx(10.25)
    selection = activation["offload"]["expert_layer"]
    assert selection["captured"] / MIB == pytest.approx(22.78125)
    assert selection["selected"] / MIB == pytest.approx(22.78125)


def test_unfused_mla_models_quadratic_attention_probability():
    estimate = estimate_memory(
        _balanced_moe_config(
            hidden_size=2048,
            ffn_hidden_size=11264,
            sequence_length=1024,
            attention_heads=16,
            kv_heads=16,
            tensor_parallel=2,
            sequence_parallel=True,
            experts=64,
            expert_parallel=4,
            topk=2,
            attention_backend="unfused",
            multi_latent_attention=True,
            kv_lora_rank=512,
            qk_head_dim=128,
            qk_pos_emb_head_dim=64,
            v_head_dim=128,
            moe_ffn_hidden_size=1408,
            moe_shared_expert_intermediate_size=2816,
            offload_modules=("MLA", "Attention", "SharedExpert"),
            unfused_attention_probability_buffers=1.0,
            offload_mib_per_layer=38.75,
            offload_min_tensor_bytes=0,
        )
    )
    activation = estimate["activation_model"]
    modules = activation["module_totals_per_layer"]

    assert modules["common"]["MLA"] / MIB == pytest.approx(2.5)
    assert modules["common"]["Attention"] / MIB == pytest.approx(26.0)
    assert modules["expert_mlp"]["SharedExpert"] / MIB == pytest.approx(10.25)
    selection = activation["offload"]["expert_layer"]
    assert selection["captured"] / MIB == pytest.approx(38.75)
    assert selection["selected"] / MIB == pytest.approx(38.75)


def test_sequence_parallel_tokens_drive_padded_expert_capacity():
    estimate = estimate_memory(
        _balanced_moe_config(
            tensor_parallel=2,
            sequence_parallel=True,
            experts=16,
            expert_parallel=4,
            topk=6,
        )
    )
    activation = estimate["activation_model"]

    assert activation["tokens_per_rank"] == 1024
    assert activation["router_tokens_per_rank"] == 512
    assert activation["expert_capacity"] == 192
    assert activation["routed_tokens_per_rank"] == 3072


def test_mtp_models_2h_projection_input_and_reuses_inner_layer_scopes():
    estimate = estimate_memory(
        _balanced_moe_config(
            mtp_num_layers=1,
            offload_modules=("MTP",),
            offload_mib_per_layer=16,
        )
    )
    activation = estimate["activation_model"]
    records = activation["module_tensors"]["mtp"]
    eh_proj = next(
        record
        for record in records
        if record["module"] == "MTP" and record["tensor"] == "eh_proj_input"
    )

    assert eh_proj["bytes"] / MIB == pytest.approx(16)
    assert eh_proj["offload_supported"] is True
    modules = {record["module"] for record in records}
    assert {"MTP", "LayerNormLinear", "Attention", "GroupedLinear", "swiglu"} <= modules
    mtp_layer = activation["offload"]["mtp_layer"]
    assert mtp_layer["captured"] / MIB == pytest.approx(16)
    assert mtp_layer["selected"] / MIB == pytest.approx(16)


def test_mtp_final_chunk_uses_one_shared_offload_budget():
    estimate = estimate_memory(
        _balanced_moe_config(
            layers=4,
            moe_layers=4,
            pipeline_parallel=2,
            world_size=4,
            global_batch_size=8,
            pipeline_schedule="interleaved-1f1b",
            layers_per_virtual_pipeline_stage=1,
            inflight_microbatches=0,
            mtp_num_layers=1,
            offload_modules=("MTP",),
            offload_mib_per_layer=16,
        )
    )
    last = next(
        candidate
        for candidate in estimate["stage_candidates"]
        if candidate["role"] == "last"
    )
    final_chunk_residency = last["schedule"]["max_resident_by_chunk"][-1]

    assert last["offload_savings"] / MIB == pytest.approx(
        16 * final_chunk_residency
    )
    assert last["components"]["offload_gpu_buffer"] / MIB == pytest.approx(16)


def test_mtp_final_chunk_is_not_selected_without_mtp_schedule_module():
    estimate = estimate_memory(
        _balanced_moe_config(
            layers=4,
            moe_layers=4,
            pipeline_parallel=2,
            world_size=4,
            global_batch_size=8,
            pipeline_schedule="interleaved-1f1b",
            layers_per_virtual_pipeline_stage=1,
            inflight_microbatches=0,
            mtp_num_layers=1,
            offload_modules=("Attention",),
            offload_mib_per_layer=20,
        )
    )
    activation = estimate["activation_model"]
    final_group = activation["offload"]["final_decoder_and_mtp_group"]

    assert final_group["scheduled"] is False
    assert final_group["captured"] == 0
    assert final_group["selected"] == 0
    assert final_group["candidate_captured"] > 20 * MIB
    assert final_group["candidate_selected"] == 20 * MIB
    assert activation["offload"]["inferred_landing_buffer"] == 0


def test_offload_reduces_each_local_layer_activation_but_adds_landing_buffer():
    baseline = estimate_memory(_balanced_moe_config(overhead_percent=10))
    offloaded = estimate_memory(
        _balanced_moe_config(
            offload_modules=("LayerNormLinear", "GroupedLinear", "swiglu"),
            offload_mib_per_layer=56,
            overhead_percent=10,
        )
    )

    local_layers = baseline["activation_model"]["local_layers"]
    assert local_layers == 4
    assert (
        baseline["peak_stage"]["components"]["saved_activations"]
        - offloaded["peak_stage"]["components"]["saved_activations"]
    ) / MIB == pytest.approx(56 * local_layers)
    assert offloaded["peak_stage"]["components"]["offload_gpu_buffer"] / MIB == 56
    assert (
        offloaded["peak_stage"]["components"]["allocator_and_unmodeled_overhead"]
        == baseline["peak_stage"]["components"]["allocator_and_unmodeled_overhead"]
    )


def test_offload_module_filter_and_budget_match_runtime_selection():
    swiglu = estimate_memory(
        _balanced_moe_config(
            offload_modules=("swiglu",),
            offload_mib_per_layer=16,
        )
    )
    selection = swiglu["activation_model"]["offload"]["expert_layer"]

    assert selection["captured"] / MIB == pytest.approx(16)
    assert selection["selected"] / MIB == pytest.approx(16)
    assert selection["captured_by_module"]["swiglu"] / MIB == pytest.approx(16)
    assert selection["selected_by_module"]["swiglu"] / MIB == pytest.approx(16)

    over_budget = estimate_memory(
        _balanced_moe_config(
            offload_modules=("swiglu",),
            offload_mib_per_layer=17,
        )
    )
    over_selection = over_budget["activation_model"]["offload"]["expert_layer"]
    assert over_selection["captured"] / MIB == pytest.approx(16)
    assert over_selection["selected"] == 0
    assert over_budget["peak_stage"]["components"]["offload_gpu_buffer"] == 0


def test_flash_attention_offload_covers_modeled_qkvo_and_lse():
    estimate = estimate_memory(
        _balanced_moe_config(
            offload_modules=("Attention",),
            offload_min_tensor_bytes=0,
            offload_mib_per_layer=18.25,
        )
    )
    selection = estimate["activation_model"]["offload"]["expert_layer"]

    assert selection["captured"] / MIB == pytest.approx(18.25)
    assert selection["selected"] / MIB == pytest.approx(18.25)
    assert selection["captured_by_module"]["Attention"] / MIB == pytest.approx(
        18.25
    )


def test_fused_attention_offload_covers_modeled_qkvo_and_softmax_stats():
    estimate = estimate_memory(
        _balanced_moe_config(
            attention_backend="fused",
            offload_modules=("Attention",),
            offload_min_tensor_bytes=0,
            offload_mib_per_layer=18.25,
        )
    )
    activation = estimate["activation_model"]
    records = [
        record
        for record in activation["module_tensors"]["common"]
        if record["module"] == "Attention"
    ]
    sizes = {record["tensor"]: record["bytes"] / MIB for record in records}

    assert sizes == pytest.approx(
        {"q": 8.0, "k": 1.0, "v": 1.0, "output": 8.0, "softmax_stats": 0.25}
    )
    selection = activation["offload"]["expert_layer"]
    assert selection["captured"] / MIB == pytest.approx(18.25)
    assert selection["selected"] / MIB == pytest.approx(18.25)
    assert selection["captured_by_module"]["Attention"] / MIB == pytest.approx(
        18.25
    )


def test_fused_mla_models_unequal_qk_and_value_widths_without_probabilities():
    estimate = estimate_memory(
        _balanced_moe_config(
            hidden_size=2048,
            ffn_hidden_size=11264,
            sequence_length=1024,
            attention_heads=16,
            kv_heads=16,
            tensor_parallel=2,
            sequence_parallel=True,
            experts=64,
            expert_parallel=4,
            topk=2,
            attention_backend="fused",
            multi_latent_attention=True,
            kv_lora_rank=512,
            qk_head_dim=128,
            qk_pos_emb_head_dim=64,
            v_head_dim=128,
            moe_ffn_hidden_size=1408,
            offload_modules=("Attention",),
            offload_min_tensor_bytes=0,
            offload_mib_per_layer=10.03125,
        )
    )
    records = [
        record
        for record in estimate["activation_model"]["module_tensors"]["common"]
        if record["module"] == "Attention"
    ]
    sizes = {record["tensor"]: record["bytes"] / MIB for record in records}

    assert sizes == pytest.approx(
        {"q": 3.0, "k": 3.0, "v": 2.0, "output": 2.0, "softmax_stats": 0.03125}
    )
    assert "attention_probs" not in sizes


def test_optimizer_shards_only_master_parameters_and_optimizer_states():
    unsharded = estimate_memory(_balanced_moe_config(optimizer_shards=1))
    sharded = estimate_memory(_balanced_moe_config(optimizer_shards=2))
    left = unsharded["peak_stage"]["components"]
    right = sharded["peak_stage"]["components"]

    assert right["model_parameters"] == left["model_parameters"]
    assert right["gradients"] == left["gradients"]
    assert right["master_parameters"] == left["master_parameters"] / 2
    assert right["optimizer_states"] == left["optimizer_states"] / 2


def test_distributed_optimizer_uses_dp_for_shared_and_expert_dp_for_experts():
    estimate = estimate_memory(
        _balanced_moe_config(
            world_size=8,
            use_distributed_optimizer=True,
            optimizer_shards=1,
        )
    )
    optimizer = estimate["optimizer_model"]
    peak = estimate["peak_stage"]
    shared = peak["local_shared_parameters"]
    experts = peak["local_expert_parameters"]

    assert optimizer["data_parallel"] == 4
    assert optimizer["expert_data_parallel"] == 2
    assert optimizer["shared_optimizer_shards"] == 4
    assert optimizer["expert_optimizer_shards"] == 2
    sharded_parameters = shared / 4 + experts / 2
    assert peak["components"]["master_parameters"] == pytest.approx(
        sharded_parameters * 4
    )
    assert peak["components"]["optimizer_states"] == pytest.approx(
        sharded_parameters * 8
    )


def test_expert_parallel_grid_is_independent_of_normal_data_parallel():
    estimate = estimate_memory(
        _balanced_moe_config(
            world_size=8,
            tensor_parallel=2,
            pipeline_parallel=2,
            expert_parallel=4,
            expert_tensor_parallel=1,
            global_batch_size=8,
            pipeline_schedule="1f1b",
            inflight_microbatches=0,
            use_distributed_optimizer=True,
        )
    )

    assert estimate["optimizer_model"]["data_parallel"] == 2
    assert estimate["optimizer_model"]["expert_data_parallel"] == 1
    assert estimate["schedule_model"]["data_parallel"] == 2


def test_invalid_expert_parallel_grid_is_rejected():
    with pytest.raises(ValueError, match=r"divisible by ETP \* EP \* PP"):
        estimate_memory(
            _balanced_moe_config(
                experts=12,
                world_size=8,
                pipeline_parallel=2,
                expert_parallel=3,
                expert_tensor_parallel=1,
                use_distributed_optimizer=True,
            )
        )


def test_default_mtp_capacity_plan_matches_current_memory_baseline():
    estimate = estimate_memory(MemoryModelConfig(**SCRIPT_CONFIG))

    assert estimate["optimizer_model"]["data_parallel"] == 8
    assert estimate["optimizer_model"]["expert_data_parallel"] == 2
    assert estimate["peak_stage"]["components"]["saved_activations"] / GIB == pytest.approx(
        12.126, abs=0.01
    )
    assert estimate["peak_stage"]["total"] / GIB == pytest.approx(62.181, abs=0.05)

    first_stage = next(
        stage for stage in estimate["stage_candidates"] if stage["role"] == "first"
    )
    allocated_like = (
        first_stage["total"]
        - first_stage["components"]["allocator_and_unmodeled_overhead"]
    )
    assert allocated_like / GIB == pytest.approx(36.411, abs=0.05)


def test_last_pipeline_rank_excludes_final_virtual_chunk_from_offload():
    estimate = estimate_memory(
        _balanced_moe_config(
            layers=4,
            moe_layers=4,
            pipeline_parallel=2,
            world_size=4,
            global_batch_size=8,
            pipeline_schedule="interleaved-1f1b",
            layers_per_virtual_pipeline_stage=1,
            inflight_microbatches=0,
            offload_modules=("swiglu",),
            offload_mib_per_layer=16,
        )
    )
    last = next(
        candidate
        for candidate in estimate["stage_candidates"]
        if candidate["role"] == "last"
    )

    assert last["schedule"]["max_resident_layer_equivalents"] == 3
    assert last["schedule"]["max_resident_by_chunk"] == [2, 1]
    assert last["offload_layer_equivalents"] == 2
    assert last["offload_savings"] / MIB == pytest.approx(32)
    assert last["components"]["offload_gpu_buffer"] / MIB == pytest.approx(16)


def test_cli_emits_machine_readable_json():
    script = Path(__file__).parents[2] / "tools/standalone_memory_model.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--layers",
            "2",
            "--hidden-size",
            "128",
            "--ffn-hidden-size",
            "256",
            "--vocab-size",
            "1024",
            "--sequence-length",
            "32",
            "--attention-heads",
            "4",
            "--pipeline-schedule",
            "manual",
            "--inflight-microbatches",
            "1",
            "--json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["peak_stage"]["total"] > 0
    assert payload["global_parameters"]["total"] > 0
    assert payload["global_parameters"]["active_total"] > 0


def test_cli_can_run_with_editable_script_defaults():
    script = Path(__file__).parents[2] / "tools/standalone_memory_model.py"
    result = subprocess.run(
        [sys.executable, str(script), "--json"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["config"]["layers"] == SCRIPT_CONFIG["layers"]
    assert payload["config"]["workspace_mib"] == SCRIPT_CONFIG["workspace_mib"]
    assert payload["peak_stage"]["total"] > 0


def test_human_readable_memory_report_uses_mib_consistently(capsys):
    estimate = estimate_memory(_balanced_moe_config(mtp_num_layers=1))

    print_report(estimate)
    report = capsys.readouterr().out

    per_layer = report.split("Per-layer saved activations", 1)[1].split(
        "Module tensors", 1
    )[0]
    module_tensors = report.split("Module tensors for one expert_mlp layer", 1)[1].split(
        "FL offload estimate", 1
    )[0]
    offload = report.split("FL offload estimate per activation group/layer", 1)[1].split(
        "Peak device memory", 1
    )[0]
    peak = report.split("Peak device memory", 1)[1]

    assert " MiB" in per_layer and "GiB" not in per_layer
    assert " MiB" in module_tensors and "GiB" not in module_tensors
    assert " MiB" in offload and "GiB" not in offload
    assert " MiB" in peak and "GiB" not in peak


def test_non_interleaved_1f1b_infers_rank_specific_residency():
    config = _balanced_moe_config(
        pipeline_parallel=4,
        world_size=8,
        global_batch_size=16,
        pipeline_schedule="1f1b",
        inflight_microbatches=0,
    )
    schedule = schedule_residency(config, local_layers=2)

    assert schedule["data_parallel"] == 2
    assert schedule["num_microbatches"] == 8
    assert [rank["activation_batches"] for rank in schedule["ranks"]] == [4, 3, 2, 1]


def test_non_pipeline_1f1b_keeps_one_activation_batch():
    config = _balanced_moe_config(
        pipeline_parallel=1,
        world_size=2,
        global_batch_size=8,
        pipeline_schedule="1f1b",
        inflight_microbatches=0,
    )
    schedule = schedule_residency(config, local_layers=8)

    assert schedule["num_microbatches"] == 4
    assert schedule["ranks"][0]["activation_batches"] == 1


def test_gpipe_keeps_all_microbatch_activations():
    config = _balanced_moe_config(
        pipeline_parallel=2,
        world_size=4,
        global_batch_size=8,
        pipeline_schedule="gpipe",
        inflight_microbatches=0,
    )
    schedule = schedule_residency(config, local_layers=4)

    assert schedule["num_microbatches"] == 4
    assert [rank["activation_batches"] for rank in schedule["ranks"]] == [4, 4]


def test_interleaved_1f1b_infers_virtual_chunk_residency():
    config = _balanced_moe_config(
        layers=4,
        moe_layers=4,
        pipeline_parallel=2,
        world_size=4,
        global_batch_size=8,
        pipeline_schedule="interleaved-1f1b",
        layers_per_virtual_pipeline_stage=1,
        inflight_microbatches=0,
    )
    schedule = schedule_residency(config, local_layers=2)

    assert schedule["num_microbatches"] == 4
    assert schedule["virtual_pipeline_chunks"] == 2
    assert [rank["warmup_virtual_microbatches"] for rank in schedule["ranks"]] == [4, 2]
    assert [rank["max_outstanding_virtual_microbatches"] for rank in schedule["ranks"]] == [5, 3]
    assert [rank["activation_batches"] for rank in schedule["ranks"]] == [2.5, 1.5]


def test_combined_interleaved_1f1b_adds_one_warmup_forward():
    config = _balanced_moe_config(
        layers=4,
        moe_layers=4,
        pipeline_parallel=2,
        world_size=4,
        global_batch_size=8,
        pipeline_schedule="interleaved-1f1b",
        layers_per_virtual_pipeline_stage=1,
        overlap_moe_expert_parallel_comm=True,
        inflight_microbatches=0,
    )
    schedule = schedule_residency(config, local_layers=2)

    assert [rank["warmup_virtual_microbatches"] for rank in schedule["ranks"]] == [5, 3]
    assert [rank["max_outstanding_virtual_microbatches"] for rank in schedule["ranks"]] == [6, 4]
    assert [rank["activation_batches"] for rank in schedule["ranks"]] == [3, 2]


def test_invalid_parallel_shape_is_rejected():
    with pytest.raises(ValueError, match="kv_heads must be divisible"):
        estimate_memory(_balanced_moe_config(tensor_parallel=16))
