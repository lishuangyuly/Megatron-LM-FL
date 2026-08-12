"""CPU checks for narrow MLA and shared-expert FL capture scopes."""

from types import SimpleNamespace

import torch

from megatron.plugin.fl_offload import offload


def _runtime_args(modules):
    return SimpleNamespace(
        fl_patch_te=True,
        fl_offload_modules=modules,
        fl_min_offloaded_tensor_size=0,
        fl_per_batch_offload_size=0,
        fl_activation_offload_stages=1,
        fl_activation_offload_stages_assignment=[0],
        fl_use_comm_stream=False,
    )


def test_mla_scope_labels_linear_input(monkeypatch):
    monkeypatch.setattr(offload, "_args", lambda: _runtime_args(["MLA"]))
    offload.reset_for_tests()
    key = (0, 0, 0)
    tensor = torch.randn(8, 16)

    with offload.record(key, group_num=1):
        with offload.tensor_scope("MLA", "kv_up_input"):
            packed = offload.maybe_pack_linear_input(tensor)

    group = offload._GROUPS[key]
    assert packed.get() is tensor
    assert len(group.tensors) == 1
    assert group.tensors[0].op_name == "MLA"
    assert group.tensors[0].tensor_name == "kv_up_input"
    offload.reset_for_tests()


def test_shared_expert_scope_is_inactive_unless_selected(monkeypatch):
    monkeypatch.setattr(offload, "_args", lambda: _runtime_args(["MLA"]))
    offload.reset_for_tests()
    key = (0, 0, 1)
    tensor = torch.randn(8, 16)

    with offload.record(key, group_num=1):
        with offload.tensor_scope("SharedExpert", "fc1_input"):
            packed = offload.maybe_pack_scoped_tensor(tensor)

    assert packed is None
    assert offload._GROUPS[key].tensors == []
    offload.reset_for_tests()


def test_shared_expert_scope_labels_activation(monkeypatch):
    monkeypatch.setattr(offload, "_args", lambda: _runtime_args(["SharedExpert"]))
    offload.reset_for_tests()
    key = (0, 0, 2)
    tensor = torch.randn(8, 16)

    with offload.record(key, group_num=1):
        with offload.tensor_scope("SharedExpert", "swiglu_input"):
            packed = offload.maybe_pack_scoped_tensor(tensor)

    group = offload._GROUPS[key]
    assert packed.get() is tensor
    assert len(group.tensors) == 1
    assert group.tensors[0].op_name == "SharedExpert"
    assert group.tensors[0].tensor_name == "swiglu_input"
    offload.reset_for_tests()


def test_mtp_scope_labels_eh_projection_input(monkeypatch):
    monkeypatch.setattr(offload, "_args", lambda: _runtime_args(["MTP"]))
    offload.reset_for_tests()
    key = (0, 0, 6)
    tensor = torch.randn(8, 32)

    with offload.record(key, group_num=1):
        with offload.tensor_scope("MTP", "eh_proj_input"):
            packed = offload.maybe_pack_linear_input(tensor)

    group = offload._GROUPS[key]
    assert packed.get() is tensor
    assert len(group.tensors) == 1
    assert group.tensors[0].op_name == "MTP"
    assert group.tensors[0].tensor_name == "eh_proj_input"
    offload.reset_for_tests()


def test_mtp_offload_enables_final_chunk_only_when_requested(monkeypatch):
    config = SimpleNamespace(mtp_num_layers=1)
    monkeypatch.setattr(offload, "_args", lambda: _runtime_args(["MTP"]))

    assert offload.mtp_offload_enabled(config) is True

    monkeypatch.setattr(offload, "_args", lambda: _runtime_args(["GroupedLinear"]))
    assert offload.mtp_offload_enabled(config) is False
    assert offload.mtp_offload_enabled(SimpleNamespace(mtp_num_layers=0)) is False


def test_group_is_reloadable_only_after_d2h_completion(monkeypatch):
    monkeypatch.setattr(offload, "_args", lambda: _runtime_args(["MTP"]))
    offload.reset_for_tests()
    key = (0, 1, 0)
    tensor = torch.randn(8, 32)

    with offload.record(key, group_num=1):
        with offload.tensor_scope("MTP", "eh_proj_input"):
            offload.maybe_pack_linear_input(tensor)

    assert offload.group_available_for_reload(key) is False
    offload._GROUPS[key].state = "offloaded"
    assert offload.group_available_for_reload(key) is True
    offload.reset_for_tests()


def test_capture_summary_reports_decoder_and_mtp_compositions(
    monkeypatch, capsys
):
    monkeypatch.setattr(
        offload,
        "_args",
        lambda: _runtime_args(["GroupedLinear", "MTP"]),
    )
    monkeypatch.setattr(
        offload,
        "get_cpu_buffer",
        lambda _size: torch.empty(0, dtype=torch.uint8),
    )
    offload.reset_for_tests()

    decoder_group = offload.ActivationGroup(
        [offload.TensorWrap(torch.ones(8), op_name="GroupedLinear")],
        key=(0, 0, 7),
        group_num=1,
    )
    decoder_group.offload_prologue()
    mtp_group = offload.ActivationGroup(
        [
            offload.TensorWrap(torch.ones(8), op_name="GroupedLinear"),
            offload.TensorWrap(
                torch.ones(8), op_name="MTP", tensor_name="eh_proj_input"
            ),
        ],
        key=(0, 1, 7),
        group_num=1,
    )
    mtp_group.offload_prologue()
    duplicate_mtp_group = offload.ActivationGroup(
        [
            offload.TensorWrap(torch.ones(8), op_name="GroupedLinear"),
            offload.TensorWrap(
                torch.ones(8), op_name="MTP", tensor_name="eh_proj_input"
            ),
        ],
        key=(1, 1, 7),
        group_num=1,
    )
    duplicate_mtp_group.offload_prologue()

    summaries = [
        line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("[FL offload] captured=")
    ]
    assert len(summaries) == 2
    assert "MTP(" not in summaries[0]
    assert "MTP(" in summaries[1]
    offload.reset_for_tests()


def test_unfused_attention_scope_captures_actual_backward_inputs(monkeypatch):
    monkeypatch.setattr(
        offload, "_args", lambda: _runtime_args(["UnfusedAttention"])
    )
    offload.reset_for_tests()
    key = (0, 0, 3)
    query = torch.randn(2, 4, 8, requires_grad=True)
    key_tensor = torch.randn(2, 4, 8, requires_grad=True)
    value = torch.randn(2, 4, 6, requires_grad=True)

    with offload.record(key, group_num=1):
        with offload.unfused_attention_saved_tensors(query, key_tensor, value):
            scores = torch.bmm(query, key_tensor.transpose(1, 2))
            probabilities = torch.softmax(scores, dim=-1)
            output = torch.bmm(probabilities, value)
        offload.maybe_pack_unfused_attention_output(output)
        output_pack = offload.maybe_pack_attention_projection(output)
        loss = output.square().mean()
    loss.backward()

    names = {tensor.tensor_name for tensor in offload._GROUPS[key].tensors}
    assert {"q", "k", "v", "attention_probs", "output"} <= names
    assert output_pack.get() is output
    assert query.grad is not None
    assert key_tensor.grad is not None
    assert value.grad is not None
    offload.reset_for_tests()


def test_unfused_attention_output_requires_projection_confirmation(monkeypatch):
    monkeypatch.setattr(
        offload, "_args", lambda: _runtime_args(["UnfusedAttention"])
    )
    offload.reset_for_tests()
    key = (0, 0, 5)
    output = torch.randn(2, 4, 8, requires_grad=True)

    with offload.record(key, group_num=1):
        offload.maybe_pack_unfused_attention_output(output)

    assert offload._GROUPS[key].tensors == []
    assert output.untyped_storage().nbytes() > 0
    offload.reset_for_tests()


def test_unfused_te_forward_wrapper_installs_narrow_capture(monkeypatch):
    from megatron.plugin.fl_offload import te_patch

    class FakeUnfusedAttention:
        def forward(self, _alibi_cache, query_layer, key_layer, value_layer):
            probabilities = torch.softmax(
                torch.bmm(query_layer, key_layer.transpose(1, 2)), dim=-1
            )
            return torch.bmm(probabilities, value_layer)

    monkeypatch.setattr(
        offload, "_args", lambda: _runtime_args(["UnfusedAttention"])
    )
    offload.reset_for_tests()
    te_patch._patch_unfused_attention(FakeUnfusedAttention)
    key = (0, 0, 4)
    query = torch.randn(2, 4, 8, requires_grad=True)
    key_tensor = torch.randn(2, 4, 8, requires_grad=True)
    value = torch.randn(2, 4, 6, requires_grad=True)

    try:
        with offload.record(key, group_num=1):
            output = FakeUnfusedAttention().forward(
                {}, query, key_tensor, value
            )
            output_pack = offload.maybe_pack_attention_projection(output)
            loss = output.square().mean()
        loss.backward()

        names = {tensor.tensor_name for tensor in offload._GROUPS[key].tensors}
        assert {"q", "k", "v", "attention_probs", "output"} <= names
        assert output_pack.get() is output
    finally:
        te_patch.restore_te_patches()
        offload.reset_for_tests()
