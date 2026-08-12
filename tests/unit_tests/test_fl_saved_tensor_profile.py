from types import SimpleNamespace

import pytest
import torch

from megatron.plugin.fl_offload import saved_tensor_profile


class _SaveTensors(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *tensors):
        ctx.save_for_backward(*tensors)
        return sum(tensor.sum() for tensor in tensors)

    @staticmethod
    def backward(ctx, grad_output):
        return tuple(torch.ones_like(tensor) * grad_output for tensor in ctx.saved_tensors)


class _SaveParameter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, parameter):
        ctx.save_for_backward(parameter)
        return parameter.square().sum()

    @staticmethod
    def backward(ctx, grad_output):
        (parameter,) = ctx.saved_tensors
        return 2 * parameter * grad_output


def _args(**overrides):
    values = {
        "fl_saved_tensor_profile": True,
        "fl_saved_tensor_profile_scopes": [],
        "fl_saved_tensor_profile_max_reports": 1,
        "gradient_accumulation_fusion": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    saved_tensor_profile.reset_for_tests()
    monkeypatch.setattr(saved_tensor_profile, "_args", lambda: _args())
    yield
    saved_tensor_profile.reset_for_tests()


def test_scope_reports_logical_and_unique_storage_bytes():
    base = torch.arange(8.0, requires_grad=True)
    first = base[:4]
    second = base[4:]

    with saved_tensor_profile.saved_tensor_scope("core_attn"):
        loss = _SaveTensors.apply(first, second)
    loss.backward()

    report = saved_tensor_profile._REPORTS[0]
    assert report["scope"] == "core_attn"
    assert report["saved_tensors"] == 2
    assert report["logical_bytes"] == base.numel() * base.element_size()
    assert report["unique_storage_bytes"] == base.untyped_storage().nbytes()
    assert report["unique_activation_storage_bytes"] == base.untyped_storage().nbytes()
    assert report["tensors"][0]["storage_ptr"] == report["tensors"][1]["storage_ptr"]


def test_scope_preserves_gradients_when_parameters_are_saved():
    parameter = torch.nn.Parameter(torch.arange(4.0))

    with saved_tensor_profile.saved_tensor_scope("qkv_linear"):
        loss = _SaveParameter.apply(parameter)
    loss.backward()

    torch.testing.assert_close(parameter.grad, 2 * parameter.detach())
    assert saved_tensor_profile._REPORTS[0]["parameter_logical_bytes"] == (
        parameter.numel() * parameter.element_size()
    )


def test_fused_gradient_accumulation_is_rejected(monkeypatch):
    monkeypatch.setattr(
        saved_tensor_profile,
        "_args",
        lambda: _args(gradient_accumulation_fusion=True),
    )

    with pytest.raises(RuntimeError, match="no-gradient-accumulation-fusion"):
        with saved_tensor_profile.saved_tensor_scope("qkv_linear"):
            pass


def test_explicit_pack_is_attributed_to_active_scope():
    tensor = torch.ones(4, requires_grad=True)

    with saved_tensor_profile.saved_tensor_scope("moe_act"):
        saved_tensor_profile.record_explicit_tensor(tensor)

    report = saved_tensor_profile._REPORTS[0]
    assert report["explicit_saved"] == 1
    assert report["autograd_saved"] == 0
    assert report["tensors"][0]["source"] == "explicit"


def test_shared_storage_is_visible_across_scopes():
    tensor = torch.ones(4, requires_grad=True)

    with saved_tensor_profile.saved_tensor_scope("core_attn"):
        _SaveTensors.apply(tensor)
    with saved_tensor_profile.saved_tensor_scope("attn_proj"):
        _SaveTensors.apply(tensor)

    projection = saved_tensor_profile._REPORTS[1]
    assert projection["cross_scope_storage_tensors"] == 1
    assert projection["tensors"][0]["shared_with_scopes"] == ["core_attn"]


def test_scope_filter_and_report_limit(monkeypatch):
    monkeypatch.setattr(
        saved_tensor_profile,
        "_args",
        lambda: _args(
            fl_saved_tensor_profile_scopes=["expert_fc1"],
            fl_saved_tensor_profile_max_reports=1,
        ),
    )
    tensor = torch.ones(4, requires_grad=True)

    with saved_tensor_profile.saved_tensor_scope("core_attn"):
        _SaveTensors.apply(tensor)
    for _ in range(2):
        with saved_tensor_profile.saved_tensor_scope("expert_fc1"):
            _SaveTensors.apply(tensor)

    assert len(saved_tensor_profile._REPORTS) == 1
    assert saved_tensor_profile._REPORTS[0]["scope"] == "expert_fc1"


def test_unknown_scope_is_rejected(monkeypatch):
    monkeypatch.setattr(
        saved_tensor_profile,
        "_args",
        lambda: _args(fl_saved_tensor_profile_scopes=["not_a_scope"]),
    )

    with pytest.raises(ValueError, match="unsupported"):
        with saved_tensor_profile.saved_tensor_scope("qkv_linear"):
            pass
