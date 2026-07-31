from types import SimpleNamespace

from megatron.plugin.fl_offload import offload, te_patch


def prepare_for_saving(*tensors):
    return tensors, "objects"


def restore_from_saved(_objects, saved_tensors):
    return saved_tensors


class _FakeLayerNormLinear:
    @staticmethod
    def forward(ctx, inputmat, weightmat, weight, bias, ln_weight, ln_out, mu, rsigma):
        if True:
            tensors_to_save, tensor_objects = prepare_for_saving(
                inputmat,
                weightmat,
                weight,
                bias,
                ln_weight,
                ln_out,
                mu,
                rsigma,
            )
            ctx.tensor_objects = tensor_objects
            return tensors_to_save

    @staticmethod
    def backward(ctx, saved_tensors):
        if True:
            (  # pylint: disable=unbalanced-tuple-unpacking
                inputmat,
                weight,
                origin_weight,
                bias,
                ln_weight,
                ln_out,
                mu,
                rsigma,
            ) = restore_from_saved(ctx.tensor_objects, saved_tensors)
            return inputmat, weight, origin_weight, bias, ln_weight, ln_out, mu, rsigma


def test_layernorm_linear_patch_explicitly_packs_input_and_ln_out(monkeypatch):
    packed = []

    def pack(tensor, op_name=None):
        packed.append((tensor, op_name))
        return ("packed", tensor)

    monkeypatch.setattr(offload, "pack_hook", pack)
    monkeypatch.setattr(offload, "unpack_hook", lambda value: value[1])
    forward_patch, backward_patch = te_patch._layernorm_linear_patches()

    try:
        te_patch._patch_class(
            _FakeLayerNormLinear,
            [forward_patch],
            [backward_patch],
        )
        ctx = SimpleNamespace()
        values = tuple(object() for _ in range(8))
        saved = _FakeLayerNormLinear.forward(ctx, *values)

        assert packed == [
            (values[0], "LayerNormLinear"),
            (values[5], "LayerNormLinear"),
        ]
        assert saved == (values[1], values[2], values[3], values[4], values[6], values[7])

        restored = _FakeLayerNormLinear.backward(ctx, saved)
        assert restored == (
            values[0],
            values[1],
            values[2],
            values[3],
            values[4],
            values[5],
            values[6],
            values[7],
        )
    finally:
        te_patch.restore_te_patches()
