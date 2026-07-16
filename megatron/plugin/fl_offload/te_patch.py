"""Version adaptation for FL's explicit TE pack/unpack modifications."""

import inspect
import sys
import textwrap


_ORIGINALS = []


def _replace_once(source, anchor, replacement, owner):
    count = source.count(anchor)
    if count != 1:
        raise RuntimeError(
            f"FL offload TE patch expected one anchor in {owner}, found {count}: {anchor!r}"
        )
    return source.replace(anchor, replacement)


def _compile(method, substitutions, name):
    source = textwrap.dedent(inspect.getsource(method))
    lines = source.splitlines()
    decorators = []
    while lines and not lines[0].lstrip().startswith("def "):
        line = lines.pop(0)
        if line.lstrip().startswith("@") and line.strip() != "@staticmethod":
            decorators.append(line)
    source = "\n".join(decorators + lines)
    for anchor, replacement in substitutions:
        source = _replace_once(source, anchor, replacement, method.__qualname__)
    source = source.replace("def forward(", f"def {name}(", 1)
    source = source.replace("def backward(", f"def {name}(", 1)
    namespace = {}
    module = sys.modules[method.__module__]
    exec(compile(source, f"<fl_offload:{method.__qualname__}>", "exec"), module.__dict__, namespace)
    return namespace[name]


def _patch_class(cls, forward_subs, backward_subs):
    forward = _compile(cls.forward, forward_subs, "_fl_offload_forward")
    backward = _compile(cls.backward, backward_subs, "_fl_offload_backward")
    _ORIGINALS.append((cls, cls.forward, cls.backward))
    cls.forward = staticmethod(forward)
    cls.backward = staticmethod(backward)


def apply_te_patches():
    if _ORIGINALS:
        return
    from transformer_engine.pytorch.module.grouped_linear import _GroupedLinear
    from transformer_engine.pytorch.module.layernorm_linear import _LayerNormLinear

    ln_forward = (
        """        tensors_to_save, tensor_objects = prepare_for_saving(
            inputmat,
            weightmat,
            weight,
            bias,
            ln_weight,
            ln_out,
            mu,
            rsigma,
        )""",
        """        from megatron.plugin.fl_offload.offload import pack_hook as _fl_pack
        ctx.fl_tensor_pack = _fl_pack(inputmat, op_name="LayerNormLinear")
        tensors_to_save, tensor_objects = prepare_for_saving(
            weightmat,
            weight,
            bias,
            ln_weight,
            ln_out,
            mu,
            rsigma,
        )""",
    )
    ln_backward = (
        """        (  # pylint: disable=unbalanced-tuple-unpacking
            inputmat,
            weight,
            origin_weight,
            bias,
            ln_weight,
            ln_out,
            mu,
            rsigma,
        ) = restore_from_saved(ctx.tensor_objects, saved_tensors)""",
        """        from megatron.plugin.fl_offload.offload import unpack_hook as _fl_unpack
        (  # pylint: disable=unbalanced-tuple-unpacking
            weight,
            origin_weight,
            bias,
            ln_weight,
            ln_out,
            mu,
            rsigma,
        ) = restore_from_saved(ctx.tensor_objects, saved_tensors)
        inputmat = _fl_unpack(ctx.fl_tensor_pack)""",
    )
    grouped_forward = (
        """        tensors_to_save, tensor_objects = prepare_for_saving(
            *inputmats,
            *weights_fp8,
            *weights,
            *biases,
        )""",
        """        from megatron.plugin.fl_offload.offload import pack_hook as _fl_pack
        ctx.fl_tensor_packs = [_fl_pack(t, op_name="GroupedLinear") for t in inputmats]
        tensors_to_save, tensor_objects = prepare_for_saving(
            *weights_fp8,
            *weights,
            *biases,
        )""",
    )
    grouped_backward = (
        """        inputmats = saved_tensors[:N]
        weights = saved_tensors[N : 2 * N]
        origin_weights = saved_tensors[2 * N : 3 * N]
        biases = saved_tensors[3 * N : 4 * N]""",
        """        from megatron.plugin.fl_offload.offload import unpack_hook as _fl_unpack
        inputmats = [_fl_unpack(t) for t in ctx.fl_tensor_packs]
        weights = saved_tensors[0:N]
        origin_weights = saved_tensors[N : 2 * N]
        biases = saved_tensors[2 * N : 3 * N]""",
    )
    try:
        _patch_class(_LayerNormLinear, [ln_forward], [ln_backward])
        _patch_class(_GroupedLinear, [grouped_forward], [grouped_backward])
    except Exception:
        restore_te_patches()
        raise


def restore_te_patches():
    while _ORIGINALS:
        cls, forward, backward = _ORIGINALS.pop()
        cls.forward = staticmethod(forward)
        cls.backward = staticmethod(backward)
