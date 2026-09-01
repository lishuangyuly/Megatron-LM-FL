"""Version adaptation for FL's explicit TE pack/unpack modifications."""

import inspect
import sys
import textwrap


_ORIGINALS = []
_MODULE_FORWARD_ORIGINALS = []


def _layernorm_linear_patches():
    forward = (
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
        """        from megatron.plugin.fl_offload.offload import (
            current_tensor_scope as _fl_scope,
            pack_hook as _fl_pack,
        )
        _fl_op_name, _fl_tensor_name = _fl_scope("LayerNormLinear", None)
        ctx.fl_tensor_pack = _fl_pack(
            inputmat, op_name=_fl_op_name, tensor_name=_fl_tensor_name
        )
        ctx.fl_ln_out_pack = _fl_pack(
            ln_out,
            op_name=_fl_op_name,
            tensor_name=(
                f"{_fl_tensor_name}.ln_out" if _fl_tensor_name else "ln_out"
            ),
        )
        tensors_to_save, tensor_objects = prepare_for_saving(
            weightmat,
            weight,
            bias,
            ln_weight,
            mu,
            rsigma,
        )""",
    )
    backward = (
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
            mu,
            rsigma,
        ) = restore_from_saved(ctx.tensor_objects, saved_tensors)
        inputmat = _fl_unpack(ctx.fl_tensor_pack)
        ln_out = _fl_unpack(ctx.fl_ln_out_pack)""",
    )
    return forward, backward


def _flash_attention_patches():
    forward = (
        """        ctx.save_for_backward(q, k, v, out_padded, softmax_lse, rng_state)""",
        """        from megatron.plugin.fl_offload.offload import pack_hook as _fl_pack
        ctx.fl_attention_packs = (
            _fl_pack(q, op_name="FlashAttention", tensor_name="q"),
            _fl_pack(k, op_name="FlashAttention", tensor_name="k"),
            _fl_pack(v, op_name="FlashAttention", tensor_name="v"),
            _fl_pack(out_padded, op_name="FlashAttention", tensor_name="output"),
            _fl_pack(softmax_lse, op_name="FlashAttention", tensor_name="softmax_lse"),
        )
        ctx.save_for_backward(rng_state)""",
    )
    backward = (
        """    q, k, v, out, softmax_lse, rng_state = ctx.saved_tensors""",
        """    from megatron.plugin.fl_offload.offload import unpack_hook as _fl_unpack
    q, k, v, out, softmax_lse = (
        _fl_unpack(tensor_pack) for tensor_pack in ctx.fl_attention_packs
    )
    (rng_state,) = ctx.saved_tensors""",
    )
    return forward, backward


def _flash_attention_varlen_patches():
    forward = (
        """        ctx.save_for_backward(
            q, k, v, out_padded, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state
        )""",
        """        from megatron.plugin.fl_offload.offload import pack_hook as _fl_pack
        ctx.fl_attention_packs = (
            _fl_pack(q, op_name="FlashAttention", tensor_name="q"),
            _fl_pack(k, op_name="FlashAttention", tensor_name="k"),
            _fl_pack(v, op_name="FlashAttention", tensor_name="v"),
            _fl_pack(out_padded, op_name="FlashAttention", tensor_name="output"),
            _fl_pack(softmax_lse, op_name="FlashAttention", tensor_name="softmax_lse"),
        )
        ctx.save_for_backward(cu_seqlens_q, cu_seqlens_k, rng_state)""",
    )
    backward = (
        (
            "    q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k, "
            "rng_state = ctx.saved_tensors"
        ),
        """    from megatron.plugin.fl_offload.offload import unpack_hook as _fl_unpack
    q, k, v, out, softmax_lse = (
        _fl_unpack(tensor_pack) for tensor_pack in ctx.fl_attention_packs
    )
    cu_seqlens_q, cu_seqlens_k, rng_state = ctx.saved_tensors""",
    )
    return forward, backward


def _fused_attention_patches():
    forward = (
        """    tensors_to_save, tensor_objects = prepare_for_saving(
        *fp8_tensors,
        *qkvo_tensors,
        cu_seqlens_q,
        cu_seqlens_kv,
        cu_seqlens_q_padded,
        cu_seqlens_kv_padded,
        *aux_ctx_tensors,
    )""",
        """    from megatron.plugin.fl_offload.offload import (
        pack_fused_attention_saved_tensors as _fl_pack_fused,
    )
    (
        fp8_tensors,
        qkvo_tensors,
        aux_ctx_tensors,
        ctx.fl_fused_attention_packs,
    ) = _fl_pack_fused(fp8_tensors, qkvo_tensors, aux_ctx_tensors)
    tensors_to_save, tensor_objects = prepare_for_saving(
        *fp8_tensors,
        *qkvo_tensors,
        cu_seqlens_q,
        cu_seqlens_kv,
        cu_seqlens_q_padded,
        cu_seqlens_kv_padded,
        *aux_ctx_tensors,
    )""",
    )
    backward = (
        """    aux_ctx_tensors = other_tensors""",
        """    from megatron.plugin.fl_offload.offload import (
        unpack_fused_attention_saved_tensors as _fl_unpack_fused,
    )
    (
        (q_fp8, k_fp8, v_fp8, out_fp8),
        (q, k, v, out),
        aux_ctx_tensors,
    ) = _fl_unpack_fused(
        ctx.fl_fused_attention_packs,
        (q_fp8, k_fp8, v_fp8, out_fp8),
        (q, k, v, out),
        other_tensors,
    )""",
    )
    return forward, backward


def _linear_input_patches():
    forward = (
        """        tensors_to_save, tensor_objects = prepare_for_saving(
            saved_inputmat,
            weightmat,
            weight,
            bias,
        )""",
        """        from megatron.plugin.fl_offload.offload import (
            maybe_pack_linear_input as _fl_maybe_pack,
        )
        ctx.fl_linear_input_pack = _fl_maybe_pack(saved_inputmat)
        tensors_to_save, tensor_objects = prepare_for_saving(
            None if ctx.fl_linear_input_pack is not None else saved_inputmat,
            weightmat,
            weight,
            bias,
        )""",
    )
    backward = (
        (
            "        inputmat, weight_fp8, weight, bias = ("
            "  # pylint: disable=unbalanced-tuple-unpacking\n"
            "            restore_from_saved(ctx.tensor_objects, saved_tensors)\n"
            "        )"
        ),
        (
            "        inputmat, weight_fp8, weight, bias = ("
            "  # pylint: disable=unbalanced-tuple-unpacking\n"
            "            restore_from_saved(ctx.tensor_objects, saved_tensors)\n"
            "        )\n"
            "        if ctx.fl_linear_input_pack is not None:\n"
            "            from megatron.plugin.fl_offload.offload import "
            "unpack_hook as _fl_unpack\n"
            "            inputmat = _fl_unpack(ctx.fl_linear_input_pack)"
        ),
    )
    return forward, backward


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


def _patch_unfused_attention(cls):
    original_forward = cls.forward
    signature = inspect.signature(original_forward)

    def forward(module, *args, **kwargs):
        bound = signature.bind(module, *args, **kwargs)
        query = bound.arguments["query_layer"]
        key = bound.arguments["key_layer"]
        value = bound.arguments["value_layer"]
        from megatron.plugin.fl_offload.offload import (
            maybe_pack_unfused_attention_output,
            unfused_attention_saved_tensors,
        )

        with unfused_attention_saved_tensors(query, key, value):
            output = original_forward(module, *args, **kwargs)
        context = output[0] if isinstance(output, tuple) else output
        maybe_pack_unfused_attention_output(context)
        return output

    _MODULE_FORWARD_ORIGINALS.append((cls, original_forward))
    cls.forward = forward


def _attention_backend(args):
    backend = getattr(args, "attention_backend", "auto")
    name = getattr(backend, "name", backend)
    return str(name).lower().rsplit(".", maxsplit=1)[-1]


def apply_te_patches():
    if _ORIGINALS or _MODULE_FORWARD_ORIGINALS:
        return
    from transformer_engine.pytorch.module.grouped_linear import _GroupedLinear
    from transformer_engine.pytorch.module.layernorm_linear import _LayerNormLinear
    from transformer_engine.pytorch.module.linear import _Linear

    from megatron.plugin.fl_offload.offload import _args

    ln_forward, ln_backward = _layernorm_linear_patches()
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
        args = _args()
        modules = set(getattr(args, "fl_offload_modules", []) or [])
        backend = _attention_backend(args)
        generic_attention = "Attention" in modules
        patch_flash = "FlashAttention" in modules or (
            generic_attention and backend in ("auto", "flash")
        )
        patch_fused = "FusedAttention" in modules or (
            generic_attention and backend in ("auto", "fused")
        )
        patch_unfused = "UnfusedAttention" in modules or (
            generic_attention and backend in ("auto", "unfused")
        )
        if patch_flash:
            try:
                from flash_attn.flash_attn_interface import (
                    FlashAttnFunc,
                    FlashAttnVarlenFunc,
                )
            except ImportError as exc:
                if backend == "flash" or "FlashAttention" in modules:
                    raise RuntimeError(
                        "FL Attention offload with the flash backend requires flash-attn v2"
                    ) from exc
            else:
                flash_forward, flash_backward = _flash_attention_patches()
                varlen_forward, varlen_backward = _flash_attention_varlen_patches()
                _patch_class(FlashAttnFunc, [flash_forward], [flash_backward])
                _patch_class(FlashAttnVarlenFunc, [varlen_forward], [varlen_backward])
        if patch_fused:
            from transformer_engine.pytorch.attention.dot_product_attention.backends import (
                FusedAttnFunc,
            )

            fused_forward, fused_backward = _fused_attention_patches()
            _patch_class(FusedAttnFunc, [fused_forward], [fused_backward])
        if patch_unfused:
            from transformer_engine.pytorch.attention.dot_product_attention.backends import (
                UnfusedDotProductAttention,
            )

            _patch_unfused_attention(UnfusedDotProductAttention)
        if any(
            module in modules
            for module in (
                "Attention",
                "FlashAttention",
                "FusedAttention",
                "UnfusedAttention",
                "MLA",
                "SharedExpert",
                "MTP",
            )
        ):
            linear_forward, linear_backward = _linear_input_patches()
            _patch_class(_Linear, [linear_forward], [linear_backward])
    except Exception:
        restore_te_patches()
        raise


def restore_te_patches():
    while _MODULE_FORWARD_ORIGINALS:
        cls, forward = _MODULE_FORWARD_ORIGINALS.pop()
        cls.forward = forward
    while _ORIGINALS:
        cls, forward, backward = _ORIGINALS.pop()
        cls.forward = staticmethod(forward)
        cls.backward = staticmethod(backward)
