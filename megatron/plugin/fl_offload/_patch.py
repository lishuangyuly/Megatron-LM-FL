"""Plugin-private Megatron monkey-patch helper (commit 5 placeholder).

Will host the ``Patch`` dataclass and ``MegatronPatchesManager`` static
registry used to wrap
``megatron.core.pipeline_parallel.schedules.get_forward_backward_func``.
The key reason this module is plugin-private is the alias-sync trick: many
FL submodules do ``from ... import get_forward_backward_func`` at import
time, so a plain ``setattr`` on the owning module is not enough — we must
walk ``sys.modules`` and replace any binding still pointing at the original
callable.

Empty in commit 1.

Private helper. Do NOT import from outside ``megatron.plugin.fl_offload``.
"""
