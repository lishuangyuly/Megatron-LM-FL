"""TE autograd-Function monkey-patches for explicit-pack offload (commit 7.2).

``apply_te_patches()`` installs the per-op patches when fl-offload is
enabled; ``remove_te_patches()`` restores originals.  Each op lives in
its own module exposing ``build_patch() -> FunctionPatch``; adding a new
op = add a module + register it in ``_PATCH_BUILDERS``.

Version guard: the patches are written against TE 2.10.0 (rev 3c34bb9a).
We check ``transformer_engine.__version__`` and warn on mismatch; the
real safety net is the anchor-uniqueness check inside ``_patch_util``
(a moved/changed anchor raises rather than mis-patching silently).
"""

from __future__ import annotations

import warnings
from typing import Callable, List

from megatron.plugin.fl_offload.te_patch import _patch_util
from megatron.plugin.fl_offload.te_patch import grouped_linear


_EXPECTED_TE_VERSION = "2.10.0"

# op_name → builder.  Add new ops here as they are patched.
_PATCH_BUILDERS: dict = {
    "GroupedLinear": grouped_linear.build_patch,
}

_INSTALLED: List[_patch_util.FunctionPatch] = []


def _check_te_version() -> None:
    try:
        import transformer_engine as te

        ver = getattr(te, "__version__", None)
    except Exception:
        ver = None
    if ver is not None and ver != _EXPECTED_TE_VERSION:
        warnings.warn(
            f"fl-offload te_patch was written against TransformerEngine "
            f"{_EXPECTED_TE_VERSION}; installed is {ver}. The anchor-guarded "
            "patch will raise if the source moved — re-verify te_patch/ if so.",
            RuntimeWarning,
        )


def apply_te_patches() -> None:
    """Install all registered TE Function patches (idempotent).

    Called from ``fl_offload.apply`` only when the plugin is enabled.
    Anchor failures propagate (fail fast) because the user explicitly
    asked for offload and a silent no-patch would offload nothing.
    """
    if _INSTALLED:
        return
    _check_te_version()
    for op_name, builder in _PATCH_BUILDERS.items():
        try:
            patch = builder()
        except Exception as exc:  # TE module import failed → skip that op
            warnings.warn(
                f"fl-offload: could not build patch for {op_name!r}: {exc!r}. "
                "Activations from this op will not be offloaded.",
                RuntimeWarning,
            )
            continue
        _patch_util.install(patch)
        _INSTALLED.append(patch)

    import os

    if os.environ.get("FL_OFFLOAD_DEBUG_PACK") == "1":
        print(
            f"[fl-offload][PATCH] installed TE patches: "
            f"{[p.function_cls.__name__ for p in _INSTALLED]}",
            flush=True,
        )

def remove_te_patches() -> None:
    """Restore original TE Function staticmethods (tests / disable)."""
    while _INSTALLED:
        _patch_util.uninstall(_INSTALLED.pop())


def patched_op_names() -> List[str]:
    """op_names that currently have a patch available (for introspection)."""
    return list(_PATCH_BUILDERS.keys())


__all__ = ["apply_te_patches", "remove_te_patches", "patched_op_names"]
