"""Monkey-patch helper for explicit-pack offload on TE autograd Functions.

Strategy (commit 7.2): instead of vendoring whole TE files, we take the
*source* of a Function's ``forward`` / ``backward`` (via ``inspect``),
apply small textual substitutions at well-known anchor lines, recompile
the result **in the original module's global namespace** (so every TE
symbol it references — ``prepare_for_saving``, ``tex``, quantizers, … —
resolves unchanged), and reassign the staticmethods.

Why text substitution rather than wrapping: the only change needed is to
divert specific activations from ``save_for_backward`` to ``pack_hook``
and recover them in backward — and that must happen *inside* the long
forward/backward bodies (TE flattens the inputmats/weights/biases into a
single save list; the semantic grouping only exists locally).  Wrapping
the outer function cannot reach those lines.  Copying the whole body
into our repo would couple us to ~450 lines of fast-moving TE internals
per Function; source-substitution keeps the diff to the few anchor lines
and self-heals across TE point releases as long as the anchors survive
(guarded: a missing anchor raises, so a silent wrong patch is impossible).

Each op is one :class:`FunctionPatch`; ``install_all`` applies them,
``uninstall_all`` restores originals (for tests / disable).
"""

from __future__ import annotations

import inspect
import sys
import textwrap
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple


@dataclass
class _Substitution:
    """One anchored text replacement applied to a function's source."""

    anchor: str          # must appear exactly once in the source
    replacement: str     # replaces ``anchor`` verbatim


@dataclass
class FunctionPatch:
    """Patch spec for one ``torch.autograd.Function``'s fwd/bwd."""

    function_cls: type
    forward_subs: List[_Substitution] = field(default_factory=list)
    backward_subs: List[_Substitution] = field(default_factory=list)
    # Saved originals for uninstall.
    _orig_forward: Callable = field(default=None, repr=False)
    _orig_backward: Callable = field(default=None, repr=False)
    _installed: bool = field(default=False, repr=False)


def _patched_source(method: Callable, subs: List[_Substitution], new_name: str) -> str:
    """Return de-dented, de-decorated, substituted source for ``method``."""
    src = textwrap.dedent(inspect.getsource(method))
    # Drop decorator lines (``@staticmethod`` etc.) above ``def``.
    lines = src.splitlines()
    while lines and not lines[0].lstrip().startswith("def "):
        lines.pop(0)
    src = "\n".join(lines)
    for sub in subs:
        count = src.count(sub.anchor)
        if count != 1:
            raise RuntimeError(
                f"fl-offload te_patch: anchor not unique in {method.__qualname__} "
                f"(found {count}×, need 1):\n  {sub.anchor!r}\n"
                "TE source may have changed — re-verify the patch."
            )
        src = src.replace(sub.anchor, sub.replacement)
    # Rename so the compiled function object is identifiable in tracebacks.
    src = src.replace("def forward(", f"def {new_name}(", 1)
    src = src.replace("def backward(", f"def {new_name}(", 1)
    return src


def _compile_in_module_namespace(method: Callable, src: str, new_name: str) -> Callable:
    """Exec ``src`` in the namespace of the module that defines ``method``."""
    module = sys.modules[method.__module__]
    ns: Dict = {}
    code = compile(src, f"<fl_offload_patch:{method.__qualname__}>", "exec")
    exec(code, module.__dict__, ns)  # noqa: S102 — intentional, TE namespace
    fn = ns[new_name]
    return fn


def install(patch: FunctionPatch) -> None:
    """Apply one :class:`FunctionPatch` (idempotent)."""
    if patch._installed:
        return
    cls = patch.function_cls
    patch._orig_forward = cls.forward
    patch._orig_backward = cls.backward

    if patch.forward_subs:
        fsrc = _patched_source(cls.forward, patch.forward_subs, "_fl_forward")
        fwd = _compile_in_module_namespace(cls.forward, fsrc, "_fl_forward")
        cls.forward = staticmethod(fwd)
    if patch.backward_subs:
        bsrc = _patched_source(cls.backward, patch.backward_subs, "_fl_backward")
        bwd = _compile_in_module_namespace(cls.backward, bsrc, "_fl_backward")
        cls.backward = staticmethod(bwd)
    patch._installed = True


def uninstall(patch: FunctionPatch) -> None:
    """Restore the original fwd/bwd staticmethods."""
    if not patch._installed:
        return
    if patch._orig_forward is not None:
        patch.function_cls.forward = staticmethod(patch._orig_forward)
    if patch._orig_backward is not None:
        patch.function_cls.backward = staticmethod(patch._orig_backward)
    patch._installed = False


__all__ = ["FunctionPatch", "_Substitution", "install", "uninstall"]
