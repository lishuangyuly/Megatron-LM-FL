"""Plugin-private Megatron monkey-patch helper (commit 6).

Hosts the :class:`Patch` dataclass and :class:`MegatronPatchesManager`
static registry used to wrap
``megatron.core.pipeline_parallel.schedules.get_forward_backward_func``.

Key trick: many FL submodules do
``from megatron.core.pipeline_parallel.schedules import
get_forward_backward_func`` at import time, binding a local alias.  A
plain ``setattr`` on the owning module is therefore not enough — we
must also walk ``sys.modules`` and replace any binding that still
points at the original callable.

Private helper. Do NOT import from outside ``megatron.plugin.fl_offload``.
"""

from __future__ import annotations

import dataclasses
import importlib
import sys
from typing import Any, Callable, List


@dataclasses.dataclass
class Patch:
    """One registered monkey-patch.

    Attributes:
        target: Dotted path to the attribute being replaced
            (e.g. ``"megatron.core.pipeline_parallel.schedules.get_forward_backward_func"``).
        replacement: Either the new value directly, or — when
            ``apply_wrapper`` is True — a wrapper factory of signature
            ``factory(original) -> replacement``.
        apply_wrapper: If True, ``replacement`` is treated as a factory.
        original: Populated by :meth:`MegatronPatchesManager.apply_patches`
            so undo / introspection works.
    """

    target: str
    replacement: Any
    apply_wrapper: bool = False
    original: Any = None


class MegatronPatchesManager:
    """Static registry of patches; apply / undo on demand."""

    _patches: List[Patch] = []
    _applied: bool = False

    @classmethod
    def register_patch(
        cls,
        target: str,
        replacement: Any,
        *,
        apply_wrapper: bool = False,
    ) -> None:
        cls._patches.append(
            Patch(target=target, replacement=replacement, apply_wrapper=apply_wrapper)
        )

    @classmethod
    def apply_patches(cls) -> None:
        if cls._applied:
            return
        for p in cls._patches:
            module_path, _, attr = p.target.rpartition(".")
            module = importlib.import_module(module_path)
            original = getattr(module, attr)
            p.original = original
            replacement = (
                p.replacement(original) if p.apply_wrapper else p.replacement
            )
            setattr(module, attr, replacement)
            _replace_matching_aliases(original, replacement)
        cls._applied = True

    @classmethod
    def undo_patches(cls) -> None:
        """Reverse every applied patch.  Idempotent."""
        if not cls._applied:
            return
        # Reverse order so dependent patches unwind cleanly.
        for p in reversed(cls._patches):
            module_path, _, attr = p.target.rpartition(".")
            try:
                module = importlib.import_module(module_path)
            except Exception:
                continue
            current = getattr(module, attr, None)
            setattr(module, attr, p.original)
            if current is not None and current is not p.original:
                _replace_matching_aliases(current, p.original)
        cls._applied = False

    @classmethod
    def _reset_for_tests(cls) -> None:
        """Undo and clear the registry. Tests only."""
        cls.undo_patches()
        cls._patches.clear()


def _replace_matching_aliases(original: Any, replacement: Any) -> None:
    """Walk ``sys.modules`` and replace every binding that ``is`` ``original``.

    ``from owner import attr`` and ``from owner import attr as alias``
    both create independent bindings in the importing module's
    ``__dict__``.  After we ``setattr`` the owning module's attribute,
    those local bindings still point at the *old* object — so we walk
    every loaded module and fix them up by identity.
    """
    for module in list(sys.modules.values()):
        if module is None:
            continue
        d = getattr(module, "__dict__", None)
        if not isinstance(d, dict):
            continue
        for name, value in list(d.items()):
            if value is original:
                try:
                    d[name] = replacement
                except Exception:
                    pass


__all__ = ["MegatronPatchesManager", "Patch"]
