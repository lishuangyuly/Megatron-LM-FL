"""CPU pinned-memory buffer pool.

A :class:`PinnedBufferPool` keyed by ``(num_bytes, dtype)`` recycles CPU
``torch.Tensor`` buffers so the offload runtime does not pay an
``empty_pinned`` / ``cudaHostAlloc`` cost on every microbatch.

Public surface:

* :func:`get_global_pool` — the single pool consulted by
  :class:`~megatron.plugin.fl_offload.group.ActivationGroup`.
* :meth:`PinnedBufferPool.pin_get` / :meth:`pin_put` — acquire / release
  one buffer.
* :meth:`peak_pinned_bytes` / :meth:`peak_buffer_count` — observability
  hooks consumed in commit 7.
* :func:`reset_global_pool` — test helper that wipes the singleton.

The pool falls back to **non-pinned** CPU memory when either
``cfg.pin_memory`` is ``False`` (user passed ``--fl-offload-no-pin-memory``)
or when the current platform reports it does not have an accelerator
available, so unit tests can run on CPU-only boxes.
"""

from collections import defaultdict
from typing import DefaultDict, List, Tuple

import torch

from megatron.plugin.fl_offload.config import get_config


# ``key = (num_bytes, dtype)`` — first-class so a future extension that
# stores non-uint8 buffers does not need a schema migration.  Today every
# allocation goes through ``dtype=torch.uint8``.
_BufferKey = Tuple[int, torch.dtype]


class PinnedBufferPool:
    """A simple free-list pool of CPU byte buffers, keyed by size + dtype."""

    def __init__(self) -> None:
        # ``_free[k]`` holds buffers ready to hand out.
        self._free: DefaultDict[_BufferKey, List[torch.Tensor]] = defaultdict(list)
        # Total number of buffers ever created — only grows; used to
        # confirm the pool is reused across steps (B7).
        self._created_count: int = 0
        # Total bytes currently checked out (i.e. acquired and not yet
        # returned) and the peak we've seen since construction.
        self._outstanding_bytes: int = 0
        self._peak_outstanding_bytes: int = 0

    # ------------------------------------------------------------------ #
    # Allocation                                                          #
    # ------------------------------------------------------------------ #
    def _alloc(self, num_bytes: int, dtype: torch.dtype) -> torch.Tensor:
        cfg = get_config()
        # Pinned memory only makes sense when we actually have an
        # accelerator to copy to; otherwise PyTorch warns and falls back
        # anyway.  Lazy-import ``cur_platform`` to keep this module
        # importable in stripped-down test environments.
        try:
            from megatron.plugin.platform import get_platform

            platform_available = get_platform().is_available()
        except Exception:
            platform_available = False

        use_pin = bool(getattr(cfg, "pin_memory", True)) and platform_available
        return torch.empty(num_bytes, dtype=dtype, device="cpu", pin_memory=use_pin)

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #
    def pin_get(
        self, num_bytes: int, dtype: torch.dtype = torch.uint8
    ) -> torch.Tensor:
        """Return a buffer of ``num_bytes`` bytes — pooled when possible."""
        if num_bytes <= 0:
            raise ValueError(
                f"pin_get requires num_bytes > 0, got {num_bytes}"
            )
        key = (num_bytes, dtype)
        free_list = self._free[key]
        if free_list:
            buf = free_list.pop()
        else:
            buf = self._alloc(num_bytes, dtype)
            self._created_count += 1
        self._outstanding_bytes += num_bytes
        if self._outstanding_bytes > self._peak_outstanding_bytes:
            self._peak_outstanding_bytes = self._outstanding_bytes
        return buf

    def pin_put(self, buf: torch.Tensor) -> None:
        """Return ``buf`` to the pool for reuse."""
        if buf is None:
            return
        if not isinstance(buf, torch.Tensor):
            raise TypeError(
                f"pin_put expects a torch.Tensor, got {type(buf).__name__}"
            )
        num_bytes = buf.numel() * buf.element_size()
        key = (num_bytes, buf.dtype)
        self._free[key].append(buf)
        self._outstanding_bytes -= num_bytes
        if self._outstanding_bytes < 0:
            # Defensive: more puts than gets is a programmer error.
            raise RuntimeError(
                "PinnedBufferPool outstanding bytes went negative — "
                "pin_put called more times than pin_get."
            )

    # ------------------------------------------------------------------ #
    # Observability                                                       #
    # ------------------------------------------------------------------ #
    def peak_pinned_bytes(self) -> int:
        """Largest sum of outstanding (checked-out) buffer bytes ever seen."""
        return self._peak_outstanding_bytes

    def peak_buffer_count(self) -> int:
        """Total number of buffers ever allocated by this pool.

        Equivalent to the high-water mark of unique buffer objects; useful
        for spotting a leak where ``pin_put`` is never called and the pool
        keeps growing.
        """
        return self._created_count

    def outstanding_bytes(self) -> int:
        return self._outstanding_bytes


_GLOBAL_POOL: PinnedBufferPool = PinnedBufferPool()


def get_global_pool() -> PinnedBufferPool:
    """Return the per-process singleton pool."""
    return _GLOBAL_POOL


def reset_global_pool() -> None:
    """Replace the singleton with a fresh, empty pool. Tests only."""
    global _GLOBAL_POOL
    _GLOBAL_POOL = PinnedBufferPool()


__all__ = [
    "PinnedBufferPool",
    "get_global_pool",
    "reset_global_pool",
]
