"""CPU pinned-memory buffer pool (commit 3 placeholder).

Will host ``PinnedBufferPool`` keyed by ``(num_bytes, dtype)`` with
``pin_get`` / ``pin_put`` and peak-tracking accessors used by the
observability hook in commit 7.

Empty in commit 1.
"""
