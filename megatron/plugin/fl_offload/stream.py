"""CUDA stream wrappers for the fl-offload plugin (commit 3 placeholder).

Will host ``get_memcpy_stream(key)`` returning lazily-created dedicated
streams for ``"offload"`` (D2H) and ``"onload"`` (H2D) work, using
``cur_platform.Stream()`` so non-CUDA platforms can degrade gracefully.

Empty in commit 1.
"""
