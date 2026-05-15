"""``saved_tensors_hooks`` adapter (commit 4 placeholder).

Will host the per-microbatch ``record(key, group_num)`` context manager and
the ``pack_hook`` / ``unpack_hook`` pair plugged into
``torch.autograd.graph.saved_tensors_hooks``.

Empty in commit 1.
"""
