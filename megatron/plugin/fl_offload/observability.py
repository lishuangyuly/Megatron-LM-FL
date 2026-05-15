"""Per-batch reporting & degradation warnings (commit 7 placeholder).

Will host ``report_per_batch(step_id)`` that prints rank-0 lines covering
pinned-pool peak bytes, residual ``_GROUPS`` keys, offloaded tensor count
and bytes, and the running degradation-warning counter.

Empty in commit 1.
"""
