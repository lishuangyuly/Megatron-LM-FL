"""Megatron schedule wrappers (commit 5 / 6 placeholder).

Will host ``wrap_schedule_for_offload(fn)`` and the ``_patch_step_funcs(...)``
context manager that temporarily swaps
``megatron.core.pipeline_parallel.schedules.forward_step`` /
``backward_step`` for offload-aware versions while the schedule is running.

Empty in commit 1.
"""
