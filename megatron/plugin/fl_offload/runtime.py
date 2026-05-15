"""Pipeline activation offload runtime façade (commit 1: empty stubs).

The runtime is the boundary the schedule wrappers will talk to in commit 5+.
For now everything is a no-op so that the plugin can be imported without any
runtime effect and downstream commits can plug in real behaviour without
changing the public surface.

Public surface frozen in this commit:

* ``PipelineActivationOffloadRuntime.enabled()``
* ``PipelineActivationOffloadRuntime.forward_microbatch(...)`` (context mgr)
* ``PipelineActivationOffloadRuntime.backward_microbatch(...)`` (context mgr)
* ``get_pipeline_offload_runtime()`` (singleton getter)
* ``OffloadAsync`` / ``OnloadAsync`` (context managers, currently no-ops)

Later commits will replace the bodies; signatures stay stable.
"""

import contextlib
from typing import Any, Optional


class OffloadAsync:
    """Context manager that will drive a microbatch's offload (commit 3+).

    Commit 1 stub: does nothing.  Kept here so import-time symbol resolution
    works for callers that already type-check against the final surface.
    """

    def __init__(self, key: Any = None, stages: int = 1) -> None:
        self.key = key
        self.stages = stages

    def __enter__(self) -> "OffloadAsync":
        return self

    def issue(self, stage_id: int) -> None:
        # Real implementation arrives in commit 3.
        return None

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class OnloadAsync:
    """Counterpart of :class:`OffloadAsync` for the backward path."""

    def __init__(self, key: Any = None, stages: int = 1) -> None:
        self.key = key
        self.stages = stages

    def __enter__(self) -> "OnloadAsync":
        return self

    def issue(self, stage_id: int) -> None:
        return None

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class PipelineActivationOffloadRuntime:
    """Schedule-facing façade.

    The schedule wrapper in commit 5 will call ``forward_microbatch`` /
    ``backward_microbatch`` once per microbatch.  In commit 1 both return
    ``contextlib.nullcontext()`` so any caller that runs ahead of the real
    implementation still works.
    """

    def enabled(self) -> bool:
        # Always False in commit 1 — the real implementation in commit 4 will
        # read ``get_config().enable``.
        return False

    def forward_microbatch(
        self,
        *,
        phase: str,
        virtual_microbatch_id: int,
        model_chunk_id: int,
        enabled: bool = True,
        offload_key: Optional[Any] = None,
    ) -> contextlib.AbstractContextManager:
        del phase, virtual_microbatch_id, model_chunk_id, enabled, offload_key
        return contextlib.nullcontext()

    def backward_microbatch(
        self,
        *,
        phase: str,
        virtual_microbatch_id: int,
        model_chunk_id: int,
        enabled: bool = True,
        offload_key: Optional[Any] = None,
    ) -> contextlib.AbstractContextManager:
        del phase, virtual_microbatch_id, model_chunk_id, enabled, offload_key
        return contextlib.nullcontext()


_RUNTIME = PipelineActivationOffloadRuntime()


def get_pipeline_offload_runtime() -> PipelineActivationOffloadRuntime:
    """Return the singleton runtime façade."""
    return _RUNTIME


__all__ = [
    "OffloadAsync",
    "OnloadAsync",
    "PipelineActivationOffloadRuntime",
    "get_pipeline_offload_runtime",
]
