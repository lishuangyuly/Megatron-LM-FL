"""Pipeline-semantic chrome trace instrumentation utilities.

See profile_docs/design.md and profile_docs/plan.md.
"""

from .core import (
    PREFIX,
    gpu_anchor,
    is_profile_enabled,
    make_name,
    profile_it,
    profile_it_with_anchor,
    set_profile_enabled,
    step_record_function,
)
from .autograd_record import (
    BWDRecordEnd,
    BWDRecordEndAnchored,
    BWDRecordStart,
    BWDRecordStartAnchored,
    bwd_record_pair,
    bwd_record_pair_with_anchor,
)

__all__ = [
    "PREFIX",
    "BWDRecordEnd",
    "BWDRecordEndAnchored",
    "BWDRecordStart",
    "BWDRecordStartAnchored",
    "bwd_record_pair",
    "bwd_record_pair_with_anchor",
    "gpu_anchor",
    "is_profile_enabled",
    "make_name",
    "profile_it",
    "profile_it_with_anchor",
    "set_profile_enabled",
    "step_record_function",
]
