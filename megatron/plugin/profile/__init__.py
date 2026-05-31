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
from .autograd_record import BWDRecordEnd, BWDRecordStart, bwd_record_pair

__all__ = [
    "PREFIX",
    "BWDRecordEnd",
    "BWDRecordStart",
    "bwd_record_pair",
    "gpu_anchor",
    "is_profile_enabled",
    "make_name",
    "profile_it",
    "profile_it_with_anchor",
    "set_profile_enabled",
    "step_record_function",
]
