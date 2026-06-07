"""Configuration dataclass for the fl-offload plugin.

This module hosts a single ``FlOffloadConfig`` dataclass plus a module-level
getter / setter pair.  The default config has ``enable=False``, so simply
importing the plugin never changes behaviour.

Later commits will:

* commit 2 — populate the config from argparse and ``set_config(...)`` after
  validation;
* commit 3 / 4 — read fields from ``get_config()`` inside the filter, pool,
  ``ActivationGroup`` and runtime façade.
"""

from dataclasses import dataclass, field
from typing import List, Optional


_DEFAULT_MIN_BYTES = 1 << 20  # 1 MiB


@dataclass
class FlOffloadConfig:
    """All runtime knobs for the fl-offload plugin.

    Field names line up 1:1 with the CLI flags introduced in commit 2 (after
    stripping the ``--fl-offload-`` prefix and replacing dashes with
    underscores).  Defaults match the "do nothing" path so that constructing
    the dataclass with no arguments yields a no-op runtime.
    """

    # Master switch.  When False every façade method short-circuits.
    enable: bool = False

    # Which Megatron schedule backend the user explicitly asked for.
    # "vanilla" defers to upstream FL behaviour; "interleaved_1f1b" forces the
    # interleaved schedule (validated in commit 2).
    pipeline_schedule_backend: str = "vanilla"

    # Tensor filter knobs (commit 3).
    min_bytes: int = _DEFAULT_MIN_BYTES
    non_contiguous: bool = False

    # CPU buffer pool knobs (commit 3).
    pin_memory: bool = True

    # Budget knobs (commit 3).  When per_batch_size > 0 it takes precedence
    # over ratio.  ``per_batch_size`` is measured in MiB so users can pass
    # fractional budgets such as ``0.5``.
    ratio: float = 1.0
    per_batch_size: float = 0.0  # in MiB; 0 means "use ratio"

    # Number of copy buckets the offload / onload work is split into
    # (commit 3).
    stages: int = 1

    # Observability knobs (commit 7).  0 disables periodic reporting.
    report_interval: int = 0
    allow_cuda_graph: bool = False


_CONFIG: FlOffloadConfig = FlOffloadConfig()

# True once ``set_config`` ran, i.e. the config reflects validated args
# rather than the import-time default.  The schedule selector uses this
# to decide whether it still needs to land the config lazily from
# megatron's global args (FL's ``pretrain()`` has no validate hook the
# plugin could chain into).
_LANDED: bool = False


def get_config() -> FlOffloadConfig:
    """Return the currently installed plugin config (singleton)."""
    return _CONFIG


def config_landed() -> bool:
    """True once :func:`set_config` has installed a real config."""
    return _LANDED


def set_config(cfg: FlOffloadConfig) -> None:
    """Replace the singleton config.

    The plugin's validate hook (commit 2) calls this after argparse so that
    downstream modules can read knobs via ``get_config()`` without threading
    the dataclass through every call site.
    """
    if not isinstance(cfg, FlOffloadConfig):
        raise TypeError(
            "set_config expects a FlOffloadConfig instance, got "
            f"{type(cfg).__name__}"
        )
    global _CONFIG, _LANDED
    _CONFIG = cfg
    _LANDED = True


def _reset_landed_for_tests() -> None:
    """Pretend ``set_config`` never ran.  Tests only."""
    global _LANDED
    _LANDED = False


__all__ = ["FlOffloadConfig", "config_landed", "get_config", "set_config"]
