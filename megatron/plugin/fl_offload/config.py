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
    # over ratio.
    ratio: float = 1.0
    per_batch_size: int = 0  # in MiB; 0 means "use ratio"

    # Number of copy buckets the offload / onload work is split into
    # (commit 3).
    stages: int = 1

    # Observability knobs (commit 7).  0 disables periodic reporting.
    report_interval: int = 0
    allow_cuda_graph: bool = False


_CONFIG: FlOffloadConfig = FlOffloadConfig()


def get_config() -> FlOffloadConfig:
    """Return the currently installed plugin config (singleton)."""
    return _CONFIG


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
    global _CONFIG
    _CONFIG = cfg


__all__ = ["FlOffloadConfig", "get_config", "set_config"]
