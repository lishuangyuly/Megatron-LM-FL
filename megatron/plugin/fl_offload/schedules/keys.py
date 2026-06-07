"""Offload-key derivation helpers.

The interleaved 1F1B key is the **logical microbatch identity**::

    ("ilv", microbatch_id_in_chunk, model_chunk_id)

Why this pair: FL's interleaved schedule builds a lookup table mapping
``virtual_microbatch_id -> (microbatch_id, model_chunk_id)`` (see
``get_schedule_table`` in ``schedules.py``).  ``microbatch_id`` is the
index *within* a model chunk, and the table is direction-agnostic — the
backward pass walks the same table with the model chunk mirrored
(``num_chunks - 1 - model_chunk_id``).  After mirroring, forward and
backward for the same logical microbatch agree on the pair, so a group
stored at forward time is found again at backward time on the same
rank, for any ``microbatch_group_size_per_vp_stage``.

Call sites:

* conventional interleaved path — ``forward_step`` receives exactly
  these two values as ``current_microbatch`` / ``vp_stage``; the
  backward side recovers the key via the ``id(output_tensor)``
  registry, so only the forward derivation matters there.
* combined (``overlap_moe_expert_parallel_comm``) path — the schedule
  helper receives both virtual ids plus the table lookup callables, so
  both sides derive the key independently and must agree (the
  symmetric pair guarantees it).

``make_offload_key_no_interleave`` (commit 9) will provide the simpler
single-element key for non-interleaved / no_pipelining paths.
"""

from __future__ import annotations

from typing import Tuple


def make_offload_key_interleaved(
    microbatch_id: int,
    model_chunk_id: int,
    forward: bool = True,
) -> Tuple[str, int, int]:
    """Return ``("ilv", microbatch_id, model_chunk_id)``.

    Args:
        microbatch_id: Microbatch index *within* the model chunk (what
            FL passes to ``forward_step`` as ``current_microbatch``,
            i.e. ``microbatch_id_table[virtual_microbatch_id]``).
        model_chunk_id: The model chunk owning this microbatch, in
            *forward* orientation (``get_model_chunk_id(vmb, forward)``
            already mirrors for the backward direction).
        forward: Advisory only — kept on the signature so call sites
            self-document direction; the returned key is identical
            either way, which is exactly the pairing property the
            backward lookup relies on.
    """
    del forward  # advisory only
    return ("ilv", int(microbatch_id), int(model_chunk_id))


__all__ = ["make_offload_key_interleaved"]
