"""Offload-key derivation helpers.

``make_offload_key_interleaved`` (commit 6) returns a three-tuple key
for ``forward_backward_pipelining_with_interleaving``.  Forward and
backward calls with the same
``(virtual_microbatch_id, model_chunk_id)`` arguments produce the same
key, so a group stored under that key during forward is reliably found
by the matching backward step.

``make_offload_key_no_interleave`` (commit 7) will provide the simpler
single-element key for non-interleaved / no_pipelining paths.
"""

from __future__ import annotations

from typing import Tuple


def make_offload_key_interleaved(
    virtual_microbatch_id: int,
    model_chunk_id: int,
    forward: bool = True,
) -> Tuple[int, int, int]:
    """Return ``(group_id, model_chunk_id, mb_in_pp)``.

    ``group_id = virtual_microbatch_id // (PP * num_model_chunks)``
    rolls over once every wave of ``PP * num_model_chunks`` microbatches.
    Within a wave, ``(model_chunk_id, mb_in_pp)`` uniquely identifies
    the forward / backward pair on this rank, so forward and backward
    on the same wave share a key while different waves never collide.

    ``forward`` is advisory — kept on the signature so call sites can
    self-document direction; the returned key is the same regardless.
    """
    del forward  # advisory only
    from megatron.core import parallel_state as ps

    pp = ps.get_pipeline_model_parallel_world_size()
    chunks = ps.get_virtual_pipeline_model_parallel_world_size() or 1
    wave = pp * chunks
    group_id = virtual_microbatch_id // wave
    mb_in_pp = virtual_microbatch_id % pp
    return (group_id, model_chunk_id, mb_in_pp)


__all__ = ["make_offload_key_interleaved"]
