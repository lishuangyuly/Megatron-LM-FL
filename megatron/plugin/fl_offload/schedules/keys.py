"""Offload-key derivation helpers (commit 5 / 6 placeholder).

Will host ``make_offload_key_no_interleave`` (commit 5) and
``make_offload_key_interleaved`` (commit 6) so that an activation group
stored during a microbatch's forward step is reliably found by the matching
backward step on the same pipeline rank.

Empty in commit 1.
"""
