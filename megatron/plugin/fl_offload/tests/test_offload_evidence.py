"""Evidence tests for fl-offload: prove the runtime really offloads.

Commit 3's tests confirm functional correctness — CPU buffer bytes match
the source, and bit-exact onload restores the tensor.  They do **not**
prove that the offload had any side effect a user cares about:

* Did GPU VRAM actually drop after the offload?
* Did the D2H / H2D copies run on the dedicated copy streams instead of
  the default compute stream?
* Did ``stages=N`` really break the work into N async launches?

This file adds one targeted test per claim.  All tests require CUDA;
the module skips cleanly on CPU-only boxes.
"""

import os
import sys
import unittest

import torch


sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
    ),
)


CUDA = torch.cuda.is_available()
skip_no_cuda = unittest.skipUnless(CUDA, "CUDA accelerator not available")


if CUDA:
    from megatron.plugin.fl_offload import group as group_mod
    from megatron.plugin.fl_offload.config import (
        FlOffloadConfig,
        get_config,
        set_config,
    )
    from megatron.plugin.fl_offload.group import ActivationGroup
    from megatron.plugin.fl_offload.pool import reset_global_pool
    from megatron.plugin.fl_offload.runtime import (
        OffloadAsync,
        OnloadAsync,
        TensorWrap,
        _reset_groups_for_tests,
        register_group,
    )
    from megatron.plugin.fl_offload.stream import (
        current_stream,
        get_memcpy_stream,
        reset_streams,
    )


@skip_no_cuda
class _CudaTestBase(unittest.TestCase):
    """Shared setup for the three evidence test classes."""

    def setUp(self):
        self._orig_cfg = get_config()
        set_config(
            FlOffloadConfig(
                enable=True,
                min_bytes=0,
                non_contiguous=False,
                pin_memory=True,
                ratio=1.0,
                per_batch_size=0.0,
                stages=1,
            )
        )
        reset_global_pool()
        reset_streams()
        _reset_groups_for_tests()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    def tearDown(self):
        set_config(self._orig_cfg)
        reset_global_pool()
        reset_streams()
        _reset_groups_for_tests()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------- #
# A. VRAM truly drops after offload                                       #
# ---------------------------------------------------------------------- #
@skip_no_cuda
class TestVramDrops(_CudaTestBase):
    """``offload_epilogue`` must actually release the GPU storage.

    The trap this test catches: if anything else (e.g. an internal
    reference, a closure, a list inside the group) still pins the
    original tensor after ``tw.x = None``, the GPU storage will not be
    freed and the only "offload" we get is an extra CPU copy.
    """

    def test_memory_allocated_drops_by_tensor_size(self):
        # 8 MiB is large enough to dwarf allocator-rounding noise but
        # small enough to fit anywhere.
        n_bytes = 8 * 1024 * 1024
        baseline = torch.cuda.memory_allocated()

        # Build a tensor whose ONLY Python reference is wrap.x.
        tensor = torch.empty(n_bytes, dtype=torch.uint8, device="cuda")
        tensor.fill_(0xAB)
        wrap = TensorWrap(tensor)
        del tensor  # now only wrap.x holds the storage

        torch.cuda.synchronize()
        pre = torch.cuda.memory_allocated()
        self.assertGreaterEqual(
            pre - baseline,
            n_bytes,
            "fixture failed — tensor did not actually claim VRAM",
        )

        group = ActivationGroup([wrap], key="vram", stages=1)
        register_group("vram", group)

        with OffloadAsync("vram", stages=1):
            pass

        # ``offload_epilogue`` sets wrap.x = None on the live stream; sync
        # to make sure the allocator sees the deletion.
        torch.cuda.synchronize()

        post = torch.cuda.memory_allocated()
        freed = pre - post
        # Allow ≤ 1 MiB of allocator rounding noise.  We MUST see most of
        # the 8 MiB released.
        self.assertGreaterEqual(
            freed,
            n_bytes - 1024 * 1024,
            f"expected ~{n_bytes} bytes freed, observed only {freed}",
        )

        # wrap.x must be None — that's the GPU reference handover that
        # makes the VRAM drop possible.
        self.assertIsNone(wrap.x)


# ---------------------------------------------------------------------- #
# B. D2H / H2D run on the dedicated copy streams                          #
# ---------------------------------------------------------------------- #
@skip_no_cuda
class TestUsesDedicatedStream(_CudaTestBase):
    """The copy must be issued on ``get_memcpy_stream(...)``.

    The trap: if ``stream_scope`` ever degrades to ``nullcontext`` while
    a real stream is available (typo, refactor accident, etc.) the copy
    silently lands on the current compute stream — correctness keeps
    passing but the runtime stops giving any compute/copy overlap.
    """

    def _capture_streams_during(self, fn):
        seen = []
        original_scope = group_mod.stream_scope

        def tracking_scope(stream):
            seen.append(stream)
            return original_scope(stream)

        group_mod.stream_scope = tracking_scope
        try:
            fn()
        finally:
            group_mod.stream_scope = original_scope
        return seen

    def test_offload_runs_on_offload_stream(self):
        wraps = [TensorWrap(torch.randn(2048, device="cuda", dtype=torch.float32))]
        group = ActivationGroup(wraps, key="stream-o", stages=1)
        register_group("stream-o", group)

        def run():
            with OffloadAsync("stream-o", stages=1):
                pass

        seen = self._capture_streams_during(run)

        offload_stream = get_memcpy_stream("offload")
        cur = current_stream()
        self.assertIsNotNone(offload_stream)
        self.assertIsNotNone(cur)

        # ``stream_scope`` must have been entered at least once during
        # offload_issue, and with the offload stream.
        self.assertGreaterEqual(len(seen), 1)
        self.assertIs(seen[0], offload_stream)
        # And critically — NOT the current/compute stream.
        self.assertIsNot(seen[0], cur)

    def test_onload_runs_on_onload_stream(self):
        wraps = [TensorWrap(torch.randn(2048, device="cuda", dtype=torch.float32))]
        group = ActivationGroup(wraps, key="stream-l", stages=1)
        register_group("stream-l", group)

        # Run offload first (we don't care which stream that uses for
        # this test — covered above); we just need the activation group
        # to be in a state where onload can act on it.
        with OffloadAsync("stream-l", stages=1):
            pass
        torch.cuda.synchronize()

        def run():
            with OnloadAsync("stream-l", stages=1):
                pass

        seen = self._capture_streams_during(run)

        onload_stream = get_memcpy_stream("onload")
        cur = current_stream()
        self.assertIsNotNone(onload_stream)

        self.assertGreaterEqual(len(seen), 1)
        self.assertIs(seen[0], onload_stream)
        self.assertIsNot(seen[0], cur)

    def test_offload_and_onload_streams_are_distinct(self):
        # Sanity check — if these collapse to a single stream the two
        # tests above pass for the wrong reason.
        self.assertIsNot(
            get_memcpy_stream("offload"),
            get_memcpy_stream("onload"),
        )


# ---------------------------------------------------------------------- #
# C. stages=N really splits the work into N async launches                #
# ---------------------------------------------------------------------- #
@skip_no_cuda
class TestStagesActuallyFire(_CudaTestBase):
    """``OffloadAsync.issue(s)`` must call ``group.offload_issue`` for
    each stage exactly once, and ``stages=N`` must visit every stage.

    The trap: if ``OffloadAsync.issue`` mis-handles its ``issued_stages``
    bookkeeping (e.g. off-by-one), the runtime either drops work
    silently or duplicates copies, both of which hide behind bit-exact
    end-to-end results because ``offload_issue`` is internally
    idempotent at the buffer level.
    """

    def _instrument(self, group):
        seen = []
        original = group.offload_issue

        def counting(stage_id):
            seen.append(stage_id)
            return original(stage_id)

        group.offload_issue = counting
        return seen

    def _make_group(self, key, n_tensors, stages):
        wraps = [
            TensorWrap(torch.arange(64, dtype=torch.uint8, device="cuda"))
            for _ in range(n_tensors)
        ]
        group = ActivationGroup(wraps, key=key, stages=stages)
        register_group(key, group)
        return group

    def test_stages_1_fires_one_issue_at_exit(self):
        group = self._make_group("s1", n_tensors=4, stages=1)
        seen = self._instrument(group)

        # No explicit ctx.issue(...) — __exit__ must flush stage 0.
        with OffloadAsync("s1", stages=1):
            pass

        self.assertEqual(seen, [0])

    def test_stages_4_fires_four_distinct_issues(self):
        group = self._make_group("s4", n_tensors=4, stages=4)
        seen = self._instrument(group)

        with OffloadAsync("s4", stages=4) as ctx:
            for s in range(4):
                ctx.issue(s)

        self.assertEqual(seen, [0, 1, 2, 3])

    def test_stages_4_exit_flushes_remaining(self):
        """Issuing only some stages leaves __exit__ to finish the rest."""
        group = self._make_group("s4f", n_tensors=4, stages=4)
        seen = self._instrument(group)

        with OffloadAsync("s4f", stages=4) as ctx:
            ctx.issue(0)
            ctx.issue(1)
            # stages 2 and 3 should be flushed on __exit__.

        self.assertEqual(seen, [0, 1, 2, 3])

    def test_repeated_issue_for_same_stage_is_idempotent(self):
        group = self._make_group("idem", n_tensors=4, stages=4)
        seen = self._instrument(group)

        with OffloadAsync("idem", stages=4) as ctx:
            ctx.issue(0)
            ctx.issue(0)  # no-op
            ctx.issue(0)  # no-op
            ctx.issue(1)
            ctx.issue(0)  # no-op (issued_stages already past 0)
            # __exit__ flushes 2 and 3.

        self.assertEqual(seen, [0, 1, 2, 3])

    def test_jump_forward_issue_fills_intermediate_stages(self):
        """``ctx.issue(3)`` from scratch must run stages 0, 1, 2, 3 in order."""
        group = self._make_group("jump", n_tensors=4, stages=4)
        seen = self._instrument(group)

        with OffloadAsync("jump", stages=4) as ctx:
            ctx.issue(3)
            # __exit__ has nothing left to do.

        self.assertEqual(seen, [0, 1, 2, 3])

    def test_buckets_match_stage_count(self):
        """``stages=N`` partitions the tensors into N buckets (Commit 3
        guarantees the partition; here we re-assert it as part of the
        evidence story so a regression breaks the right test)."""
        group = self._make_group("bk", n_tensors=8, stages=4)
        # Run prologue via OffloadAsync.__enter__.
        ctx = OffloadAsync("bk", stages=4)
        ctx.__enter__()
        try:
            self.assertEqual(len(group.copy_buckets), 4)
            # 8 tensors round-robin into 4 buckets → 2 per bucket.
            for bucket in group.copy_buckets:
                self.assertEqual(len(bucket), 2)
        finally:
            ctx.__exit__(None, None, None)


if __name__ == "__main__":
    unittest.main()
