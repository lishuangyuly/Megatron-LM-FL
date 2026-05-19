"""Unit tests for ``ActivationGroup`` + ``OffloadAsync`` / ``OnloadAsync``.

These tests need CUDA for the bit-exact full-link assertions, so the
whole module skips on CPU-only boxes.  Sort / budget / bucket logic is
exercised first against synthetic ``TensorWrap`` instances so any
regressions are diagnosed before the heavier copy paths run.
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


if CUDA:  # heavy imports gated to avoid spurious failures on CPU CI
    from megatron.plugin.fl_offload.config import (
        FlOffloadConfig,
        get_config,
        set_config,
    )
    from megatron.plugin.fl_offload.group import (
        ActivationGroup,
        CopyTaskGroup,
    )
    from megatron.plugin.fl_offload.pool import reset_global_pool
    from megatron.plugin.fl_offload.runtime import (
        OffloadAsync,
        OnloadAsync,
        TensorWrap,
        _reset_groups_for_tests,
        has_group,
        register_group,
    )
    from megatron.plugin.fl_offload.stream import reset_streams


@skip_no_cuda
class _GroupTestBase(unittest.TestCase):
    def setUp(self):
        self._orig_cfg = get_config()
        # Loosen min_bytes to 0 so synthetic small tensors aren't filtered.
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

    def tearDown(self):
        set_config(self._orig_cfg)
        reset_global_pool()
        reset_streams()
        _reset_groups_for_tests()


@skip_no_cuda
class TestCopyTaskGroup(unittest.TestCase):
    def test_round_robin_distribution(self):
        wraps = [TensorWrap(torch.zeros(1, device="cuda")) for _ in range(4)]
        ctg = CopyTaskGroup(stages=2)
        for idx, tw in enumerate(wraps):
            ctg.add(idx, tw)
        buckets = ctg.get_buckets()
        self.assertEqual(len(buckets), 2)
        self.assertEqual([id(x) for x in buckets[0]],
                         [id(wraps[0]), id(wraps[2])])
        self.assertEqual([id(x) for x in buckets[1]],
                         [id(wraps[1]), id(wraps[3])])

    def test_stages_floor_to_one(self):
        ctg = CopyTaskGroup(stages=0)
        self.assertEqual(len(ctg.get_buckets()), 1)


@skip_no_cuda
class TestActivationGroupSort(_GroupTestBase):
    def test_contiguous_and_large_first(self):
        # Mix one non-contiguous + three contiguous, varying sizes.
        a = torch.zeros(32, device="cuda")            # contig, 128B
        b = torch.zeros(16, device="cuda")            # contig, 64B
        c = torch.zeros(8, 4, device="cuda").transpose(0, 1)  # non-contig, 128B
        d = torch.zeros(64, device="cuda")            # contig, 256B

        wraps = [TensorWrap(t) for t in [a, b, c, d]]
        group = ActivationGroup(list(wraps), key="sort-test", stages=1)

        ordered = [tw.x for tw in group.tensors]
        # Contiguous ones come first, by descending numel.
        self.assertIs(ordered[0], d)
        self.assertIs(ordered[1], a)
        self.assertIs(ordered[2], b)
        # Non-contig last.
        self.assertIs(ordered[3], c)


@skip_no_cuda
class TestActivationGroupBudget(_GroupTestBase):
    def _build(self, sizes, **cfg_overrides):
        cfg = FlOffloadConfig(
            enable=True, min_bytes=0, non_contiguous=False, pin_memory=True,
            ratio=1.0, per_batch_size=0.0, stages=1,
        )
        for k, v in cfg_overrides.items():
            setattr(cfg, k, v)
        set_config(cfg)
        # All tensors are uint8 so nbytes == numel.
        wraps = [TensorWrap(torch.arange(n, dtype=torch.uint8, device="cuda")) for n in sizes]
        return wraps

    def test_per_batch_size_zero_uses_ratio(self):
        wraps = self._build([100, 200, 300], ratio=1.0)
        group = ActivationGroup(list(wraps), key="b-r1", stages=1)
        register_group("b-r1", group)
        with OffloadAsync("b-r1", stages=1):
            pass
        self.assertEqual(len(group.offloaded_tensors), 3)

    def test_ratio_half_caps_bytes(self):
        wraps = self._build([100, 200, 300], ratio=0.5)
        group = ActivationGroup(list(wraps), key="b-r05", stages=1)
        register_group("b-r05", group)
        with OffloadAsync("b-r05", stages=1):
            pass
        # 300 + 200 = 500 > ceil(600 * 0.5) = 300; only 300 (largest) fits.
        offloaded_sizes = sorted(tw.shape[0] for tw in group.offloaded_tensors)
        self.assertEqual(offloaded_sizes, [300])

    def test_per_batch_size_overrides_ratio(self):
        # ratio=0 would have rejected everything, but per_batch_size > 0
        # should win.  per_batch_size is in MiB, so use 1 MiB to cover all.
        wraps = self._build([100, 200], ratio=0.0, per_batch_size=1.0)
        group = ActivationGroup(list(wraps), key="b-mib", stages=1)
        register_group("b-mib", group)
        with OffloadAsync("b-mib", stages=1):
            pass
        self.assertEqual(len(group.offloaded_tensors), 2)

    def test_budget_skips_tensors_that_dont_fit(self):
        # Budget of exactly 300 bytes — fits the 300-tensor only.  100 + 200
        # = 300 also fits but the sort puts 300 first.
        wraps = self._build([100, 200, 300], per_batch_size=300 / (2 ** 20))
        group = ActivationGroup(list(wraps), key="b-cut", stages=1)
        register_group("b-cut", group)
        with OffloadAsync("b-cut", stages=1):
            pass
        offloaded_sizes = sorted(tw.shape[0] for tw in group.offloaded_tensors)
        # 300 fits (300 == budget); 200 doesn't (300 + 200 > 300); 100 fits
        # (300 + 100 ≤ 300?  300 + 100 = 400 > 300, no).  So only 300.
        self.assertEqual(offloaded_sizes, [300])


@skip_no_cuda
class TestActivationGroupBuckets(_GroupTestBase):
    def test_round_robin_across_stages(self):
        sizes = [16, 32, 48, 64]
        wraps = [TensorWrap(torch.arange(n, dtype=torch.uint8, device="cuda")) for n in sizes]
        group = ActivationGroup(list(wraps), key="bucket", stages=2)
        register_group("bucket", group)
        with OffloadAsync("bucket", stages=2):
            pass

        # All four tensors were eligible (default ratio=1.0, min_bytes=0).
        self.assertEqual(len(group.offloaded_tensors), 4)
        # Stages == 2 → 2 buckets of size 2.
        self.assertEqual(len(group.copy_buckets), 2)
        self.assertEqual(len(group.copy_buckets[0]), 2)
        self.assertEqual(len(group.copy_buckets[1]), 2)


@skip_no_cuda
class TestOffloadIssueIdempotent(_GroupTestBase):
    def test_repeated_issue_does_not_recopy(self):
        # Build group, then patch its offload_issue to count calls.
        wraps = [TensorWrap(torch.arange(64, dtype=torch.uint8, device="cuda"))]
        group = ActivationGroup(list(wraps), key="idem", stages=1)
        register_group("idem", group)

        call_count = {"n": 0}
        original_issue = group.offload_issue

        def counting_issue(stage_id):
            call_count["n"] += 1
            return original_issue(stage_id)

        group.offload_issue = counting_issue

        ctx = OffloadAsync("idem", stages=1)
        ctx.__enter__()
        try:
            ctx.issue(0)
            ctx.issue(0)
            ctx.issue(0)
        finally:
            ctx.__exit__(None, None, None)

        # Only one actual underlying issue call, even though we requested
        # stage 0 three times and then __exit__ also tried to flush.
        self.assertEqual(call_count["n"], 1)


@skip_no_cuda
class TestRoundTripBitExact(_GroupTestBase):
    def _run_round_trip(self, original_tensors, key, stages):
        wraps = [TensorWrap(t) for t in original_tensors]
        group = ActivationGroup(list(wraps), key=key, stages=stages)
        register_group(key, group)
        snapshots = [t.clone() for t in original_tensors]

        with OffloadAsync(key, stages=stages):
            pass
        # ``non_blocking=True`` D2H requires a host-side sync before we
        # peek at the pinned buffer from Python.
        torch.cuda.synchronize()

        # After offload, GPU references are dropped.
        for tw in wraps:
            self.assertIsNone(tw.x)
        # CPU buffers should match the snapshots byte-for-byte.
        for snap, tw in zip(snapshots, wraps):
            cpu_view = tw.cpu_buffer.view(snap.dtype)[: snap.numel()].reshape(snap.shape)
            torch.testing.assert_close(cpu_view, snap.cpu())

        # The group still lives in ``_GROUPS`` (only OnloadAsync pops it).
        self.assertTrue(has_group(key))

        with OnloadAsync(key, stages=stages):
            pass
        torch.cuda.synchronize()

        for snap, tw in zip(snapshots, wraps):
            self.assertIsNotNone(tw.x)
            torch.testing.assert_close(tw.x, snap.to(tw.x.device))

        # After OnloadAsync.__exit__, the group must be popped.
        self.assertFalse(has_group(key))

    def test_round_trip_stages_1(self):
        ts = [
            torch.randn(64, device="cuda", dtype=torch.float32),
            torch.randn(128, device="cuda", dtype=torch.float32),
        ]
        self._run_round_trip(ts, key="rt-1", stages=1)

    def test_round_trip_stages_4(self):
        ts = [
            torch.randn(64, device="cuda", dtype=torch.float32),
            torch.randn(128, device="cuda", dtype=torch.float32),
            torch.randn(96, device="cuda", dtype=torch.float32),
            torch.randn(32, device="cuda", dtype=torch.float32),
        ]
        self._run_round_trip(ts, key="rt-4", stages=4)


@skip_no_cuda
class TestOnloadEpiloguePopsGroup(_GroupTestBase):
    def test_group_popped_on_clean_exit(self):
        wraps = [TensorWrap(torch.arange(64, dtype=torch.uint8, device="cuda"))]
        group = ActivationGroup(list(wraps), key="popme", stages=1)
        register_group("popme", group)
        with OffloadAsync("popme", stages=1):
            pass
        self.assertTrue(has_group("popme"))
        with OnloadAsync("popme", stages=1):
            pass
        self.assertFalse(has_group("popme"))


@skip_no_cuda
class TestUnregisteredKeyIsNoOp(_GroupTestBase):
    def test_offload_async_no_group_is_safe(self):
        # No register_group call.  Both contexts should silently degrade.
        with OffloadAsync("missing", stages=2) as ctx:
            ctx.issue(0)
        with OnloadAsync("missing", stages=2) as ctx:
            ctx.issue(0)


if __name__ == "__main__":
    unittest.main()
