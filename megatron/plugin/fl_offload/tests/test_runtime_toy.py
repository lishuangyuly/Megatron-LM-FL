"""Commit 5 end-to-end tests: saved_tensors_hooks + runtime façade.

Two layers of tests:

1. **Hook bookkeeping** (CPU-friendly): exercise ``pack_hook`` /
   ``unpack_hook`` / ``record()`` directly with synthetic tensors.  These
   prove the filter is honoured, nesting is safe, and groups are
   registered under their key.
2. **Toy autograd B-series**: build a 4-block ``nn.Linear``-stack toy
   model on CUDA, wrap a mock schedule loop in
   ``forward_microbatch`` / ``backward_microbatch``, and assert that
   gradients are bit-exact against the no-plugin baseline across
   B1–B7.  These prove the runtime façade glues hooks + ActivationGroup
   + Offload/OnloadAsync into a working microbatch lifecycle.

The hook-bookkeeping tests run unconditionally; the toy autograd tests
require CUDA and skip cleanly otherwise.
"""

import os
import sys
import types
import unittest

import torch
import torch.nn as nn


sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
    ),
)


from megatron.plugin.fl_offload import hooks as hooks_mod
from megatron.plugin.fl_offload.config import (
    FlOffloadConfig,
    get_config,
    set_config,
)
from megatron.plugin.fl_offload.hooks import (
    current_collection,
    pack_hook,
    record,
    unpack_hook,
)
from megatron.plugin.fl_offload.pool import (
    get_global_pool,
    reset_global_pool,
)
from megatron.plugin.fl_offload.runtime import (
    TensorPack,
    TensorWrap,
    _reset_groups_for_tests,
    get_group,
    get_pipeline_offload_runtime,
    has_group,
)
from megatron.plugin.fl_offload.stream import reset_streams


CUDA = torch.cuda.is_available()
skip_no_cuda = unittest.skipUnless(CUDA, "CUDA accelerator not available")


# =====================================================================
# Layer 1: hook bookkeeping  (CPU-friendly — no offload needs to fire)
# =====================================================================
class TestPackUnpackBookkeeping(unittest.TestCase):
    """``pack_hook`` / ``unpack_hook`` / ``record()`` integration."""

    def setUp(self):
        self._orig_cfg = get_config()
        # Permissive config so eligibility never depends on min_bytes.
        # Use CPU-friendly defaults so this whole class runs without GPU.
        set_config(
            FlOffloadConfig(
                enable=True,
                min_bytes=0,
                non_contiguous=False,
                pin_memory=False,
                ratio=1.0,
                per_batch_size=0.0,
                stages=1,
            )
        )
        reset_global_pool()
        reset_streams()
        _reset_groups_for_tests()
        hooks_mod._reset_state_for_tests()

    def tearDown(self):
        set_config(self._orig_cfg)
        reset_global_pool()
        reset_streams()
        _reset_groups_for_tests()
        hooks_mod._reset_state_for_tests()

    # ---- pack/unpack outside a record() ------------------------------
    def test_pack_outside_record_does_not_register(self):
        """No ``record()`` open → eligible tensors are not collected."""
        t = torch.zeros(4, device="cpu")
        self.assertIsNone(current_collection())
        pack = pack_hook(t)
        self.assertIsInstance(pack, TensorPack)
        self.assertIs(pack.get(), t)
        self.assertIsNone(current_collection())

    def test_pack_unpack_none_passthrough(self):
        self.assertIsNone(pack_hook(None))
        self.assertIsNone(unpack_hook(None))

    # ---- single record() --------------------------------------------
    def test_pack_inside_record_appends_only_eligible(self):
        """Only CUDA-eligible tensors land in the collection; CPU
        tensors are filtered by rule 1."""
        # The default permissive config still requires a CUDA device
        # (rule 1).  This test does not assert the CPU tensor lands in
        # the collection — that's the point of rule 1.  We instead
        # assert ``record`` registered an empty group under the key.
        with record(key="single", group_num=1):
            t = torch.zeros(4, device="cpu")
            pack_hook(t)
            self.assertEqual(len(current_collection()), 0)
        self.assertTrue(has_group("single"))
        g = get_group("single")
        self.assertEqual(len(g.tensors), 0)

    def test_record_registers_even_when_collection_empty(self):
        """Backward-side ``get_group(key)`` must always succeed if
        ``record(key)`` ran — even for a microbatch with no eligible
        tensors."""
        with record(key="empty", group_num=4):
            pass
        self.assertTrue(has_group("empty"))

    @skip_no_cuda
    def test_pack_inside_record_appends_eligible_cuda_tensor(self):
        with record(key="cuda-one", group_num=1):
            t = torch.zeros(64, device="cuda", dtype=torch.float32)
            pack_hook(t)
            coll = current_collection()
            self.assertEqual(len(coll), 1)
            self.assertIs(coll[0].x, t)
        g = get_group("cuda-one")
        self.assertEqual(len(g.tensors), 1)

    @skip_no_cuda
    def test_view_tensor_rejected_by_default(self):
        """B6 in miniature: non-contiguous views are dropped."""
        with record(key="view", group_num=1):
            base = torch.zeros(8, 16, device="cuda", dtype=torch.float32)
            v = base.transpose(0, 1)  # non-contig view
            pack_hook(v)
            self.assertEqual(len(current_collection()), 0)

    @skip_no_cuda
    def test_view_tensor_accepted_when_non_contiguous_enabled(self):
        set_config(
            FlOffloadConfig(
                enable=True,
                min_bytes=0,
                non_contiguous=True,
                pin_memory=False,
                ratio=1.0,
            )
        )
        with record(key="view-ok", group_num=1):
            base = torch.zeros(8, 16, device="cuda", dtype=torch.float32)
            v = base.transpose(0, 1).detach()
            pack_hook(v)
            self.assertEqual(len(current_collection()), 1)

    # ---- nested record() --------------------------------------------
    def test_nested_record_restores_outer_collection(self):
        """B-side: nested record() does not leak into the outer
        collection and both keys end up in ``_GROUPS``."""
        self.assertIsNone(current_collection())
        with record(key="outer", group_num=1):
            outer = current_collection()
            self.assertIsNotNone(outer)
            self.assertEqual(len(outer), 0)

            with record(key="inner", group_num=2):
                inner = current_collection()
                self.assertIsNot(inner, outer)
                self.assertEqual(len(inner), 0)

            # Outer must be restored (same list object).
            self.assertIs(current_collection(), outer)
        self.assertIsNone(current_collection())
        self.assertTrue(has_group("outer"))
        self.assertTrue(has_group("inner"))

    def test_exception_inside_record_still_restores_state(self):
        try:
            with record(key="boom", group_num=1):
                raise RuntimeError("forward crashed")
        except RuntimeError:
            pass
        # _OFFLOAD_TENSORS must be back to None even after an exception.
        self.assertIsNone(current_collection())
        # And the group is still registered (caller may want to clean it
        # up, but the contract is "record always registers").
        self.assertTrue(has_group("boom"))


# =====================================================================
# Layer 2: toy autograd  (CUDA required)
# =====================================================================
class _ToyBlock(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.l1 = nn.Linear(hidden, hidden)
        self.l2 = nn.Linear(hidden, hidden)

    def forward(self, x):
        return self.l2(torch.relu(self.l1(x)))


class _ToyModel(nn.Module):
    def __init__(self, n_blocks=4, hidden=1024):
        super().__init__()
        self.blocks = nn.ModuleList(
            [_ToyBlock(hidden) for _ in range(n_blocks)]
        )

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x


@skip_no_cuda
class TestRuntimeToyAutograd(unittest.TestCase):
    """B1–B7 bit-exact gradient checks on a 4-block toy model."""

    HIDDEN = 1024
    BATCH = 8
    NUM_MICROBATCHES = 4

    def setUp(self):
        self._orig_cfg = get_config()
        torch.manual_seed(0xFEEDBEEF)
        torch.cuda.manual_seed_all(0xFEEDBEEF)

        # Single model instance — we restore from state_dict between
        # baseline and plugin runs to keep parameter weights identical.
        self.model = _ToyModel(n_blocks=4, hidden=self.HIDDEN).cuda()
        self._initial_state = {
            k: v.detach().clone() for k, v in self.model.state_dict().items()
        }

        # Fixed inputs and targets per test instance.
        self.inputs = [
            torch.randn(self.BATCH, self.HIDDEN, device="cuda")
            for _ in range(self.NUM_MICROBATCHES)
        ]
        self.targets = [
            torch.randn(self.BATCH, self.HIDDEN, device="cuda")
            for _ in range(self.NUM_MICROBATCHES)
        ]

        reset_global_pool()
        reset_streams()
        _reset_groups_for_tests()
        hooks_mod._reset_state_for_tests()
        torch.cuda.synchronize()

    def tearDown(self):
        set_config(self._orig_cfg)
        reset_global_pool()
        reset_streams()
        _reset_groups_for_tests()
        hooks_mod._reset_state_for_tests()
        torch.cuda.synchronize()

    # ---- helpers ---------------------------------------------------
    def _restore_model(self):
        self.model.load_state_dict(self._initial_state)
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    def _grad_snapshot(self):
        return {
            n: p.grad.detach().clone()
            for n, p in self.model.named_parameters()
            if p.grad is not None
        }

    def _run_one_iteration(self):
        runtime = get_pipeline_offload_runtime()
        losses = []
        for mb in range(self.NUM_MICROBATCHES):
            with runtime.forward_microbatch(
                phase="toy",
                virtual_microbatch_id=mb,
                model_chunk_id=0,
            ):
                out = self.model(self.inputs[mb])
                loss = ((out - self.targets[mb]) ** 2).mean()
            losses.append(loss)

        for mb in range(self.NUM_MICROBATCHES):
            with runtime.backward_microbatch(
                phase="toy",
                virtual_microbatch_id=mb,
                model_chunk_id=0,
            ):
                losses[mb].backward()
        torch.cuda.synchronize()
        return self._grad_snapshot()

    def _baseline(self):
        set_config(FlOffloadConfig(enable=False))
        self._restore_model()
        return self._run_one_iteration()

    def _with_plugin(self, **cfg_overrides):
        cfg_kw = dict(
            enable=True,
            min_bytes=0,
            non_contiguous=False,
            pin_memory=True,
            ratio=1.0,
            per_batch_size=0.0,
            stages=1,
        )
        cfg_kw.update(cfg_overrides)
        set_config(FlOffloadConfig(**cfg_kw))
        self._restore_model()
        return self._run_one_iteration()

    def _assert_grads_equal(self, baseline, candidate):
        self.assertEqual(set(baseline.keys()), set(candidate.keys()))
        for name in baseline:
            torch.testing.assert_close(
                baseline[name],
                candidate[name],
                rtol=0,
                atol=0,
                msg=f"grad mismatch for {name}",
            )

    # ---- B-series ---------------------------------------------------
    def test_b1_plugin_disabled_is_bit_exact(self):
        baseline = self._baseline()
        # Explicitly install a config with enable=False through set_config
        # to confirm the disabled path is not just "no with block".
        no_op = self._with_plugin(enable=False)
        self._assert_grads_equal(baseline, no_op)

    def test_b2_ratio_zero_is_bit_exact(self):
        baseline = self._baseline()
        passthrough = self._with_plugin(ratio=0.0)
        self._assert_grads_equal(baseline, passthrough)

    def test_b3_ratio_one_full_offload_is_bit_exact(self):
        baseline = self._baseline()
        full = self._with_plugin(ratio=1.0)
        self._assert_grads_equal(baseline, full)

    def test_b4_partial_budget_is_bit_exact(self):
        baseline = self._baseline()
        # 0.5 MiB budget against tensors that are roughly
        # batch * hidden * 4 bytes = 8 * 1024 * 4 = 32 KiB each.
        # ~16 tensors fit; partial offload exercises the "tensor too
        # big to fit" skip path while still doing real offloads.
        partial = self._with_plugin(per_batch_size=0.5)
        self._assert_grads_equal(baseline, partial)

    def test_b5_stages_4_is_bit_exact(self):
        baseline = self._baseline()
        staged = self._with_plugin(stages=4)
        self._assert_grads_equal(baseline, staged)

    def test_b7_pool_peak_stable_across_iterations(self):
        """Run several iterations sequentially and assert the pinned
        buffer pool does not grow after warm-up.  Real Megatron training
        will see PP * unique_shapes worth of peak; here the toy schedule
        is sequential so peak == one full microbatch's worth."""
        set_config(
            FlOffloadConfig(
                enable=True, min_bytes=0, ratio=1.0, stages=1, pin_memory=True
            )
        )
        # Warm-up iter populates the pool.
        self._restore_model()
        self._run_one_iteration()
        warmup_peak = get_global_pool().peak_buffer_count()
        self.assertGreater(
            warmup_peak, 0, "fixture failed — no buffers allocated"
        )

        # Many more iterations must not grow the pool.
        for _ in range(49):
            self._restore_model()
            self._run_one_iteration()
        post_peak = get_global_pool().peak_buffer_count()
        self.assertEqual(
            post_peak,
            warmup_peak,
            f"pool grew: warmup={warmup_peak}, after-50={post_peak} "
            "(buffer is leaking or pool key is unstable)",
        )

    def test_groups_drained_after_full_iteration(self):
        """All microbatch groups must be popped by the end of one
        iteration so the next iteration starts clean."""
        self._with_plugin(ratio=1.0)
        # After _with_plugin runs an iteration, _GROUPS should be empty.
        from megatron.plugin.fl_offload.runtime import _GROUPS

        self.assertEqual(len(_GROUPS), 0)


if __name__ == "__main__":
    unittest.main()
