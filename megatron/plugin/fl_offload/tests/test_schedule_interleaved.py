"""Tests for interleaved-schedule wrapping (commit 6).

CPU-only.  Verifies:

* ``make_offload_key_interleaved`` produces consistent keys for the
  forward / backward pair of one microbatch; different waves yield
  different ``group_id``; ``forward`` arg is advisory.
* ``_is_no_offload_boundary`` flags last-PP-rank × last-chunk only.
* ``wrap_schedule_for_offload`` swaps ``forward_step`` / ``backward_step``
  for the duration of the inner call and restores them on exit.
* The id-keyed registry round-trips a forward → backward call pair so
  ``offload_backward_step`` sees the matching key.
* With ``enable=False`` the runtime context is ``nullcontext`` and no
  group / no key is registered.
"""

from __future__ import annotations

import unittest
from unittest import mock

from megatron.plugin.fl_offload.config import FlOffloadConfig, get_config, set_config
from megatron.plugin.fl_offload.schedules import wrappers
from megatron.plugin.fl_offload.schedules.keys import make_offload_key_interleaved


def _enabled() -> FlOffloadConfig:
    return FlOffloadConfig(enable=True, ratio=1.0)


def _disabled() -> FlOffloadConfig:
    return FlOffloadConfig(enable=False)


class _BaseCase(unittest.TestCase):
    def setUp(self) -> None:
        self._saved_cfg = get_config()
        set_config(_disabled())
        wrappers._reset_registry_for_tests()

    def tearDown(self) -> None:
        wrappers._reset_registry_for_tests()
        set_config(self._saved_cfg)


def _patch_parallel_state(pp_rank: int, pp_size: int, vp_size: int):
    """Context: stub parallel_state's three getters with constants."""
    from megatron.core import parallel_state as ps

    return mock.patch.multiple(
        ps,
        get_pipeline_model_parallel_rank=mock.MagicMock(return_value=pp_rank),
        get_pipeline_model_parallel_world_size=mock.MagicMock(return_value=pp_size),
        get_virtual_pipeline_model_parallel_world_size=mock.MagicMock(return_value=vp_size),
    )


class TestKeyDerivation(_BaseCase):
    def test_forward_backward_keys_match(self) -> None:
        with _patch_parallel_state(pp_rank=0, pp_size=4, vp_size=2):
            kf = make_offload_key_interleaved(5, 1, forward=True)
            kb = make_offload_key_interleaved(5, 1, forward=False)
        self.assertEqual(kf, kb)

    def test_wave_boundary_changes_group_id(self) -> None:
        # PP=4, chunks=2 → wave=8.  vmb=0 → group 0; vmb=8 → group 1.
        with _patch_parallel_state(pp_rank=0, pp_size=4, vp_size=2):
            k0 = make_offload_key_interleaved(0, 0)
            k_wave = make_offload_key_interleaved(8, 0)
        self.assertNotEqual(k0[0], k_wave[0])
        # mb_in_pp is vmb % pp; vmb=0 and vmb=8 both → 0.
        self.assertEqual(k0[2], k_wave[2])

    def test_mb_in_pp_modulo(self) -> None:
        with _patch_parallel_state(pp_rank=0, pp_size=4, vp_size=2):
            keys = [make_offload_key_interleaved(i, 0) for i in range(4)]
        self.assertEqual([k[2] for k in keys], [0, 1, 2, 3])

    def test_different_model_chunks_distinct(self) -> None:
        with _patch_parallel_state(pp_rank=0, pp_size=2, vp_size=2):
            ka = make_offload_key_interleaved(0, 0)
            kb = make_offload_key_interleaved(0, 1)
        self.assertNotEqual(ka, kb)


class TestNoOffloadBoundary(_BaseCase):
    def test_last_pp_rank_last_chunk_true(self) -> None:
        with _patch_parallel_state(pp_rank=3, pp_size=4, vp_size=2):
            self.assertTrue(wrappers._is_no_offload_boundary(1))

    def test_last_pp_rank_but_not_last_chunk_false(self) -> None:
        with _patch_parallel_state(pp_rank=3, pp_size=4, vp_size=2):
            self.assertFalse(wrappers._is_no_offload_boundary(0))

    def test_not_last_pp_rank_false(self) -> None:
        with _patch_parallel_state(pp_rank=0, pp_size=4, vp_size=2):
            self.assertFalse(wrappers._is_no_offload_boundary(1))

    def test_no_vpp_single_chunk_path(self) -> None:
        # vp_size=1 means num_chunks=1; chunk_id 0 is also "last".
        with _patch_parallel_state(pp_rank=1, pp_size=2, vp_size=1):
            self.assertTrue(wrappers._is_no_offload_boundary(0))


class _FakeSched:
    """A stand-in for ``megatron.core.pipeline_parallel.schedules``.

    Only carries the two attributes we patch (``forward_step`` /
    ``backward_step``).  Tests reach into it to confirm the swap
    actually happened.
    """

    def __init__(self) -> None:
        self.forward_step = lambda *a, **kw: ("output", 1)
        self.backward_step = lambda *a, **kw: "grad"


class TestPatchStepFuncs(_BaseCase):
    def test_patch_swaps_and_restores(self) -> None:
        from megatron.core.pipeline_parallel import schedules as core_sched

        original_fwd = core_sched.forward_step
        original_bwd = core_sched.backward_step

        with wrappers._patch_step_funcs():
            self.assertIsNot(core_sched.forward_step, original_fwd)
            self.assertIsNot(core_sched.backward_step, original_bwd)

        # Restored on exit.
        self.assertIs(core_sched.forward_step, original_fwd)
        self.assertIs(core_sched.backward_step, original_bwd)


class TestForwardBackwardKeyRoundTrip(_BaseCase):
    """Verify the id-keyed registry hands forward's key to backward."""

    def test_registry_round_trip_when_enabled(self) -> None:
        set_config(_enabled())

        sentinel_output = object()  # any non-None object with stable id()
        captured_call = {}

        def orig_forward_step(*args, **kwargs):
            return (sentinel_output, 7)

        def orig_backward_step(*args, **kwargs):
            captured_call["args"] = args
            return "grad"

        with _patch_parallel_state(pp_rank=0, pp_size=2, vp_size=2):
            patched_fwd = wrappers._make_offload_forward_step(orig_forward_step)
            patched_bwd = wrappers._make_offload_backward_step(orig_backward_step)

            # Forward populates the registry.
            out, ntok = patched_fwd(current_microbatch=3, vp_stage=0)
            self.assertIs(out, sentinel_output)
            self.assertEqual(ntok, 7)
            self.assertIn(id(sentinel_output), wrappers._REGISTRY.by_output_id)

            expected_key = make_offload_key_interleaved(3, 0)
            self.assertEqual(
                wrappers._REGISTRY.by_output_id[id(sentinel_output)], expected_key
            )

            # Backward consumes it.
            _ = patched_bwd(None, sentinel_output, None, None)
            self.assertNotIn(id(sentinel_output), wrappers._REGISTRY.by_output_id)

    def test_no_registry_when_disabled(self) -> None:
        set_config(_disabled())

        sentinel_output = object()

        def orig_forward_step(*args, **kwargs):
            return (sentinel_output, 1)

        with _patch_parallel_state(pp_rank=0, pp_size=2, vp_size=2):
            patched_fwd = wrappers._make_offload_forward_step(orig_forward_step)
            patched_fwd(current_microbatch=0, vp_stage=0)

        # Nothing should be registered when the runtime is disabled
        # (the runtime façade returns nullcontext and the wrapper
        # short-circuits the id-tracking path).
        self.assertNotIn(id(sentinel_output), wrappers._REGISTRY.by_output_id)

    def test_no_registry_at_no_offload_boundary(self) -> None:
        set_config(_enabled())

        sentinel_output = object()

        def orig_forward_step(*args, **kwargs):
            return (sentinel_output, 1)

        # Last PP rank × last chunk → boundary → enabled=False at the
        # call site, so we still skip the registry path.
        with _patch_parallel_state(pp_rank=1, pp_size=2, vp_size=2):
            patched_fwd = wrappers._make_offload_forward_step(orig_forward_step)
            patched_fwd(current_microbatch=0, vp_stage=1)

        self.assertNotIn(id(sentinel_output), wrappers._REGISTRY.by_output_id)


class TestWrapScheduleForOffload(_BaseCase):
    def test_inner_call_sees_patched_step_funcs(self) -> None:
        from megatron.core.pipeline_parallel import schedules as core_sched

        original_fwd = core_sched.forward_step
        captured = {}

        def inner(*args, **kwargs):
            captured["fwd_inside"] = core_sched.forward_step
            captured["original"] = original_fwd
            return "ok"

        wrapped = wrappers.wrap_schedule_for_offload(inner)
        result = wrapped()
        self.assertEqual(result, "ok")
        # Inside the inner call, forward_step was swapped.
        self.assertIsNot(captured["fwd_inside"], captured["original"])
        # After the call, forward_step is the original again.
        self.assertIs(core_sched.forward_step, original_fwd)


if __name__ == "__main__":
    unittest.main()
