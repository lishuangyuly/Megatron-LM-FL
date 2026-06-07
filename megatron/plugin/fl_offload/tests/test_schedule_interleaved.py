"""Tests for interleaved-schedule wrapping (commit 6 + combined path).

CPU-only.  Verifies:

* ``make_offload_key_interleaved`` returns the logical
  ``(microbatch_id, model_chunk_id)`` pair, identical for forward and
  backward derivations of the same microbatch.
* ``_is_no_offload_boundary`` flags last-PP-rank × last-chunk only.
* ``_patch_step_funcs`` swaps ``forward_step`` / ``backward_step`` /
  ``combined_1f1b_schedule_for_interleaved_pipelining`` for the
  duration of the inner call and restores them on exit.
* The id-keyed registry round-trips a forward → backward call pair on
  the conventional path.
* The combined-path wrapper onloads b before the helper, records f
  during it, derives symmetric keys via FL's schedule-table lookups,
  handles f-only / b-only steps, and respects the no-offload boundary.
"""

from __future__ import annotations

import contextlib
import unittest
from unittest import mock

from megatron.plugin.fl_offload.config import FlOffloadConfig, get_config, set_config
from megatron.plugin.fl_offload.runtime import get_pipeline_offload_runtime
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
        kf = make_offload_key_interleaved(5, 1, forward=True)
        kb = make_offload_key_interleaved(5, 1, forward=False)
        self.assertEqual(kf, kb)

    def test_key_is_logical_pair(self) -> None:
        self.assertEqual(make_offload_key_interleaved(3, 0), ("ilv", 3, 0))

    def test_different_microbatches_distinct(self) -> None:
        # Regression for the wave-formula collision: in-chunk microbatch
        # ids 0 and pp must NOT collide.
        keys = {make_offload_key_interleaved(n, 0) for n in range(8)}
        self.assertEqual(len(keys), 8)

    def test_different_model_chunks_distinct(self) -> None:
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


class TestPatchStepFuncs(_BaseCase):
    def test_patch_swaps_and_restores(self) -> None:
        from megatron.core.pipeline_parallel import schedules as core_sched

        original_fwd = core_sched.forward_step
        original_bwd = core_sched.backward_step
        original_combined = core_sched.combined_1f1b_schedule_for_interleaved_pipelining

        with wrappers._patch_step_funcs():
            self.assertIsNot(core_sched.forward_step, original_fwd)
            self.assertIsNot(core_sched.backward_step, original_bwd)
            self.assertIsNot(
                core_sched.combined_1f1b_schedule_for_interleaved_pipelining,
                original_combined,
            )

        # Restored on exit.
        self.assertIs(core_sched.forward_step, original_fwd)
        self.assertIs(core_sched.backward_step, original_bwd)
        self.assertIs(
            core_sched.combined_1f1b_schedule_for_interleaved_pipelining,
            original_combined,
        )


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


# ---------------------------------------------------------------------- #
# Combined (overlap_moe_expert_parallel_comm) path                       #
# ---------------------------------------------------------------------- #
# Schedule table from FL's docstring example (PP2, 5 microbatches, VP2):
#   virtual_microbatch_id | 0 1 2 3 4 5 6 7 8 9
#   microbatch_id         | 0 1 2 0 1 2 3 4 3 4
#   model_chunk_id        | 0 0 0 1 1 1 0 0 1 1
_MB_TABLE = [0, 1, 2, 0, 1, 2, 3, 4, 3, 4]
_MC_TABLE = [0, 0, 0, 1, 1, 1, 0, 0, 1, 1]
_NUM_CHUNKS = 2


def _table_get_mb_id(iteration_id, forward):
    assert forward
    return _MB_TABLE[iteration_id]


def _table_get_mc_id(virtual_microbatch_id, forward):
    mc = _MC_TABLE[virtual_microbatch_id]
    if not forward:
        mc = _NUM_CHUNKS - mc - 1
    return mc


def _fake_combined_helper_factory(events):
    """Return a fake with FL's parameter names (for signature binding)."""

    def fake_combined_helper(
        config,
        forward_step_func,
        data_iterator,
        model,
        num_microbatches,
        forward_data_store,
        forward_step_helper_preprocess,
        forward_step_helper_postprocess,
        backward_step_helper_preprocess,
        backward_step_helper_postprocess,
        get_microbatch_id_in_model_chunk,
        get_model_chunk_id,
        check_first_val_step,
        is_first_microbatch_for_model_chunk,
        collect_non_loss_data,
        f_virtual_microbatch_id=None,
        b_virtual_microbatch_id=None,
        pre_forward=None,
        pre_backward=None,
        post_forward=None,
        post_backward=None,
    ):
        events.append("orig")
        return ("output", "grad")

    return fake_combined_helper


def _combined_call_args(f_vmb, b_vmb):
    """Positional args mirroring schedules.py's call site."""
    return (
        dict(  # kwargs
            f_virtual_microbatch_id=f_vmb,
            b_virtual_microbatch_id=b_vmb,
        ),
        (  # positional
            "config",
            "fwd_func",
            "data_iter",
            "model",
            5,
            [],
            "fsh_pre",
            "fsh_post",
            "bsh_pre",
            "bsh_post",
            _table_get_mb_id,
            _table_get_mc_id,
            "check_first_val",
            "is_first_mb",
            False,
        ),
    )


class TestCombinedHelper(_BaseCase):
    def _run(self, f_vmb, b_vmb, pp_rank=0, pp_size=2, vp_size=2):
        """Run the wrapped fake helper; return (events, fwd_calls, bwd_calls)."""
        events = []
        fwd_calls = []
        bwd_calls = []
        runtime = get_pipeline_offload_runtime()

        @contextlib.contextmanager
        def spy_forward_microbatch(**kwargs):
            fwd_calls.append(kwargs)
            events.append("fwd_enter")
            yield
            events.append("fwd_exit")

        class _SpyBwdCtx:
            def __enter__(self_inner):
                events.append("bwd_enter")
                return self_inner

            def __exit__(self_inner, *exc):
                events.append("bwd_exit")
                return False

        def spy_backward_microbatch(**kwargs):
            bwd_calls.append(kwargs)
            return _SpyBwdCtx()

        fake = _fake_combined_helper_factory(events)
        kwargs, args = _combined_call_args(f_vmb, b_vmb)
        with _patch_parallel_state(pp_rank, pp_size, vp_size):
            with mock.patch.object(
                runtime, "forward_microbatch", side_effect=spy_forward_microbatch
            ), mock.patch.object(
                runtime, "backward_microbatch", side_effect=spy_backward_microbatch
            ):
                wrapped = wrappers._make_offload_combined_helper(fake)
                result = wrapped(*args, **kwargs)
        self.assertEqual(result, ("output", "grad"))
        return events, fwd_calls, bwd_calls

    def test_steady_step_ordering_and_keys(self) -> None:
        set_config(_enabled())
        # f_vmb=3 → forward (mb 0, chunk 1).  b_vmb=0 → backward of the
        # logical microbatch (mb 0, chunk 1 - 0 - 1 = 1).  Same key.
        events, fwd_calls, bwd_calls = self._run(f_vmb=3, b_vmb=0)
        # Onload-before, record-during, offload-exit, then bwd ctx exit.
        self.assertEqual(
            events, ["bwd_enter", "fwd_enter", "orig", "fwd_exit", "bwd_exit"]
        )
        self.assertEqual(len(fwd_calls), 1)
        self.assertEqual(len(bwd_calls), 1)
        f_key = fwd_calls[0]["offload_key"]
        b_key = bwd_calls[0]["offload_key"]
        self.assertEqual(f_key, make_offload_key_interleaved(0, 1))
        self.assertEqual(f_key, b_key)

    def test_forward_only_warmup_step(self) -> None:
        set_config(_enabled())
        events, fwd_calls, bwd_calls = self._run(f_vmb=0, b_vmb=None)
        self.assertEqual(events, ["fwd_enter", "orig", "fwd_exit"])
        self.assertEqual(len(bwd_calls), 0)
        self.assertEqual(
            fwd_calls[0]["offload_key"], make_offload_key_interleaved(0, 0)
        )

    def test_backward_only_cooldown_step(self) -> None:
        set_config(_enabled())
        events, fwd_calls, bwd_calls = self._run(f_vmb=None, b_vmb=9)
        self.assertEqual(events, ["bwd_enter", "orig", "bwd_exit"])
        self.assertEqual(len(fwd_calls), 0)
        # b_vmb=9 → mb_table[9]=4, chunk = 2 - 1 - 1 = 0.
        self.assertEqual(
            bwd_calls[0]["offload_key"], make_offload_key_interleaved(4, 0)
        )

    def test_disabled_is_passthrough(self) -> None:
        set_config(_disabled())
        events, fwd_calls, bwd_calls = self._run(f_vmb=3, b_vmb=0)
        self.assertEqual(events, ["orig"])
        self.assertEqual(len(fwd_calls), 0)
        self.assertEqual(len(bwd_calls), 0)

    def test_boundary_forward_disabled_flag(self) -> None:
        set_config(_enabled())
        # pp_rank=1 of pp_size=2 (last rank); f_vmb=3 is chunk 1 = last
        # chunk of 2 → boundary → enabled=False passed to the runtime.
        events, fwd_calls, _ = self._run(f_vmb=3, b_vmb=None, pp_rank=1)
        self.assertEqual(len(fwd_calls), 1)
        self.assertFalse(fwd_calls[0]["enabled"])

    def test_key_symmetry_across_all_table_entries(self) -> None:
        """Every logical microbatch: forward key == backward key."""
        set_config(_enabled())
        # Build logical -> key maps from both directions over the table.
        fwd_keys = {}
        bwd_keys = {}
        for vmb in range(len(_MB_TABLE)):
            mb = _table_get_mb_id(vmb, True)
            fwd_keys[(mb, _table_get_mc_id(vmb, True))] = make_offload_key_interleaved(
                mb, _table_get_mc_id(vmb, True), forward=True
            )
            bwd_keys[(mb, _table_get_mc_id(vmb, False))] = make_offload_key_interleaved(
                mb, _table_get_mc_id(vmb, False), forward=False
            )
        # Identical logical id set, identical key per logical id.
        self.assertEqual(set(fwd_keys), set(bwd_keys))
        for logical, key in fwd_keys.items():
            self.assertEqual(key, bwd_keys[logical])


if __name__ == "__main__":
    unittest.main()
