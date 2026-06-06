"""Tests for observability state + reporting (commit 7).

CPU-only.  Exercises:

* ``record_microbatch_offload`` accumulates total + interval counters.
* ``record_degradation_warning`` increments only the degradation field.
* ``report_after_step`` ticks ``train_steps_seen`` even when reporting
  is disabled; only prints (and resets interval counters) when the step
  is interval-aligned.
* ``format_report`` produces a single-line summary with the expected
  fields.
* The interval is read from ``FlOffloadConfig`` so toggling the config
  at runtime works without restarting.
* ``OffloadAsync.__exit__`` / ``OnloadAsync.__exit__`` fire the
  degradation hook when ``issued_stages < stages`` on entry, and stay
  silent when the caller pre-issued every stage (Commit 7's pattern).
"""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from unittest import mock

from megatron.plugin.fl_offload import observability
from megatron.plugin.fl_offload.config import FlOffloadConfig, get_config, set_config


class _BaseCase(unittest.TestCase):
    def setUp(self) -> None:
        self._saved_cfg = get_config()
        set_config(FlOffloadConfig(enable=False, report_interval=10))
        observability._reset_state_for_tests()

    def tearDown(self) -> None:
        observability._reset_state_for_tests()
        set_config(self._saved_cfg)


class TestRecordMicrobatchOffload(_BaseCase):
    def test_accumulates_total_and_interval(self) -> None:
        observability.record_microbatch_offload(3, 1024)
        observability.record_microbatch_offload(2, 512)
        s = observability.get_state()
        self.assertEqual(s.total_tensors, 5)
        self.assertEqual(s.total_bytes, 1536)
        self.assertEqual(s.interval_tensors, 5)
        self.assertEqual(s.interval_bytes, 1536)

    def test_zero_input_is_noop(self) -> None:
        observability.record_microbatch_offload(0, 0)
        s = observability.get_state()
        self.assertEqual(s.total_tensors, 0)
        self.assertEqual(s.total_bytes, 0)


class TestDegradationWarning(_BaseCase):
    def test_independent_counter(self) -> None:
        observability.record_degradation_warning()
        observability.record_degradation_warning()
        s = observability.get_state()
        self.assertEqual(s.degradation_warnings, 2)
        self.assertEqual(s.total_tensors, 0)


class TestReportCadence(_BaseCase):
    def test_step_counter_advances_even_when_disabled(self) -> None:
        set_config(FlOffloadConfig(enable=False, report_interval=0))
        for _ in range(3):
            observability.report_after_step()
        self.assertEqual(observability.get_state().train_steps_seen, 3)

    def test_interval_alignment_only(self) -> None:
        set_config(FlOffloadConfig(enable=False, report_interval=2))
        buf = io.StringIO()
        with redirect_stdout(buf):
            observability.record_microbatch_offload(1, 100)
            observability.report_after_step()  # step 1: no print
            observability.report_after_step()  # step 2: print + reset
            observability.record_microbatch_offload(1, 200)
            observability.report_after_step()  # step 3: no print
        out = buf.getvalue()
        # Exactly one report line.
        self.assertEqual(out.count("[fl-offload]"), 1)
        # After the reset, interval counters started over and only the
        # post-reset record(1, 200) remains.
        s = observability.get_state()
        self.assertEqual(s.interval_tensors, 1)
        self.assertEqual(s.interval_bytes, 200)
        # Totals are cumulative regardless of the reset.
        self.assertEqual(s.total_tensors, 2)
        self.assertEqual(s.total_bytes, 300)


class TestFormatReport(_BaseCase):
    def test_contains_all_fields(self) -> None:
        s = observability.get_state()
        s.interval_tensors = 100
        s.interval_bytes = (5 << 30) + 200 * (1 << 20)  # 5.20 GiB
        s.degradation_warnings = 1
        line = observability.format_report(s, step_id=42)
        for needle in (
            "[fl-offload]",
            "step=42",
            "tensors=100",
            "act_bytes=",
            "GiB",
            "peak_pinned_MiB=",
            "groups_resident=",
            "degradation=1",
        ):
            self.assertIn(needle, line)


class TestOffloadAsyncDegradationHook(_BaseCase):
    """OffloadAsync.__exit__ fires degradation when undrained on entry."""

    def test_undrained_exit_fires_warning(self) -> None:
        # Build a minimal real ActivationGroup so OffloadAsync isn't
        # disabled. We register it under a sentinel key and exercise the
        # __exit__ path with issued_stages < stages.
        from megatron.plugin.fl_offload.group import ActivationGroup
        from megatron.plugin.fl_offload.runtime import (
            OffloadAsync,
            _reset_groups_for_tests,
            register_group,
        )

        _reset_groups_for_tests()
        key = ("test", "undrained")
        register_group(key, ActivationGroup([], key, stages=4))

        ofa = OffloadAsync(key, stages=4)
        ofa.__enter__()
        # Don't issue anything; __exit__ should hit the degradation hook.
        ofa.__exit__(None, None, None)
        _reset_groups_for_tests()

        self.assertEqual(observability.get_state().degradation_warnings, 1)

    def test_drained_exit_is_silent(self) -> None:
        from megatron.plugin.fl_offload.group import ActivationGroup
        from megatron.plugin.fl_offload.runtime import (
            OffloadAsync,
            _reset_groups_for_tests,
            register_group,
        )

        _reset_groups_for_tests()
        key = ("test", "drained")
        register_group(key, ActivationGroup([], key, stages=4))

        ofa = OffloadAsync(key, stages=4)
        ofa.__enter__()
        for s in range(ofa.stages):
            ofa.issue(s)
        ofa.__exit__(None, None, None)
        _reset_groups_for_tests()

        self.assertEqual(observability.get_state().degradation_warnings, 0)


if __name__ == "__main__":
    unittest.main()
