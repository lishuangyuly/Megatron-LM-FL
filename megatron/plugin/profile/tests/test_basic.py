"""Smoke tests for megatron.plugin.profile.

Runs on CPU-only; no CUDA required. Verifies:
  * default-disabled state and toggle semantics
  * make_name format
  * profile_it adds mcfl: events only when enabled
  * BWDRecord{Start,End} preserve numerics bit-exact in both states
  * backward-side mcfl: events appear in trace when enabled
"""

import os
import sys
import unittest
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

# Make the repo root importable so ``from megatron.plugin.profile ...`` resolves.
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from megatron.plugin.profile import (  # noqa: E402
    PREFIX,
    bwd_record_pair,
    is_profile_enabled,
    make_name,
    profile_it,
    profile_it_with_anchor,
    set_profile_enabled,
    step_record_function,
)


def _events_with(prof_obj, substring):
    """Return list of events whose name contains both PREFIX and substring."""
    return [
        e for e in prof_obj.events()
        if PREFIX in e.name and substring in e.name
    ]


class TestMakeName(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(make_name(func="foo"), "mcfl: func=foo")

    def test_multi_field(self):
        self.assertEqual(
            make_name(func="foo", phase="forward"),
            "mcfl: func=foo&phase=forward",
        )

    def test_with_ids(self):
        n = make_name(
            func="forward_backward_step",
            f_virtual_microbatch_id=3,
            b_virtual_microbatch_id=0,
        )
        self.assertEqual(
            n,
            "mcfl: func=forward_backward_step"
            "&f_virtual_microbatch_id=3&b_virtual_microbatch_id=0",
        )

    def test_empty(self):
        self.assertEqual(make_name(), "mcfl:")


class TestEnableDisable(unittest.TestCase):
    def setUp(self):
        set_profile_enabled(False)

    def tearDown(self):
        set_profile_enabled(False)

    def test_default_disabled(self):
        self.assertFalse(is_profile_enabled())

    def test_toggle(self):
        set_profile_enabled(True)
        self.assertTrue(is_profile_enabled())
        set_profile_enabled(False)
        self.assertFalse(is_profile_enabled())


class TestProfileIt(unittest.TestCase):
    def setUp(self):
        set_profile_enabled(False)

    def tearDown(self):
        set_profile_enabled(False)

    def test_call_correctness_disabled(self):
        @profile_it("mcfl: func=test_dis")
        def f(x):
            return x * 2
        self.assertEqual(f(3), 6)

    def test_call_correctness_enabled(self):
        set_profile_enabled(True)
        @profile_it("mcfl: func=test_en")
        def f(x):
            return x * 3
        self.assertEqual(f(4), 12)

    def test_no_event_when_disabled(self):
        @profile_it("mcfl: func=should_not_appear")
        def f(x):
            return x + 1
        with profile(activities=[ProfilerActivity.CPU]) as p:
            f(torch.tensor(1.0))
        self.assertEqual(len(_events_with(p, "should_not_appear")), 0)

    def test_event_appears_when_enabled(self):
        set_profile_enabled(True)
        @profile_it("mcfl: func=should_appear")
        def f(x):
            return x + 1
        with profile(activities=[ProfilerActivity.CPU]) as p:
            f(torch.tensor(1.0))
        self.assertGreater(len(_events_with(p, "should_appear")), 0)


class TestProfileItWithAnchor(unittest.TestCase):
    def setUp(self):
        set_profile_enabled(True)

    def tearDown(self):
        set_profile_enabled(False)

    def test_event_appears(self):
        @profile_it_with_anchor("mcfl: func=anchor_test")
        def f(x):
            return x + 1
        with profile(activities=[ProfilerActivity.CPU]) as p:
            f(torch.tensor(1.0))
        self.assertGreater(len(_events_with(p, "anchor_test")), 0)


class TestStepRecordFunction(unittest.TestCase):
    def setUp(self):
        set_profile_enabled(True)

    def tearDown(self):
        set_profile_enabled(False)

    def test_kwargs_propagate(self):
        with profile(activities=[ProfilerActivity.CPU]) as p:
            with step_record_function(
                func="step_t",
                f_virtual_microbatch_id=1,
                b_virtual_microbatch_id=0,
            ):
                _ = torch.tensor(1.0) + 1
        self.assertGreater(len(_events_with(p, "step_t")), 0)
        self.assertGreater(
            len(_events_with(p, "f_virtual_microbatch_id=1")), 0
        )

    def test_disabled_is_no_op(self):
        set_profile_enabled(False)
        with profile(activities=[ProfilerActivity.CPU]) as p:
            with step_record_function(func="should_not_be_there"):
                _ = torch.tensor(1.0) + 1
        self.assertEqual(len(_events_with(p, "should_not_be_there")), 0)


class TestBwdRecordPairNumerical(unittest.TestCase):
    """BWDRecord{Start,End} must not change numerics in either state."""

    def _compute(self, use_pair):
        torch.manual_seed(42)
        x = torch.randn(8, 8, requires_grad=True)
        if use_pair:
            start, end = bwd_record_pair()
            x_in = start(x)
            y = (x_in * x_in).sum()
            y = end(y, "mcfl: phase=backward&func=sq_sum")
        else:
            y = (x * x).sum()
        y.backward()
        return y.detach().clone(), x.grad.detach().clone()

    def test_match_when_disabled(self):
        set_profile_enabled(False)
        y_bare, g_bare = self._compute(use_pair=False)
        y_inst, g_inst = self._compute(use_pair=True)
        torch.testing.assert_close(y_bare, y_inst, rtol=0, atol=0)
        torch.testing.assert_close(g_bare, g_inst, rtol=0, atol=0)

    def test_match_when_enabled(self):
        set_profile_enabled(True)
        try:
            y_bare, g_bare = self._compute(use_pair=False)
            y_inst, g_inst = self._compute(use_pair=True)
            torch.testing.assert_close(y_bare, y_inst, rtol=0, atol=0)
            torch.testing.assert_close(g_bare, g_inst, rtol=0, atol=0)
        finally:
            set_profile_enabled(False)

    def test_multi_tensor_inputs(self):
        set_profile_enabled(True)
        try:
            torch.manual_seed(0)
            a = torch.randn(4, requires_grad=True)
            b = torch.randn(4, requires_grad=True)
            start, end = bwd_record_pair()
            a2, b2 = start(a, b)
            out = (a2 * b2).sum()
            out = end(out, "mcfl: phase=backward&func=dotmul")
            out.backward()
            # gradients should match the un-instrumented version
            torch.manual_seed(0)
            a_ref = torch.randn(4, requires_grad=True)
            b_ref = torch.randn(4, requires_grad=True)
            out_ref = (a_ref * b_ref).sum()
            out_ref.backward()
            torch.testing.assert_close(a.grad, a_ref.grad, rtol=0, atol=0)
            torch.testing.assert_close(b.grad, b_ref.grad, rtol=0, atol=0)
        finally:
            set_profile_enabled(False)


class TestBwdRecordPairTrace(unittest.TestCase):
    def test_backward_event_present_when_enabled(self):
        set_profile_enabled(True)
        try:
            x = torch.randn(4, requires_grad=True)
            start, end = bwd_record_pair()
            with profile(activities=[ProfilerActivity.CPU]) as p:
                x_in = start(x)
                y = (x_in * x_in).sum()
                y = end(y, "mcfl: phase=backward&func=test_bwd_event")
                y.backward()
            self.assertGreater(
                len(_events_with(p, "test_bwd_event")), 0
            )
        finally:
            set_profile_enabled(False)

    def test_no_backward_event_when_disabled(self):
        set_profile_enabled(False)
        x = torch.randn(4, requires_grad=True)
        start, end = bwd_record_pair()
        with profile(activities=[ProfilerActivity.CPU]) as p:
            x_in = start(x)
            y = (x_in * x_in).sum()
            y = end(y, "mcfl: phase=backward&func=must_be_absent")
            y.backward()
        self.assertEqual(
            len(_events_with(p, "must_be_absent")), 0
        )


if __name__ == "__main__":
    unittest.main()
