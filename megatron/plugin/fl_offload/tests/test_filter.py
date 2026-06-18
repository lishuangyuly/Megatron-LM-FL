"""Unit tests for ``is_tensor_eligible``.

CPU-only tests are run unconditionally; rules that need an actual CUDA
tensor (rules 2 / 3 / 4 / 5 reuse CUDA storage in their assertions) are
skipped when the accelerator is unavailable.
"""

import os
import sys
import types
import unittest

import torch


sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
    ),
)


from megatron.plugin.fl_offload.filter import is_tensor_eligible


CUDA = torch.cuda.is_available()
skip_no_cuda = unittest.skipUnless(CUDA, "CUDA accelerator not available")


def _cfg(**kw):
    base = dict(min_bytes=64, non_contiguous=False)
    base.update(kw)
    return types.SimpleNamespace(**base)


class TestRule1NotTensorNotCudaNotParameter(unittest.TestCase):
    def test_non_tensor_rejected(self):
        self.assertFalse(is_tensor_eligible(123, _cfg()))
        self.assertFalse(is_tensor_eligible("nope", _cfg()))
        self.assertFalse(is_tensor_eligible(None, _cfg()))

    def test_cpu_tensor_rejected(self):
        t = torch.zeros(64, dtype=torch.float32)  # 256 bytes, CPU
        self.assertFalse(is_tensor_eligible(t, _cfg()))

    @skip_no_cuda
    def test_parameter_rejected(self):
        p = torch.nn.Parameter(torch.zeros(64, dtype=torch.float32, device="cuda"))
        self.assertFalse(is_tensor_eligible(p, _cfg()))


@skip_no_cuda
class TestRule2LeafRequiresGrad(unittest.TestCase):
    def test_leaf_requires_grad_rejected(self):
        t = torch.zeros(64, dtype=torch.float32, device="cuda", requires_grad=True)
        self.assertTrue(t.is_leaf)
        self.assertFalse(is_tensor_eligible(t, _cfg()))

    def test_non_leaf_eligible(self):
        # An activation derived from a parameter is not a leaf.
        p = torch.nn.Parameter(torch.zeros(64, dtype=torch.float32, device="cuda"))
        act = p * 2  # 256 bytes, non-leaf
        self.assertFalse(act.is_leaf)
        # 256 bytes > min_bytes=64 → eligible.
        self.assertTrue(is_tensor_eligible(act, _cfg()))


@skip_no_cuda
class TestRule3Contiguity(unittest.TestCase):
    def test_view_rejected_by_default(self):
        base = torch.zeros(8, 16, dtype=torch.float32, device="cuda")
        v = base.transpose(0, 1)  # non-contiguous view
        self.assertFalse(v.is_contiguous())
        self.assertFalse(is_tensor_eligible(v, _cfg()))

    def test_view_accepted_when_non_contiguous_flag_set(self):
        base = torch.zeros(8, 16, dtype=torch.float32, device="cuda")
        v = base.transpose(0, 1).detach()  # non-contiguous, no autograd
        # 8 * 16 * 4 = 512 bytes > 64.
        self.assertTrue(is_tensor_eligible(v, _cfg(non_contiguous=True)))

    def test_storage_offset_rejected_by_default(self):
        base = torch.zeros(128, dtype=torch.float32, device="cuda")
        v = base[16:].detach()  # storage_offset == 16 (in elements)
        # contiguous but offset-non-zero → rule 3 rejects it.
        self.assertFalse(is_tensor_eligible(v, _cfg()))


@skip_no_cuda
class TestRule4MinBytes(unittest.TestCase):
    def test_below_threshold_rejected(self):
        # 16 floats = 64 bytes, threshold 65 → rejected.
        t = torch.zeros(16, dtype=torch.float32, device="cuda")
        cfg = _cfg(min_bytes=65)
        self.assertFalse(is_tensor_eligible(t.detach(), cfg))

    def test_at_threshold_accepted(self):
        t = torch.zeros(16, dtype=torch.float32, device="cuda")
        cfg = _cfg(min_bytes=64)
        self.assertTrue(is_tensor_eligible(t.detach(), cfg))


@skip_no_cuda
class TestRule5BroadcastMaskShape(unittest.TestCase):
    def test_n_1_1_k_shape_rejected(self):
        t = torch.zeros(4, 1, 1, 32, dtype=torch.float32, device="cuda")
        self.assertFalse(is_tensor_eligible(t.detach(), _cfg(min_bytes=1)))

    def test_n_2_1_k_accepted(self):
        # shape[1] != 1 → not the heuristic, should pass.
        t = torch.zeros(4, 2, 1, 32, dtype=torch.float32, device="cuda")
        self.assertTrue(is_tensor_eligible(t.detach(), _cfg(min_bytes=1)))


@skip_no_cuda
class TestExplicitPackBypassesShapeRules(unittest.TestCase):
    """commit 7.2: explicit pack (patched Function) skips Rules 3 & 5.

    A producer that hands us a tensor by name has already decided it is
    worth offloading; non-contiguous views (e.g. MoE permuted tokens) and
    (N,1,1,K) shapes must still be collected.  Rules 1/2/4 still apply.
    """

    def test_noncontiguous_view_accepted_when_explicit(self):
        base = torch.zeros(8, 16, dtype=torch.float32, device="cuda")
        v = base.transpose(0, 1).detach()  # non-contiguous view, 512B
        self.assertFalse(v.is_contiguous())
        # hooks path (explicit=False) rejects it...
        self.assertFalse(is_tensor_eligible(v, _cfg(min_bytes=1)))
        # ...explicit-pack path accepts it.
        self.assertTrue(is_tensor_eligible(v, _cfg(min_bytes=1), explicit=True))

    def test_n_1_1_k_accepted_when_explicit(self):
        t = torch.zeros(4, 1, 1, 32, dtype=torch.float32, device="cuda").detach()
        self.assertFalse(is_tensor_eligible(t, _cfg(min_bytes=1)))
        self.assertTrue(is_tensor_eligible(t, _cfg(min_bytes=1), explicit=True))

    def test_min_bytes_still_enforced_when_explicit(self):
        # 16 floats = 64 bytes; min_bytes=65 → rejected even explicit.
        t = torch.zeros(16, dtype=torch.float32, device="cuda").detach()
        self.assertFalse(
            is_tensor_eligible(t, _cfg(min_bytes=65), explicit=True)
        )

    def test_parameter_still_rejected_when_explicit(self):
        p = torch.nn.Parameter(
            torch.zeros(64, dtype=torch.float32, device="cuda")
        )
        self.assertFalse(is_tensor_eligible(p, _cfg(min_bytes=1), explicit=True))


if __name__ == "__main__":
    unittest.main()
