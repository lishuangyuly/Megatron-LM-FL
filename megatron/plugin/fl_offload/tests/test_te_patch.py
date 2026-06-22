"""Tests for the explicit-pack TE monkey-patch (commit 7.2).

CPU-runnable parts (no GPU needed):
* anchor uniqueness — the patch finds its substitution sites exactly once
  in the live TE source (guards against silent TE drift);
* patched source compiles and references our pack/unpack logic;
* install / uninstall swap and restore the staticmethods, idempotently;
* the -N backward slice offset matches the save-layout derivation.

The real forward/backward numerical round-trip needs a CUDA TE GEMM and
is covered by the real-training B-series (plan 7.2 测试).
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
    ),
)


def _te_available() -> bool:
    try:
        import transformer_engine.pytorch.module.grouped_linear  # noqa: F401

        return True
    except Exception:
        return False


TE = _te_available()
skip_no_te = unittest.skipUnless(TE, "TransformerEngine not importable")


@skip_no_te
class TestPatchUtil(unittest.TestCase):
    def test_anchors_unique_in_live_source(self):
        import inspect
        import textwrap

        import transformer_engine.pytorch.module.grouped_linear as gl
        from megatron.plugin.fl_offload.te_patch import grouped_linear as glp

        cases = [
            (gl._GroupedLinear.forward, glp._FORWARD_ANCHOR),
            (gl._GroupedLinear.backward, glp._BACKWARD_ANCHOR),
        ]
        for method, anchor in cases:
            src = textwrap.dedent(inspect.getsource(method))
            lines = src.splitlines()
            while lines and not lines[0].lstrip().startswith("def "):
                lines.pop(0)
            src = "\n".join(lines)
            self.assertEqual(
                src.count(anchor), 1,
                f"anchor not unique in {method.__qualname__}",
            )

    def test_patched_source_compiles_and_substitutes(self):
        import transformer_engine.pytorch.module.grouped_linear as gl
        from megatron.plugin.fl_offload.te_patch import _patch_util as pu
        from megatron.plugin.fl_offload.te_patch import grouped_linear as glp

        fsrc = pu._patched_source(
            gl._GroupedLinear.forward,
            [pu._Substitution(glp._FORWARD_ANCHOR, glp._FORWARD_REPLACEMENT)],
            "_fl_forward",
        )
        self.assertIn("fl_tensor_packs", fsrc)
        self.assertNotIn("*inputmats,", fsrc)  # inputmats removed from save
        compile(fsrc, "<t>", "exec")

        bsrc = pu._patched_source(
            gl._GroupedLinear.backward,
            [pu._Substitution(glp._BACKWARD_ANCHOR, glp._BACKWARD_REPLACEMENT)],
            "_fl_backward",
        )
        self.assertIn("_fl_unpack", bsrc)
        # offset: weights now starts at 0, biases at 2N (was 3N..4N)
        self.assertIn("weights = saved_tensors[0 : N]", bsrc)
        self.assertIn("biases = saved_tensors[2 * N : 3 * N]", bsrc)
        compile(bsrc, "<t>", "exec")

    def test_missing_anchor_raises(self):
        import transformer_engine.pytorch.module.grouped_linear as gl
        from megatron.plugin.fl_offload.te_patch import _patch_util as pu

        with self.assertRaises(RuntimeError):
            pu._patched_source(
                gl._GroupedLinear.forward,
                [pu._Substitution("this anchor does not exist", "x")],
                "_fl_x",
            )


@skip_no_te
class TestInstallUninstall(unittest.TestCase):
    def setUp(self):
        from megatron.plugin.fl_offload import te_patch

        self.te_patch = te_patch
        import transformer_engine.pytorch.module.grouped_linear as gl

        self.gl = gl
        self._orig_fwd = gl._GroupedLinear.forward
        self._orig_bwd = gl._GroupedLinear.backward

    def tearDown(self):
        self.te_patch.remove_te_patches()

    def test_install_swaps_then_uninstall_restores(self):
        gl = self.gl
        self.te_patch.apply_te_patches()
        self.assertIsNot(gl._GroupedLinear.forward, self._orig_fwd)
        self.assertIsNot(gl._GroupedLinear.backward, self._orig_bwd)
        # patched methods reference our bookkeeping
        fcode = gl._GroupedLinear.forward.__code__
        self.assertIn("fl_tensor_packs", fcode.co_names)
        self.te_patch.remove_te_patches()
        self.assertIs(gl._GroupedLinear.forward, self._orig_fwd)
        self.assertIs(gl._GroupedLinear.backward, self._orig_bwd)

    def test_install_idempotent(self):
        gl = self.gl
        self.te_patch.apply_te_patches()
        f1 = gl._GroupedLinear.forward
        self.te_patch.apply_te_patches()  # second call no-op
        self.assertIs(gl._GroupedLinear.forward, f1)

    def test_remove_without_install_is_safe(self):
        self.te_patch.remove_te_patches()  # no raise


if __name__ == "__main__":
    unittest.main()
