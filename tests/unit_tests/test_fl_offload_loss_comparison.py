from pathlib import Path

import pytest

from examples.fl_offload.compare_losses import compare_losses, load_losses


def test_load_and_compare_iteration_losses(tmp_path):
    baseline_path = tmp_path / "baseline.log"
    offload_path = tmp_path / "offload.log"
    baseline_path.write_text(
        "iteration 1/ 3 | lm loss: 9.100000E+00 |\n"
        "iteration 2/ 3 | lm loss: 9.000000E+00 |\n"
    )
    offload_path.write_text(
        "iteration 1/ 3 | lm loss: 9.100000E+00 |\n"
        "iteration 2/ 3 | lm loss: 9.000000E+00 |\n"
    )

    compare_losses(load_losses(baseline_path), load_losses(offload_path))


def test_compare_iteration_losses_rejects_drift():
    with pytest.raises(ValueError, match="iteration 2"):
        compare_losses({1: 9.1, 2: 9.0}, {1: 9.1, 2: 8.9})
