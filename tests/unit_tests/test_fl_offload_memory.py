import subprocess
import sys
from pathlib import Path


def _line(rank, iteration, peak):
    return (
        f"[FL memory] rank={rank} iteration={iteration} "
        f"allocated_mib=50.00 peak_allocated_mib={peak:.2f} "
        "reserved_mib=120.00 peak_reserved_mib=120.00"
    )


def _write_log(path, peaks_by_rank):
    lines = []
    for rank, peaks in peaks_by_rank.items():
        lines.extend(_line(rank, iteration, peak) for iteration, peak in enumerate(peaks, 1))
    path.write_text("\n".join(lines), encoding="utf-8")


def _run_comparison(baseline_log, offload_log):
    script = Path(__file__).parents[2] / "examples/fl_offload/compare_memory.py"
    return subprocess.run(
        [
            sys.executable,
            str(script),
            "--baseline-log",
            str(baseline_log),
            "--offload-log",
            str(offload_log),
            "--warmup-iters",
            "1",
            "--min-reduction-mib",
            "1",
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_memory_comparison_accepts_steady_state_reduction(tmp_path):
    baseline = tmp_path / "baseline.log"
    offload = tmp_path / "offload.log"
    _write_log(baseline, {0: [150, 105, 100], 1: [140, 95, 90]})
    _write_log(offload, {0: [155, 82, 80], 1: [145, 76, 75]})
    result = _run_comparison(baseline, offload)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "global baseline_peak_mib=105.00 offload_peak_mib=82.00" in result.stdout
    assert "PASSED" in result.stdout


def test_memory_comparison_rejects_global_peak_regression(tmp_path):
    baseline = tmp_path / "baseline.log"
    offload = tmp_path / "offload.log"
    _write_log(baseline, {0: [150, 100, 100]})
    _write_log(offload, {0: [150, 104, 103]})
    result = _run_comparison(baseline, offload)
    assert result.returncode == 1
    assert "regressed by 4.00 MiB" in result.stdout
    assert "FAILED" in result.stdout
