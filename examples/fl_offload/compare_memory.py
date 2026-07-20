#!/usr/bin/env python3
"""Compare steady-state per-iteration GPU peaks from two training logs."""

import argparse
import re
from collections import defaultdict
from pathlib import Path


MEMORY_LINE = re.compile(
    r"\[FL memory\] rank=(?P<rank>\d+) iteration=(?P<iteration>\d+) "
    r"allocated_mib=(?P<allocated>[0-9.]+) "
    r"peak_allocated_mib=(?P<peak_allocated>[0-9.]+) "
    r"reserved_mib=(?P<reserved>[0-9.]+) "
    r"peak_reserved_mib=(?P<peak_reserved>[0-9.]+)"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-log", type=Path, required=True)
    parser.add_argument("--offload-log", type=Path, required=True)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--min-reduction-mib", type=float, default=1.0)
    parser.add_argument(
        "--max-rank-regression-mib",
        type=float,
        default=None,
        help=(
            "Optional strict per-rank regression limit. By default, local regressions "
            "are warnings and the distributed global peak determines pass/fail."
        ),
    )
    return parser.parse_args()


def load_peaks(path, warmup_iters):
    records = defaultdict(list)
    contents = path.read_text(encoding="utf-8", errors="replace")
    for match in MEMORY_LINE.finditer(contents):
        if int(match.group("iteration")) > warmup_iters:
            records[int(match.group("rank"))].append(float(match.group("peak_allocated")))
    if not records:
        raise ValueError(f"no steady-state FL memory records found in {path}")
    return records


def main():
    args = parse_args()
    baseline_records = load_peaks(args.baseline_log, args.warmup_iters)
    offload_records = load_peaks(args.offload_log, args.warmup_iters)
    if set(baseline_records) != set(offload_records):
        raise SystemExit(
            "rank sets differ: "
            f"baseline={sorted(baseline_records)} offload={sorted(offload_records)}"
        )

    errors = []
    baseline_peaks = {}
    offload_peaks = {}
    for rank in sorted(baseline_records):
        if len(baseline_records[rank]) != len(offload_records[rank]):
            errors.append(
                f"rank {rank} sample counts differ: "
                f"baseline={len(baseline_records[rank])} "
                f"offload={len(offload_records[rank])}"
            )
        baseline_peak = max(baseline_records[rank])
        offload_peak = max(offload_records[rank])
        baseline_peaks[rank] = baseline_peak
        offload_peaks[rank] = offload_peak
        reduction = baseline_peak - offload_peak
        reduction_percent = 100.0 * reduction / baseline_peak
        print(
            f"[FL memory-check] rank={rank} baseline_peak_mib={baseline_peak:.2f} "
            f"offload_peak_mib={offload_peak:.2f} reduction_mib={reduction:.2f} "
            f"reduction_percent={reduction_percent:.2f}"
        )
        if reduction < 0:
            regression = -reduction
            print(
                f"[FL memory-check] WARNING: rank {rank} regressed by "
                f"{regression:.2f} MiB relative to its local baseline"
            )
            if (
                args.max_rank_regression_mib is not None
                and regression > args.max_rank_regression_mib
            ):
                errors.append(f"rank {rank} regressed by {regression:.2f} MiB")

    baseline_global = max(baseline_peaks.values())
    offload_global = max(offload_peaks.values())
    global_reduction = baseline_global - offload_global
    global_percent = 100.0 * global_reduction / baseline_global
    print(
        f"[FL memory-check] global baseline_peak_mib={baseline_global:.2f} "
        f"offload_peak_mib={offload_global:.2f} reduction_mib={global_reduction:.2f} "
        f"reduction_percent={global_percent:.2f}"
    )
    if global_reduction < args.min_reduction_mib:
        errors.append(
            f"global peak reduction {global_reduction:.2f} MiB is below "
            f"required {args.min_reduction_mib:.2f} MiB"
        )

    if errors:
        print("[FL memory-check] FAILED")
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(1)
    print(
        "[FL memory-check] PASSED: offload reduces the distributed "
        "steady-state training peak memory"
    )


if __name__ == "__main__":
    main()
