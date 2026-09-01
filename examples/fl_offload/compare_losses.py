"""Compare iteration losses from baseline and FL offload smoke logs."""

import argparse
import math
from pathlib import Path
import re


LOSS_PATTERN = re.compile(
    r"iteration\s+(?P<iteration>\d+)/.*?lm loss:\s*(?P<loss>[0-9.eE+-]+)"
)


def load_losses(path):
    losses = {}
    for line in path.read_text(errors="replace").splitlines():
        match = LOSS_PATTERN.search(line)
        if match is not None:
            losses[int(match.group("iteration"))] = float(match.group("loss"))
    if not losses:
        raise ValueError(f"no iteration losses found in {path}")
    return losses


def compare_losses(baseline, offload, relative_tolerance=1e-6, absolute_tolerance=1e-7):
    if baseline.keys() != offload.keys():
        raise ValueError(
            "baseline/offload iteration sets differ: "
            f"{sorted(baseline)} != {sorted(offload)}"
        )
    mismatches = []
    for iteration in sorted(baseline):
        if not math.isclose(
            baseline[iteration],
            offload[iteration],
            rel_tol=relative_tolerance,
            abs_tol=absolute_tolerance,
        ):
            mismatches.append(
                f"iteration {iteration}: baseline={baseline[iteration]:.9g}, "
                f"offload={offload[iteration]:.9g}"
            )
    if mismatches:
        raise ValueError("loss mismatch:\n  " + "\n  ".join(mismatches))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-log", type=Path, required=True)
    parser.add_argument("--offload-log", type=Path, required=True)
    parser.add_argument("--relative-tolerance", type=float, default=1e-6)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-7)
    args = parser.parse_args()

    baseline = load_losses(args.baseline_log)
    offload = load_losses(args.offload_log)
    compare_losses(
        baseline,
        offload,
        relative_tolerance=args.relative_tolerance,
        absolute_tolerance=args.absolute_tolerance,
    )
    print(
        f"[FL loss-check] PASSED: {len(baseline)} iteration losses match",
        flush=True,
    )


if __name__ == "__main__":
    main()
