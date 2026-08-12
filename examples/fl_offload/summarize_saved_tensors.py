#!/usr/bin/env python3
"""Summarize FL saved-tensor profiler JSON lines from training logs."""

import argparse
import json
from collections import defaultdict
from pathlib import Path


PREFIX = "[FL saved-tensor-profile] "
MIB = 1024 * 1024


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path, help="Training log files to read")
    parser.add_argument("--rank", type=int, action="append", help="Only include this rank")
    parser.add_argument("--scope", action="append", help="Only include this semantic scope")
    parser.add_argument(
        "--details",
        action="store_true",
        help="Show activation tensor signatures, ordered by logical size",
    )
    parser.add_argument(
        "--include-parameters",
        action="store_true",
        help="Include Parameter tensors in the detailed table",
    )
    parser.add_argument(
        "--min-mib",
        type=float,
        default=0.0,
        help="Minimum logical tensor size shown by --details",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=30,
        help="Maximum rows in detailed and shared-storage tables (0 means unlimited)",
    )
    return parser.parse_args()


def load_reports(paths):
    reports = []
    decoder = json.JSONDecoder()
    for path in paths:
        if not path.is_file():
            raise SystemExit(f"log file not found: {path}")
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            for line_number, line in enumerate(stream, 1):
                marker = line.find(PREFIX)
                if marker < 0:
                    continue
                payload = line[marker + len(PREFIX) :].lstrip()
                try:
                    report, _ = decoder.raw_decode(payload)
                except json.JSONDecodeError as error:
                    raise SystemExit(f"invalid profiler JSON at {path}:{line_number}: {error}")
                report["_path"] = str(path)
                report["_line"] = line_number
                reports.append(report)
    return reports


def format_mib(value):
    return f"{value / MIB:.2f}"


def format_average_mib(total, count):
    return format_mib(total / max(count, 1))


def print_table(headers, rows):
    if not rows:
        return
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(str(value)))
    template = "  ".join(f"{{:{width}}}" for width in widths)
    print(template.format(*headers))
    print(template.format(*(('-' * width) for width in widths)))
    for row in rows:
        print(template.format(*(str(value) for value in row)))


def storage_key(rank, tensor):
    pointer = tensor.get("storage_ptr")
    size = tensor.get("storage_nbytes")
    if pointer is None or size is None:
        return None
    return rank, tensor.get("device"), pointer, size


def filter_reports(reports, ranks=None, scopes=None):
    rank_filter = set(ranks or [])
    scope_filter = set(scopes or [])
    return [
        report
        for report in reports
        if (not rank_filter or report["rank"] in rank_filter)
        and (not scope_filter or report["scope"] in scope_filter)
    ]


def print_scope_summary(reports):
    grouped = defaultdict(list)
    for report in reports:
        grouped[(report["rank"], report["scope"])].append(report)

    rows = []
    for (rank, scope), items in sorted(grouped.items()):
        rows.append(
            (
                rank,
                scope,
                len(items),
                f"{sum(item['saved_tensors'] for item in items) / len(items):.1f}",
                format_average_mib(
                    sum(item["activation_logical_bytes"] for item in items), len(items)
                ),
                format_average_mib(
                    sum(item["unique_activation_storage_bytes"] for item in items), len(items)
                ),
                format_average_mib(
                    sum(item["parameter_logical_bytes"] for item in items), len(items)
                ),
                sum(item["explicit_saved"] for item in items),
                sum(item["cross_scope_storage_tensors"] for item in items),
            )
        )

    print("[saved-tensor-summary] per-call averages")
    print_table(
        (
            "rank",
            "scope",
            "calls",
            "saved",
            "act_logical_MiB",
            "act_storage_MiB",
            "param_MiB",
            "explicit",
            "shared",
        ),
        rows,
    )


def collect_storage(reports):
    storages = {}
    unknown_storage_logical_bytes = defaultdict(int)
    for report in reports:
        rank = report["rank"]
        for tensor in report.get("tensors", []):
            if tensor.get("parameter"):
                continue
            key = storage_key(rank, tensor)
            if key is None:
                unknown_storage_logical_bytes[rank] += tensor.get("logical_bytes", 0)
                continue
            entry = storages.setdefault(
                key,
                {
                    "rank": rank,
                    "device": tensor.get("device"),
                    "storage_ptr": tensor.get("storage_ptr"),
                    "storage_nbytes": tensor.get("storage_nbytes", 0),
                    "scopes": set(),
                    "shapes": set(),
                },
            )
            entry["scopes"].add(report["scope"])
            entry["shapes"].add(str(tensor.get("shape")))
    return storages, unknown_storage_logical_bytes


def print_rank_totals(reports):
    storages, unknown = collect_storage(reports)
    totals = defaultdict(int)
    shared = defaultdict(int)
    for entry in storages.values():
        totals[entry["rank"]] += entry["storage_nbytes"]
        if len(entry["scopes"]) > 1:
            shared[entry["rank"]] += entry["storage_nbytes"]

    ranks = sorted({report["rank"] for report in reports})
    rows = [
        (
            rank,
            format_mib(totals[rank]),
            format_mib(shared[rank]),
            format_mib(unknown[rank]),
        )
        for rank in ranks
    ]
    print("\n[saved-tensor-summary] union across reported scopes")
    print_table(
        ("rank", "unique_activation_MiB", "cross_scope_shared_MiB", "unknown_storage_logical_MiB"),
        rows,
    )


def limited(items, top):
    return items if top == 0 else items[:top]


def print_shared_storages(reports, top):
    storages, _ = collect_storage(reports)
    shared = [entry for entry in storages.values() if len(entry["scopes"]) > 1]
    shared.sort(key=lambda item: item["storage_nbytes"], reverse=True)
    if not shared:
        return
    rows = [
        (
            item["rank"],
            format_mib(item["storage_nbytes"]),
            ",".join(sorted(item["scopes"])),
            ";".join(sorted(item["shapes"])),
            item["storage_ptr"],
        )
        for item in limited(shared, top)
    ]
    print("\n[saved-tensor-summary] storage shared by multiple scopes")
    print_table(("rank", "storage_MiB", "scopes", "shapes", "storage_ptr"), rows)


def print_details(reports, include_parameters, min_mib, top):
    signatures = defaultdict(lambda: {"count": 0, "logical_bytes": 0, "storage_nbytes": 0})
    for report in reports:
        for tensor in report.get("tensors", []):
            if tensor.get("parameter") and not include_parameters:
                continue
            logical_bytes = tensor.get("logical_bytes", 0)
            if logical_bytes < min_mib * MIB:
                continue
            key = (
                report["rank"],
                report["scope"],
                tensor.get("source"),
                "parameter" if tensor.get("parameter") else "activation",
                tensor.get("dtype"),
                str(tensor.get("shape")),
                str(tensor.get("stride")),
            )
            item = signatures[key]
            item["count"] += 1
            item["logical_bytes"] = max(item["logical_bytes"], logical_bytes)
            item["storage_nbytes"] = max(
                item["storage_nbytes"], tensor.get("storage_nbytes") or 0
            )

    ordered = sorted(
        signatures.items(),
        key=lambda pair: (pair[1]["logical_bytes"], pair[1]["storage_nbytes"]),
        reverse=True,
    )
    rows = []
    for key, item in limited(ordered, top):
        rank, scope, source, kind, dtype, shape, stride = key
        rows.append(
            (
                rank,
                scope,
                source,
                kind,
                item["count"],
                format_mib(item["logical_bytes"]),
                format_mib(item["storage_nbytes"]),
                dtype,
                shape,
                stride,
            )
        )
    print("\n[saved-tensor-summary] tensor signatures")
    print_table(
        (
            "rank",
            "scope",
            "source",
            "kind",
            "count",
            "logical_MiB",
            "storage_MiB",
            "dtype",
            "shape",
            "stride",
        ),
        rows,
    )


def main():
    args = parse_args()
    reports = filter_reports(load_reports(args.logs), args.rank, args.scope)
    if not reports:
        raise SystemExit("no matching FL saved-tensor profiler records found")
    print(f"[saved-tensor-summary] reports={len(reports)} files={len(args.logs)}")
    print_scope_summary(reports)
    print_rank_totals(reports)
    print_shared_storages(reports, args.top)
    if args.details:
        print_details(reports, args.include_parameters, args.min_mib, args.top)


if __name__ == "__main__":
    main()
