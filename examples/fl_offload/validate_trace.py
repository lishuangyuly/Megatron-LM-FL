#!/usr/bin/env python3
"""Validate FL offload/reload lifecycle and scheduling in Chrome traces."""

import argparse
import gzip
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


PREFIX = "mcfl:"
EXPECTED_LOCATIONS = {
    0: "after_combine_bwd",
    1: "after_dispatch_fwd",
    2: "after_dispatch_bwd",
    3: "after_combine_fwd",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--stages", type=int, default=4)
    return parser.parse_args()


def load_trace(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    return payload.get("traceEvents", payload)


def parse_name(name):
    fields = {}
    payload = name[len(PREFIX) :].strip()
    for part in payload.split("&"):
        if "=" in part:
            key, value = part.split("=", 1)
            fields[key] = value
    return fields


def rank_from_path(path):
    match = re.search(r"rank[-_]?(\d+)", path.name)
    return int(match.group(1)) if match else -1


def contains(parent, child):
    if parent["rank"] != child["rank"]:
        return False
    if parent.get("pid") != child.get("pid") or parent.get("tid") != child.get("tid"):
        return False
    parent_end = parent["ts"] + parent.get("dur", 0)
    child_end = child["ts"] + child.get("dur", 0)
    return parent["ts"] <= child["ts"] and child_end <= parent_end


def fail(errors, message):
    errors.append(message)


def main():
    args = parse_args()
    paths = sorted(args.trace_dir.glob("trace_rank*_step*.json*"))
    if not paths:
        raise SystemExit(f"no trace files found in {args.trace_dir}")

    events = []
    device_copies = Counter()
    for path in paths:
        rank = rank_from_path(path)
        for event in load_trace(path):
            name = event.get("name", "")
            compact_name = (
                re.sub(r"[^a-z0-9]", "", name.lower()) if isinstance(name, str) else ""
            )
            if any(marker in compact_name for marker in ("dtoh", "d2h", "devicetohost")):
                device_copies[(rank, "d2h")] += 1
            if any(marker in compact_name for marker in ("htod", "h2d", "hosttodevice")):
                device_copies[(rank, "h2d")] += 1
            if (
                event.get("ph") == "X"
                and isinstance(name, str)
                and name.startswith(PREFIX)
            ):
                events.append({**event, "rank": rank, "fields": parse_name(name)})

    errors = []
    by_func = Counter(event["fields"].get("func") for event in events)
    required_funcs = {
        "forward_backward_step",
        "attn",
        "dispatch",
        "moe",
        "combine",
        "fl_issue_loads",
        "fl_offload",
        "fl_reload",
    }
    missing_funcs = sorted(required_funcs - set(by_func))
    if missing_funcs:
        fail(errors, f"missing semantic event types: {', '.join(missing_funcs)}")

    steps = [event for event in events if event["fields"].get("func") == "forward_backward_step"]
    schedule_events = [
        event for event in events if event["fields"].get("func") == "fl_issue_loads"
    ]
    copy_issues = [
        event
        for event in events
        if event["fields"].get("func") in {"fl_offload", "fl_reload"}
        and event["fields"].get("phase") == "issue"
    ]

    for event in schedule_events:
        fields = event["fields"]
        schedule_stage = int(fields["schedule_stage"])
        expected = EXPECTED_LOCATIONS[schedule_stage % len(EXPECTED_LOCATIONS)]
        if fields.get("location") != expected:
            fail(
                errors,
                f"rank {event['rank']} schedule stage {schedule_stage} has location "
                f"{fields.get('location')}, expected {expected}",
            )
        if not any(contains(step, event) for step in steps):
            fail(errors, f"rank {event['rank']} schedule stage {schedule_stage} is outside a step")

    for event in copy_issues:
        fields = event["fields"]
        containing_schedules = [
            schedule for schedule in schedule_events if contains(schedule, event)
        ]
        if not containing_schedules:
            fail(
                errors,
                f"rank {event['rank']} {fields.get('func')} sequence "
                f"{fields.get('sequence_id')} stage {fields.get('stage_id')} "
                "is outside fl_issue_loads",
            )
        elif all(
            schedule["fields"].get("stage_id") != fields.get("stage_id")
            for schedule in containing_schedules
        ):
            fail(
                errors,
                f"rank {event['rank']} {fields.get('func')} sequence "
                f"{fields.get('sequence_id')} stage {fields.get('stage_id')} "
                "is nested at a different schedule stage",
            )

    lifecycle = defaultdict(lambda: defaultdict(list))
    for event in events:
        fields = event["fields"]
        func = fields.get("func")
        sequence_id = fields.get("sequence_id")
        phase = fields.get("phase")
        if func in {"fl_offload", "fl_reload"} and sequence_id is not None:
            lifecycle[(event["rank"], func, int(sequence_id))][phase].append(event)

    complete_counts = Counter()
    complete_sequences = defaultdict(set)
    partial_counts = Counter()
    expected_stages = list(range(args.stages))
    for (rank, func, sequence_id), phases in lifecycle.items():
        prologues = sorted(phases.get("prologue", []), key=lambda event: event["ts"])
        epilogues = sorted(phases.get("epilogue", []), key=lambda event: event["ts"])
        issues = sorted(phases.get("issue", []), key=lambda event: event["ts"])
        stage_ids = [int(event["fields"]["stage_id"]) for event in issues]
        if len(stage_ids) != len(set(stage_ids)):
            fail(errors, f"rank {rank} {func} sequence {sequence_id} repeats a stage: {stage_ids}")

        complete = bool(prologues and epilogues)
        if complete:
            complete_counts[(rank, func)] += 1
            complete_sequences[(rank, func)].add(sequence_id)
            if len(prologues) != 1 or len(epilogues) != 1:
                fail(
                    errors,
                    f"rank {rank} {func} sequence {sequence_id} has invalid lifecycle counts "
                    f"prologue={len(prologues)} epilogue={len(epilogues)}",
                )
            if stage_ids != expected_stages:
                fail(
                    errors,
                    f"rank {rank} {func} sequence {sequence_id} stages are {stage_ids}, "
                    f"expected {expected_stages}",
                )
            start = prologues[0]["ts"]
            end = epilogues[0]["ts"] + epilogues[0].get("dur", 0)
            if any(not start <= event["ts"] <= end for event in issues):
                fail(errors, f"rank {rank} {func} sequence {sequence_id} issue is out of lifecycle")
        else:
            partial_counts[(rank, func)] += 1

    for rank in sorted({event["rank"] for event in events}):
        for func in ("fl_offload", "fl_reload"):
            if complete_counts[(rank, func)] == 0:
                fail(errors, f"rank {rank} has no complete {func} sequence in the trace window")
        paired = complete_sequences[(rank, "fl_offload")] & complete_sequences[
            (rank, "fl_reload")
        ]
        if not paired:
            fail(errors, f"rank {rank} has no complete offload/reload sequence pair")
        for direction in ("d2h", "h2d"):
            if device_copies[(rank, direction)] == 0:
                fail(errors, f"rank {rank} trace has no GPU {direction} memcpy activity")

    location_counts = Counter(
        (event["rank"], event["fields"].get("location")) for event in schedule_events
    )
    print(f"[trace-check] files={len(paths)} semantic_events={len(events)}")
    for rank in sorted({event["rank"] for event in events}):
        for name in EXPECTED_LOCATIONS.values():
            if location_counts[(rank, name)] == 0:
                fail(errors, f"rank {rank} has no issue_loads event at {name}")
        paired_count = len(
            complete_sequences[(rank, "fl_offload")]
            & complete_sequences[(rank, "fl_reload")]
        )
        locations = ", ".join(
            f"{name}={location_counts[(rank, name)]}" for name in EXPECTED_LOCATIONS.values()
        )
        print(
            f"[trace-check] rank={rank} steps="
            f"{sum(event['rank'] == rank for event in steps)} "
            f"offload_complete={complete_counts[(rank, 'fl_offload')]} "
            f"reload_complete={complete_counts[(rank, 'fl_reload')]} "
            f"offload_boundary_partial={partial_counts[(rank, 'fl_offload')]} "
            f"reload_boundary_partial={partial_counts[(rank, 'fl_reload')]} "
            f"paired_sequences={paired_count} "
            f"gpu_d2h={device_copies[(rank, 'd2h')]} "
            f"gpu_h2d={device_copies[(rank, 'h2d')]} "
            f"locations: {locations}"
        )

    if errors:
        print("[trace-check] FAILED")
        for message in errors:
            print(f"  - {message}")
        raise SystemExit(1)
    print("[trace-check] PASSED: staged D2H/H2D counts and schedule positions are consistent")


if __name__ == "__main__":
    main()
