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
    parser.add_argument(
        "--analyze-overlap",
        action="store_true",
        help="Correlate FL memcpy activity with GPU communication and compute kernels.",
    )
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


def interval_overlap(left, right):
    start = max(left["ts"], right["ts"])
    end = min(
        left["ts"] + left.get("dur", 0),
        right["ts"] + right.get("dur", 0),
    )
    return max(0, end - start)


def covered_duration(event, candidates):
    event_start = event["ts"]
    event_end = event_start + event.get("dur", 0)
    intervals = []
    for candidate in candidates:
        start = max(event_start, candidate["ts"])
        end = min(event_end, candidate["ts"] + candidate.get("dur", 0))
        if start < end:
            intervals.append((start, end))
    if not intervals:
        return 0
    intervals.sort()
    covered = 0
    current_start, current_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            covered += current_end - current_start
            current_start, current_end = start, end
    return covered + current_end - current_start


def compact_event_name(event):
    name = event.get("name", "")
    return re.sub(r"[^a-z0-9]", "", name.lower()) if isinstance(name, str) else ""


def copy_direction(event):
    name = compact_event_name(event)
    if any(marker in name for marker in ("dtoh", "d2h", "devicetohost")):
        return "d2h"
    if any(marker in name for marker in ("htod", "h2d", "hosttodevice")):
        return "h2d"
    return None


def is_gpu_kernel(event):
    if event.get("ph") != "X" or event.get("dur", 0) <= 0:
        return False
    categories = {part.strip() for part in str(event.get("cat", "")).lower().split(",")}
    return bool(categories & {"kernel", "gpu_kernel"})


def is_communication_kernel(event):
    if not is_gpu_kernel(event):
        return False
    name = compact_event_name(event)
    return any(
        marker in name
        for marker in (
            "nccl",
            "rccl",
            "allreduce",
            "alltoall",
            "allgather",
            "reducescatter",
            "sendrecv",
        )
    )


def is_cpu_semantic_event(event, name):
    if event.get("ph") != "X" or not isinstance(name, str) or not name.startswith(PREFIX):
        return False
    category = event.get("cat", "")
    if not category:
        return True
    categories = {part.strip() for part in category.lower().split(",")}
    return "user_annotation" in categories and "gpu_user_annotation" not in categories


def main():
    args = parse_args()
    paths = sorted(args.trace_dir.glob("trace_rank*_step*.json*"))
    if not paths:
        raise SystemExit(f"no trace files found in {args.trace_dir}")

    events = []
    raw_events = []
    device_copies = Counter()
    projected_semantic_events = Counter()
    seen_semantic_events = set()
    for path in paths:
        rank = rank_from_path(path)
        for event in load_trace(path):
            raw_events.append({**event, "rank": rank})
            name = event.get("name", "")
            direction = copy_direction(event)
            if direction == "d2h":
                device_copies[(rank, "d2h")] += 1
            if direction == "h2d":
                device_copies[(rank, "h2d")] += 1
            if isinstance(name, str) and name.startswith(PREFIX):
                if "gpu_user_annotation" in str(event.get("cat", "")).lower():
                    projected_semantic_events[rank] += 1
                if is_cpu_semantic_event(event, name):
                    identity = (
                        rank,
                        name,
                        event.get("pid"),
                        event.get("tid"),
                        event.get("ts"),
                        event.get("dur"),
                    )
                    if identity not in seen_semantic_events:
                        seen_semantic_events.add(identity)
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
            f"gpu_projected_annotations_ignored={projected_semantic_events[rank]} "
            f"locations: {locations}"
        )

    if args.analyze_overlap:
        projected_issues = []
        gpu_copies = []
        communication_kernels = []
        compute_kernels = []
        for event in raw_events:
            name = event.get("name", "")
            category = str(event.get("cat", "")).lower()
            if (
                isinstance(name, str)
                and name.startswith(PREFIX)
                and "gpu_user_annotation" in category
            ):
                fields = parse_name(name)
                if (
                    fields.get("func") in {"fl_offload", "fl_reload"}
                    and fields.get("phase") == "issue"
                ):
                    projected_issues.append({**event, "fields": fields})
            direction = copy_direction(event)
            if direction is not None and event.get("ph") == "X" and event.get("dur", 0) > 0:
                gpu_copies.append({**event, "direction": direction})
            if is_communication_kernel(event):
                communication_kernels.append(event)
            elif is_gpu_kernel(event):
                compute_kernels.append(event)

        for rank in sorted({event["rank"] for event in events}):
            rank_issues = [event for event in projected_issues if event["rank"] == rank]
            matched_copies = []
            for copy in (event for event in gpu_copies if event["rank"] == rank):
                expected_func = "fl_offload" if copy["direction"] == "d2h" else "fl_reload"
                if any(
                    issue["fields"].get("func") == expected_func
                    and issue.get("pid") == copy.get("pid")
                    and issue.get("tid") == copy.get("tid")
                    and interval_overlap(issue, copy) > 0
                    for issue in rank_issues
                ):
                    matched_copies.append(copy)

            rank_comm = [event for event in communication_kernels if event["rank"] == rank]
            rank_compute = [event for event in compute_kernels if event["rank"] == rank]
            copy_time = sum(event.get("dur", 0) for event in matched_copies)
            communication_overlap = sum(
                covered_duration(
                    copy,
                    [
                        event
                        for event in rank_comm
                        if event.get("tid") != copy.get("tid")
                    ],
                )
                for copy in matched_copies
            )
            compute_overlap = sum(
                covered_duration(
                    copy,
                    [
                        event
                        for event in rank_compute
                        if event.get("tid") != copy.get("tid")
                    ],
                )
                for copy in matched_copies
            )
            d2h = sum(copy["direction"] == "d2h" for copy in matched_copies)
            h2d = sum(copy["direction"] == "h2d" for copy in matched_copies)
            compute_percent = 100.0 * compute_overlap / copy_time if copy_time else 0.0
            communication_percent = (
                100.0 * communication_overlap / copy_time if copy_time else 0.0
            )
            communication_gap = (
                "unknown"
                if not matched_copies or not rank_comm
                else "yes" if communication_overlap == 0 else "no"
            )
            actual_compute_overlap = "yes" if compute_overlap > 0 else "no"
            print(
                f"[trace-overlap] rank={rank} projected_issues={len(rank_issues)} "
                f"matched_d2h={d2h} matched_h2d={h2d} "
                f"copy_time_us={copy_time:.3f} "
                f"compute_overlap_us={compute_overlap:.3f} "
                f"compute_overlap_percent={compute_percent:.2f} "
                f"communication_overlap_us={communication_overlap:.3f} "
                f"communication_overlap_percent={communication_percent:.2f} "
                f"communication_gap={communication_gap} "
                f"actual_compute_overlap={actual_compute_overlap} "
                f"communication_kernels={len(rank_comm)} compute_kernels={len(rank_compute)}"
            )
            if rank_issues and not matched_copies:
                fail(errors, f"rank {rank} has no GPU memcpy attributable to FL issue ranges")

    if errors:
        print("[trace-check] FAILED")
        for message in errors:
            print(f"  - {message}")
        raise SystemExit(1)
    print("[trace-check] PASSED: staged D2H/H2D counts and schedule positions are consistent")


if __name__ == "__main__":
    main()
