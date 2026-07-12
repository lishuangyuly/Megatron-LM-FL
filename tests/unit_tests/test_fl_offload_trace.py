import json
import subprocess
import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from megatron.plugin.profile import (
    PREFIX,
    bwd_record_pair,
    semantic_record,
    set_profile_enabled,
)


def _semantic_names(profiler):
    return [event.name for event in profiler.events() if event.name.startswith(PREFIX)]


def test_profile_backward_range_is_numerically_transparent():
    torch.manual_seed(17)
    baseline = torch.randn(8, requires_grad=True)
    (baseline.square().sum()).backward()

    torch.manual_seed(17)
    instrumented = torch.randn(8, requires_grad=True)
    set_profile_enabled(True)
    try:
        start, end = bwd_record_pair()
        with profile(activities=[ProfilerActivity.CPU]) as profiler:
            value = start(instrumented)
            loss = end(value.square().sum(), "mcfl: phase=backward&func=test")
            loss.backward()
    finally:
        set_profile_enabled(False)

    torch.testing.assert_close(instrumented.grad, baseline.grad, rtol=0, atol=0)
    assert "mcfl: phase=backward&func=test" in _semantic_names(profiler)


def test_semantic_record_is_disabled_by_default():
    set_profile_enabled(False)
    with profile(activities=[ProfilerActivity.CPU]) as profiler:
        with semantic_record(func="must_not_appear"):
            torch.ones(1).add_(1)
    assert not _semantic_names(profiler)


def _event(name, ts, dur=1, category="user_annotation", tid=1):
    return {
        "name": name,
        "ph": "X",
        "cat": category,
        "pid": 1,
        "tid": tid,
        "ts": ts,
        "dur": dur,
    }


def _write_trace(path: Path, missing_stage=None):
    events = [
        _event("Memcpy DtoH (Device -> Host)", 10, 1, category="gpu_memcpy", tid=7),
        _event("Memcpy HtoD (Host -> Device)", 12, 1, category="gpu_memcpy", tid=7),
        _event(
            "mcfl: func=forward_backward_step&f_virtual_microbatch_id=0"
            "&f_microbatch_id=0&f_model_chunk_id=0"
            "&b_virtual_microbatch_id=0&b_model_chunk_id=0",
            0,
            1000,
        ),
        _event("mcfl: phase=forward&func=attn", 20, 5),
        _event("mcfl: phase=forward&func=dispatch", 30, 5),
        _event("mcfl: phase=forward&func=moe", 40, 5),
        _event("mcfl: phase=forward&func=combine", 50, 5),
        _event("mcfl: func=fl_offload&phase=prologue&sequence_id=0", 60, 2),
        _event("mcfl: func=fl_reload&phase=prologue&sequence_id=0", 70, 2),
    ]
    locations = (
        "after_combine_bwd",
        "after_dispatch_fwd",
        "after_dispatch_bwd",
        "after_combine_fwd",
    )
    for stage_id, location in enumerate(locations):
        schedule_ts = 100 + stage_id * 100
        events.append(
            _event(
                f"mcfl: func=fl_issue_loads&schedule_stage={stage_id}"
                f"&stage_id={stage_id}&location={location}",
                schedule_ts,
                20,
            )
        )
        if stage_id != missing_stage:
            for func, offset in (("fl_offload", 2), ("fl_reload", 5)):
                events.append(
                    _event(
                        f"mcfl: func={func}&phase=issue&sequence_id=0&stage_id={stage_id}",
                        schedule_ts + offset,
                        1,
                    )
                )
    events.extend(
        [
            _event("mcfl: func=fl_offload&phase=epilogue&sequence_id=0", 600, 2),
            _event("mcfl: func=fl_reload&phase=epilogue&sequence_id=0", 610, 2),
        ]
    )
    events.extend(
        {
            **event,
            "cat": "gpu_user_annotation",
            "tid": 7,
        }
        for event in list(events)
        if event["name"].startswith(PREFIX)
        and event["name"].split("&", 1)[0]
        in {
            "mcfl: func=fl_offload",
            "mcfl: func=fl_reload",
        }
    )
    path.write_text(json.dumps({"traceEvents": events}), encoding="utf-8")


def _run_validator(trace_dir):
    script = Path(__file__).parents[2] / "examples/fl_offload/validate_trace.py"
    return subprocess.run(
        [sys.executable, str(script), "--trace-dir", str(trace_dir), "--stages", "4"],
        check=False,
        capture_output=True,
        text=True,
    )


def test_trace_validator_accepts_complete_four_stage_lifecycle(tmp_path):
    _write_trace(tmp_path / "trace_rank0_step3.json")
    result = _run_validator(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASSED" in result.stdout
    assert "gpu_projected_annotations_ignored=12" in result.stdout


def test_trace_validator_rejects_missing_stage(tmp_path):
    _write_trace(tmp_path / "trace_rank0_step3.json", missing_stage=2)
    result = _run_validator(tmp_path)
    assert result.returncode == 1
    assert "stages are [0, 1, 3]" in result.stdout
