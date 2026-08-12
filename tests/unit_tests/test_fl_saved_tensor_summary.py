import json

from examples.fl_offload import summarize_saved_tensors


def _report(scope, pointer, shared_scopes=None):
    tensor = {
        "source": "autograd",
        "shape": [4, 8],
        "stride": [8, 1],
        "dtype": "torch.bfloat16",
        "device": "cuda:0",
        "logical_bytes": 64,
        "storage_ptr": pointer,
        "storage_nbytes": 64,
        "parameter": False,
        "shared_with_scopes": shared_scopes or [],
    }
    return {
        "rank": 0,
        "scope": scope,
        "invocation": 0,
        "saved_tensors": 1,
        "autograd_saved": 1,
        "explicit_saved": 0,
        "activation_logical_bytes": 64,
        "parameter_logical_bytes": 0,
        "unique_activation_storage_bytes": 64,
        "cross_scope_storage_tensors": int(bool(shared_scopes)),
        "tensors": [tensor],
    }


def test_load_and_summarize_reports(tmp_path, capsys):
    log = tmp_path / "train.log"
    reports = [
        _report("core_attn", 1234),
        _report("attn_proj", 1234, ["core_attn"]),
    ]
    log.write_text(
        "unrelated output\n"
        + "\n".join(
            summarize_saved_tensors.PREFIX + json.dumps(report) for report in reports
        )
        + "\n",
        encoding="utf-8",
    )

    loaded = summarize_saved_tensors.load_reports([log])
    assert len(loaded) == 2
    summarize_saved_tensors.print_scope_summary(loaded)
    summarize_saved_tensors.print_rank_totals(loaded)
    summarize_saved_tensors.print_shared_storages(loaded, top=30)

    output = capsys.readouterr().out
    assert "core_attn" in output
    assert "attn_proj" in output
    assert "cross_scope_shared_MiB" in output
    assert "attn_proj,core_attn" in output


def test_filter_reports():
    reports = [_report("core_attn", 1), {**_report("expert_fc1", 2), "rank": 1}]

    assert summarize_saved_tensors.filter_reports(reports, ranks=[1])[0]["scope"] == "expert_fc1"
    assert summarize_saved_tensors.filter_reports(reports, scopes=["core_attn"])[0]["rank"] == 0
