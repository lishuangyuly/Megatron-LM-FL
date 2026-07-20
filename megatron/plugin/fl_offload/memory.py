"""Training-step GPU memory measurement for FL offload validation."""

from megatron.plugin.platform import get_platform


MIB = 1 << 20


def reset_training_peak():
    """Start a peak-memory window after all prior asynchronous work is complete."""
    platform = get_platform()
    platform.synchronize()
    platform.reset_peak_memory_stats()


def report_training_peak(iteration):
    """Close a peak-memory window and report allocator values on every rank."""
    import torch.distributed

    from .offload import assert_runtime_idle

    platform = get_platform()
    platform.synchronize()
    assert_runtime_idle()
    allocated = platform.memory_allocated() / MIB
    peak_allocated = platform.max_memory_allocated() / MIB
    reserved = platform.memory_reserved() / MIB
    peak_reserved = platform.max_memory_reserved() / MIB
    rank = torch.distributed.get_rank()
    print(
        "[FL memory] "
        f"rank={rank} iteration={iteration} "
        f"allocated_mib={allocated:.2f} peak_allocated_mib={peak_allocated:.2f} "
        f"reserved_mib={reserved:.2f} peak_reserved_mib={peak_reserved:.2f}",
        flush=True,
    )
