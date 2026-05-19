"""Unit tests for ``pool.py`` and ``stream.py``.

The pool tests run on CPU-only boxes because the pool transparently
falls back to non-pinned allocation when ``cur_platform`` reports no
accelerator.  The stream tests skip on CPU-only platforms.
"""

import os
import sys
import unittest

import torch


sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
    ),
)


from megatron.plugin.fl_offload.config import (
    FlOffloadConfig,
    get_config,
    set_config,
)
from megatron.plugin.fl_offload.pool import (
    PinnedBufferPool,
    get_global_pool,
    reset_global_pool,
)
from megatron.plugin.fl_offload.stream import (
    get_memcpy_stream,
    has_async_streams,
    reset_streams,
)


CUDA = torch.cuda.is_available()
skip_no_cuda = unittest.skipUnless(CUDA, "CUDA accelerator not available")


class TestPinnedBufferPool(unittest.TestCase):
    def setUp(self):
        # Ensure pool behaviour does not depend on a leaked config from
        # earlier tests.
        self._orig_cfg = get_config()
        set_config(FlOffloadConfig(pin_memory=False))  # CPU-friendly default
        reset_global_pool()

    def tearDown(self):
        reset_global_pool()
        set_config(self._orig_cfg)

    def test_pin_get_returns_correct_size(self):
        pool = get_global_pool()
        buf = pool.pin_get(128)
        self.assertEqual(buf.numel(), 128)
        self.assertEqual(buf.dtype, torch.uint8)
        self.assertEqual(buf.device.type, "cpu")
        pool.pin_put(buf)

    def test_pin_get_zero_bytes_rejected(self):
        pool = get_global_pool()
        with self.assertRaises(ValueError):
            pool.pin_get(0)

    def test_put_then_get_reuses_same_object(self):
        pool = get_global_pool()
        buf1 = pool.pin_get(256)
        # Mark the storage so we can identify the same object later.
        buf1.fill_(7)
        pool.pin_put(buf1)
        buf2 = pool.pin_get(256)
        # Pool reuses the identical Tensor instance (FIFO/LIFO doesn't
        # matter here — for a single-item pool both reduce to "same").
        self.assertIs(buf1, buf2)
        self.assertEqual(int(buf2[0].item()), 7)
        pool.pin_put(buf2)

    def test_different_sizes_do_not_alias(self):
        pool = get_global_pool()
        a = pool.pin_get(256)
        b = pool.pin_get(512)
        self.assertIsNot(a, b)
        self.assertEqual(a.numel(), 256)
        self.assertEqual(b.numel(), 512)
        pool.pin_put(a)
        pool.pin_put(b)

    def test_repeated_get_put_does_not_grow_pool(self):
        # B7 in miniature: 50 iterations of (get, put) at the same size
        # must allocate exactly one buffer.
        pool = get_global_pool()
        for _ in range(50):
            buf = pool.pin_get(1024)
            pool.pin_put(buf)
        self.assertEqual(pool.peak_buffer_count(), 1)
        self.assertEqual(pool.outstanding_bytes(), 0)

    def test_peak_outstanding_bytes_tracking(self):
        pool = get_global_pool()
        a = pool.pin_get(128)
        b = pool.pin_get(256)
        self.assertEqual(pool.outstanding_bytes(), 128 + 256)
        self.assertEqual(pool.peak_pinned_bytes(), 128 + 256)

        pool.pin_put(a)
        self.assertEqual(pool.outstanding_bytes(), 256)
        # Peak doesn't shrink.
        self.assertEqual(pool.peak_pinned_bytes(), 128 + 256)

        c = pool.pin_get(64)
        # 256 + 64 == 320 < previous peak 384, so peak stays.
        self.assertEqual(pool.peak_pinned_bytes(), 384)

        pool.pin_put(b)
        pool.pin_put(c)

    def test_double_put_raises(self):
        pool = get_global_pool()
        buf = pool.pin_get(64)
        pool.pin_put(buf)
        # A second put without a matching get should drive outstanding
        # bytes negative and raise.
        with self.assertRaises(RuntimeError):
            pool.pin_put(buf)

    def test_pin_put_rejects_wrong_type(self):
        pool = get_global_pool()
        with self.assertRaises(TypeError):
            pool.pin_put("not a tensor")

    def test_reset_global_pool_replaces_singleton(self):
        first = get_global_pool()
        _ = first.pin_get(64)  # never returned — leaks in the old pool
        reset_global_pool()
        second = get_global_pool()
        self.assertIsNot(first, second)
        self.assertEqual(second.peak_buffer_count(), 0)
        self.assertEqual(second.outstanding_bytes(), 0)

    @skip_no_cuda
    def test_uses_pinned_memory_when_enabled(self):
        set_config(FlOffloadConfig(pin_memory=True))
        reset_global_pool()
        pool = get_global_pool()
        buf = pool.pin_get(512)
        # is_pinned is the canonical PyTorch check.
        self.assertTrue(buf.is_pinned())
        pool.pin_put(buf)


class TestStreams(unittest.TestCase):
    def setUp(self):
        reset_streams()

    def tearDown(self):
        reset_streams()

    def test_invalid_key_raises(self):
        with self.assertRaises(ValueError):
            get_memcpy_stream("garbage")

    def test_keys_return_distinct_streams(self):
        offload = get_memcpy_stream("offload")
        onload = get_memcpy_stream("onload")
        if CUDA:
            self.assertIsNotNone(offload)
            self.assertIsNotNone(onload)
            self.assertIsNot(offload, onload)
        else:
            # On CPU-only, both are None — that's the documented fallback.
            self.assertIsNone(offload)
            self.assertIsNone(onload)

    def test_same_key_returns_same_stream(self):
        a = get_memcpy_stream("offload")
        b = get_memcpy_stream("offload")
        self.assertIs(a, b)

    def test_reset_streams_clears_cache(self):
        first = get_memcpy_stream("offload")
        reset_streams()
        second = get_memcpy_stream("offload")
        if CUDA:
            self.assertIsNot(first, second)
        else:
            self.assertIsNone(first)
            self.assertIsNone(second)

    def test_has_async_streams_matches_platform(self):
        # On CUDA boxes async streams exist; otherwise not.
        self.assertEqual(has_async_streams(), CUDA)


if __name__ == "__main__":
    unittest.main()
