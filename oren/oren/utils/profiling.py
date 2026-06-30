import math
import os
import resource
import sys
import threading
import time
from typing import ClassVar

import torch
from tqdm import tqdm

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore[assignment]

# `resource.getrusage(RUSAGE_SELF).ru_maxrss` is in KiB on Linux and bytes on macOS. Normalize to bytes everywhere.
_MAXRSS_TO_BYTES = 1 if sys.platform == "darwin" else 1024


class TimerRecords:
    """Global accumulator of timer measurements, keyed by message label.

    Mirrors `erl::common::BlockTimerRecords`: every `CpuTimer` / `GpuTimer` instance with
    `record=True` appends to this registry on exit (after any configured warmup).
    """

    _records: ClassVar[dict[str, dict[str, float]]] = {}
    _labels: ClassVar[list[str]] = []
    _max_label_length: ClassVar[int] = 0
    _lock: ClassVar[threading.Lock] = threading.Lock()

    @classmethod
    def add_record(cls, label: str, duration_s: float) -> None:
        """Append a single measurement for `label`. `duration_s` is in seconds.

        Tracks sum, sum-of-squares, and running max so `export_records` can report a per-label population
        standard deviation and worst-case duration without retaining the raw sample list (memory stays O(1)
        per label).
        """
        with cls._lock:
            if len(label) > cls._max_label_length:
                cls._max_label_length = len(label)
            record = cls._records.get(label)
            if record is None:
                record = {"current": 0.0, "total": 0.0, "total_sq": 0.0, "max": 0.0, "count": 0}
                cls._records[label] = record
                cls._labels.append(label)
                cls._labels.sort()
            record["current"] = duration_s
            record["total"] += duration_s
            record["total_sq"] += duration_s * duration_s
            if duration_s > record["max"]:
                record["max"] = duration_s
            record["count"] += 1

    @classmethod
    def print_records(cls) -> None:
        """Print a formatted summary of all recorded timers. Durations shown in milliseconds."""
        with cls._lock:
            if not cls._labels:
                tqdm.write("Timer Records: (empty)")
                return
            label_w = cls._max_label_length + 2
            lines = ["Timer Records:"]
            lines.append(
                f"{'Label':<{label_w}}"
                f"{'Current (ms)':>15}"
                f"{'Mean (ms)':>15}"
                f"{'Std (ms)':>15}"
                f"{'Max (ms)':>15}"
                f"{'Total (ms)':>15}"
                f"{'Count':>10}"
            )
            lines.append("-" * (label_w + 15 + 15 + 15 + 15 + 15 + 10))
            for label in cls._labels:
                rec = cls._records[label]
                count = rec["count"]
                if count > 0:
                    mean = rec["total"] / count
                    variance = rec["total_sq"] / count - mean * mean
                    std = math.sqrt(variance) if variance > 0 else 0.0
                else:
                    mean = 0.0
                    std = 0.0
                lines.append(
                    f"{label:<{label_w}}"
                    f"{rec['current'] * 1000.0:>15.3f}"
                    f"{mean * 1000.0:>15.3f}"
                    f"{std * 1000.0:>15.3f}"
                    f"{rec['max'] * 1000.0:>15.3f}"
                    f"{rec['total'] * 1000.0:>15.3f}"
                    f"{count:>10d}"
                )
            tqdm.write("\n".join(lines))

    @classmethod
    def export_records(cls) -> dict[str, dict[str, float | int]]:
        """Snapshot the accumulated timer registry into a plain dict.

        Returns a `{label: {"average_s", "std_s", "max_s", "total_s", "count"}}` mapping suitable for YAML /
        JSON dumping. Durations stay in seconds (matching the in-memory representation); convert to
        milliseconds at the presentation layer if needed. `std_s` is the population standard deviation derived
        from the running sum / sum-of-squares (0.0 when count < 2); `max_s` is the all-time max of any single
        record.

        Acquires `cls._lock` for the duration of the snapshot so no in-flight `add_record` tears a row mid-copy.
        """
        snapshot: dict[str, dict[str, float | int]] = {}
        with cls._lock:
            for label, rec in cls._records.items():
                count = int(rec.get("count", 0))
                total_s = float(rec.get("total", 0.0))
                total_sq_s = float(rec.get("total_sq", 0.0))
                max_s = float(rec.get("max", 0.0))
                if count > 0:
                    average_s = total_s / count
                    variance = total_sq_s / count - average_s * average_s
                    std_s = math.sqrt(variance) if variance > 0 else 0.0
                else:
                    average_s = 0.0
                    std_s = 0.0
                snapshot[label] = {
                    "average_s": average_s,
                    "std_s": std_s,
                    "max_s": max_s,
                    "total_s": total_s,
                    "count": count,
                }
        return snapshot

    @classmethod
    def clear(cls) -> None:
        """Drop all accumulated records."""
        with cls._lock:
            cls._records.clear()
            cls._labels.clear()
            cls._max_label_length = 0


def print_records() -> None:
    """Module-level shortcut for `TimerRecords.print_records()`."""
    TimerRecords.print_records()


class MemoryRecords:
    """Global accumulator of memory-profiler measurements, keyed by message label.

    Mirrors `TimerRecords` but stores per-block memory usage. For each `CpuMemoryProfiler` / `GpuMemoryProfiler`
    exit (after warmup), two values are appended:

        peak_bytes  -- maximum memory the block held above its entry baseline. For CPU, this is the rise in the
                        process's `ru_maxrss` between entry and exit (clamped to >= 0); for GPU, this is
                        `torch.cuda.max_memory_allocated() - allocated_at_entry` after resetting peak stats at
                        entry. Approximates a per-block "peak working set" without sampling threads.
        delta_bytes -- change in currently-resident / currently-allocated memory between entry and exit. Positive
                        means the block left memory behind (potential leak); negative means it freed more than it
                        allocated.

    In addition to the above (baselined) values, callers may pass `raw` -- a `{name: int}` dict of un-baselined
    end-of-block readings. The CpuMemoryProfiler passes `end_rss` / `end_maxrss`; the GpuMemoryProfiler passes
    `end_allocated` / `end_peak`. Each named reading accumulates an independent (max, mean, std, count) so a
    label tracked by both profilers (typical: "train with frame", entered under both `with` clauses) ends up with
    four raw-stat entries side by side. Use these to recover total process memory usage -- the baselined
    peak/delta only describe the block's incremental footprint above whatever was already allocated at entry.

    Sizes are kept in bytes in memory; convert to MiB / GiB at the presentation layer.
    """

    # Record value mixes scalar counters with a nested `raw_stats` dict, so we use a loose `Any` here -- the
    # nested-dict access pattern is intentional and tracked by the docstring above instead of by the type.
    _records: ClassVar[dict[str, dict]] = {}
    _labels: ClassVar[list[str]] = []
    _max_label_length: ClassVar[int] = 0
    _lock: ClassVar[threading.Lock] = threading.Lock()

    @classmethod
    def add_record(
        cls,
        label: str,
        peak_bytes: int,
        delta_bytes: int,
        raw: dict[str, int] | None = None,
    ) -> None:
        """Append a single (peak, delta) measurement for `label`, and optionally a dict of raw end-of-block readings.

        Args:
            label: record key (matches the profiler's `message`).
            peak_bytes: baselined peak (block exit peak minus block entry allocated, clamped >= 0).
            delta_bytes: net change in resident / allocated bytes across the block.
            raw: optional `{reading_name: bytes}` of un-baselined absolute readings -- typically `end_rss` /
                `end_maxrss` from CpuMemoryProfiler and `end_allocated` / `end_peak` from GpuMemoryProfiler.
                Each reading accumulates its own (max, sum, sum-of-squares, count) so the snapshot exporter can
                report per-reading max / mean / std.
        """
        with cls._lock:
            if len(label) > cls._max_label_length:
                cls._max_label_length = len(label)
            record = cls._records.get(label)
            if record is None:
                # Sum-of-squares and per-axis max are tracked so the snapshot exporter can report mean / std /
                # max for both `peak_bytes` and `delta_bytes` without retaining the raw sample list. `count` is
                # the same denominator for both axes.
                record = {
                    "current_peak": 0,
                    "current_delta": 0,
                    "max_peak": 0,
                    "max_delta": 0,
                    "total_peak": 0,
                    "total_delta": 0,
                    "total_sq_peak": 0.0,
                    "total_sq_delta": 0.0,
                    "count": 0,
                    "raw_stats": {},
                }
                cls._records[label] = record
                cls._labels.append(label)
                cls._labels.sort()
            record["current_peak"] = peak_bytes
            record["current_delta"] = delta_bytes
            if peak_bytes > record["max_peak"]:
                record["max_peak"] = peak_bytes
            if delta_bytes > record["max_delta"]:
                record["max_delta"] = delta_bytes
            record["total_peak"] += peak_bytes
            record["total_delta"] += delta_bytes
            # peak_bytes is clamped >= 0; delta_bytes can be negative but its square is always >= 0, so the
            # variance computation in export_records is well-defined for both.
            record["total_sq_peak"] += float(peak_bytes) * float(peak_bytes)
            record["total_sq_delta"] += float(delta_bytes) * float(delta_bytes)
            record["count"] = record["count"] + 1
            if raw:
                raw_stats = record["raw_stats"]
                for name, value in raw.items():
                    v = float(value)
                    entry = raw_stats.get(name)
                    if entry is None:
                        entry = {"max": 0.0, "total": 0.0, "total_sq": 0.0, "count": 0}
                        raw_stats[name] = entry
                    if v > entry["max"]:
                        entry["max"] = v
                    entry["total"] += v
                    entry["total_sq"] += v * v
                    entry["count"] += 1

    @classmethod
    def print_records(cls) -> None:
        """Print a formatted summary of all recorded memory blocks. Sizes shown in MiB."""
        with cls._lock:
            if not cls._labels:
                tqdm.write("Memory Records: (empty)")
                return
            label_w = cls._max_label_length + 2
            mib = 1024.0 * 1024.0
            lines = ["Memory Records:"]
            lines.append(
                f"{'Label':<{label_w}}"
                f"{'Peak (MiB)':>14}"
                f"{'Avg Peak (MiB)':>18}"
                f"{'Std Peak (MiB)':>18}"
                f"{'Max Peak (MiB)':>18}"
                f"{'Avg Delta (MiB)':>18}"
                f"{'Std Delta (MiB)':>18}"
                f"{'Max Delta (MiB)':>18}"
                f"{'Count':>10}"
            )
            lines.append("-" * (label_w + 14 + 18 * 6 + 10))
            for label in cls._labels:
                rec = cls._records[label]
                count = rec["count"]
                if count > 0:
                    avg_peak = rec["total_peak"] / count
                    var_peak = rec["total_sq_peak"] / count - avg_peak * avg_peak
                    std_peak = math.sqrt(var_peak) if var_peak > 0 else 0.0
                    avg_delta = rec["total_delta"] / count
                    var_delta = rec["total_sq_delta"] / count - avg_delta * avg_delta
                    std_delta = math.sqrt(var_delta) if var_delta > 0 else 0.0
                else:
                    avg_peak = std_peak = avg_delta = std_delta = 0.0
                lines.append(
                    f"{label:<{label_w}}"
                    f"{rec['current_peak'] / mib:>14.3f}"
                    f"{avg_peak / mib:>18.3f}"
                    f"{std_peak / mib:>18.3f}"
                    f"{rec['max_peak'] / mib:>18.3f}"
                    f"{avg_delta / mib:>18.3f}"
                    f"{std_delta / mib:>18.3f}"
                    f"{rec['max_delta'] / mib:>18.3f}"
                    f"{count:>10d}"
                )
            tqdm.write("\n".join(lines))

    @classmethod
    def export_records(cls) -> dict[str, dict[str, float | int]]:
        """Snapshot the memory record table into a plain dict for YAML / JSON dumping.

        Returns a `{label: {peak_bytes, average_peak_bytes, std_peak_bytes, max_peak_bytes,
        total_delta_bytes, average_delta_bytes, std_delta_bytes, max_delta_bytes, count, raw_stats}}` mapping.
        `peak_bytes` is the most recent baselined peak; `max_peak_bytes` is the all-time-high baselined peak.
        Standard deviations are population-variance derived (`E[X^2] - E[X]^2`, sqrt with FP-rounding guard);
        they are 0.0 when count < 2. `raw_stats[name] = {max_bytes, mean_bytes, std_bytes, count}` is the
        un-baselined end-of-block distribution per named reading (e.g. `end_rss`, `end_maxrss` from CPU and
        `end_allocated`, `end_peak` from GPU).

        Acquires `cls._lock` for the duration of the snapshot so no in-flight `add_record` tears a row mid-copy.
        """
        snapshot: dict[str, dict[str, float | int]] = {}
        with cls._lock:
            for label, rec in cls._records.items():
                count = int(rec.get("count", 0))
                total_peak = float(rec.get("total_peak", 0))
                total_delta = float(rec.get("total_delta", 0))
                total_sq_peak = float(rec.get("total_sq_peak", 0.0))
                total_sq_delta = float(rec.get("total_sq_delta", 0.0))
                if count > 0:
                    avg_peak = total_peak / count
                    var_peak = total_sq_peak / count - avg_peak * avg_peak
                    std_peak = math.sqrt(var_peak) if var_peak > 0 else 0.0
                    avg_delta = total_delta / count
                    var_delta = total_sq_delta / count - avg_delta * avg_delta
                    std_delta = math.sqrt(var_delta) if var_delta > 0 else 0.0
                else:
                    avg_peak = std_peak = avg_delta = std_delta = 0.0
                entry: dict = {
                    "peak_bytes": float(rec.get("current_peak", 0)),
                    "average_peak_bytes": avg_peak,
                    "std_peak_bytes": std_peak,
                    "max_peak_bytes": float(rec.get("max_peak", 0)),
                    "total_delta_bytes": total_delta,
                    "average_delta_bytes": avg_delta,
                    "std_delta_bytes": std_delta,
                    "max_delta_bytes": float(rec.get("max_delta", 0)),
                    "count": count,
                }
                raw_stats = rec.get("raw_stats", {}) or {}
                if raw_stats:
                    raw_snap: dict[str, dict[str, float | int]] = {}
                    for name, st in raw_stats.items():
                        c = int(st.get("count", 0))
                        s = float(st.get("total", 0.0))
                        sq = float(st.get("total_sq", 0.0))
                        mx = float(st.get("max", 0.0))
                        mean = s / c if c > 0 else 0.0
                        # population variance via E[X^2] - E[X]^2; guard against tiny negative from FP rounding.
                        variance = (sq / c) - (mean * mean) if c > 0 else 0.0
                        std = math.sqrt(variance) if variance > 0 else 0.0
                        raw_snap[name] = {
                            "max_bytes": mx,
                            "mean_bytes": mean,
                            "std_bytes": std,
                            "count": c,
                        }
                    entry["raw_stats"] = raw_snap
                snapshot[label] = entry
        return snapshot

    @classmethod
    def clear(cls) -> None:
        """Drop all accumulated memory records."""
        with cls._lock:
            cls._records.clear()
            cls._labels.clear()
            cls._max_label_length = 0


def print_memory_records() -> None:
    """Module-level shortcut for `MemoryRecords.print_records()`."""
    MemoryRecords.print_records()


def _process_rss_bytes() -> int:
    """Return the current resident set size of this process in bytes.

    Prefers `psutil.Process().memory_info().rss` when psutil is available; falls back to parsing
    `/proc/self/statm` (resident pages * page size). Returns 0 on platforms with neither (e.g. Windows without
    psutil).
    """
    if psutil is not None:
        return int(psutil.Process().memory_info().rss)
    try:
        with open("/proc/self/statm", "r") as f:
            pages = int(f.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return 0


def _process_maxrss_bytes() -> int:
    """Return the peak resident set size of this process so far, normalized to bytes across Linux / macOS.

    Backed by `resource.getrusage(RUSAGE_SELF).ru_maxrss`. The OS reports this in KiB on Linux and bytes on macOS;
    `_MAXRSS_TO_BYTES` handles the conversion. Monotonically non-decreasing over the process lifetime, so the
    difference between two readings is the peak growth attributable to whatever ran in between.
    """
    try:
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * _MAXRSS_TO_BYTES
    except (AttributeError, OSError):
        return 0


class CpuTimer:

    def __init__(
        self,
        message,
        warmup: int = 0,
        enable: bool = True,
        verbose: bool = True,
        record: bool = True,
    ):
        """Configure a CPU wall-clock timer context manager.

        Args:
            message: Label printed alongside the measured time.
            warmup: Number of initial enter/exit cycles to skip when computing the running average.
            enable: When False, the context manager is a no-op.
            verbose: When True, print the timing summary on every successful exit.
            record: When True, append each post-warmup measurement to `TimerRecords` keyed by `message`.
        """
        self.message = message
        self.warmup = warmup
        self.enable = enable
        self.verbose = verbose
        self.record = record
        self.cnt = 0
        self.t = 0
        self.average_t = 0
        self._total_t = 0
        self.total_t = 0

    def __enter__(self):
        if not self.enable:
            return self
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        if self.cnt < self.warmup:
            self.cnt += 1
            return
        self.cnt += 1
        self.t = self.end - self.start
        self._total_t += self.t
        self.average_t = self._total_t / (self.cnt - self.warmup)
        self.total_t = self.average_t * self.cnt
        if self.record:
            TimerRecords.add_record(self.message, self.t)
        if self.verbose:
            tqdm.write(f"{self.message}: {self.t:.6f}(cur)/{self.average_t:.6f}(avg)/{self.total_t:.6f}(total) seconds")


def cpu_timer(message, warmup=0, enable=True, verbose=True, record=True):
    """Decorator factory that wraps the decorated function in a `CpuTimer` context manager.

    Args:
        message: Label passed to the underlying `CpuTimer`.
        warmup: Number of initial calls to ignore when averaging.
        enable: When False, timing is disabled and the function runs untouched.
        verbose: When True, print a per-call summary line.
        record: When True, append each post-warmup measurement to `TimerRecords`.

    Returns:
        A decorator that returns a wrapped function preserving the original return value.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            with CpuTimer(message, warmup=warmup, enable=enable, verbose=verbose, record=record):
                return func(*args, **kwargs)

        return wrapper

    return decorator


class GpuTimer:

    def __init__(
        self,
        message,
        warmup: int = 0,
        enable: bool = True,
        verbose: bool = True,
        record: bool = True,
    ):
        """Configure a CUDA-event-based GPU timer context manager.

        Args:
            message: Label printed alongside the measured time.
            warmup: Number of initial enter/exit cycles to skip when computing the running average.
            enable: When False, the context manager is a no-op.
            verbose: When True, print the timing summary on every successful exit.
            record: When True, append each post-warmup measurement to `TimerRecords` keyed by `message`.
        """
        self.message = message
        self.warmup = warmup
        self.enable = enable
        self.verbose = verbose
        self.record = record
        self.cnt = 0
        self.t = 0
        self.average_t = 0
        self._total_t = 0
        self.total_t = 0

    def __enter__(self):
        if not self.enable:
            return self
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.start.record()
        return self

    def __exit__(self, *args):
        if not self.enable:
            return
        self.end.record()
        torch.cuda.synchronize()
        self.t = self.start.elapsed_time(self.end) / 1e3
        if self.cnt < self.warmup:
            self.cnt += 1
            return
        self.cnt += 1
        self._total_t += self.t
        self.average_t = self._total_t / (self.cnt - self.warmup)
        self.total_t = self.average_t * self.cnt
        if self.record:
            TimerRecords.add_record(self.message, self.t)
        if self.verbose:
            tqdm.write(f"{self.message}: {self.t:.6f}(cur)/{self.average_t:.6f}(avg)/{self.total_t:.6f}(total) seconds")


def gpu_timer(message, warmup=0, enable=True, verbose=True, record=True):
    """Decorator factory that wraps the decorated function in a `GpuTimer` context manager.

    Args:
        message: Label passed to the underlying `GpuTimer`.
        warmup: Number of initial calls to ignore when averaging.
        enable: When False, timing is disabled and the function runs untouched.
        verbose: When True, print a per-call summary line.
        record: When True, append each post-warmup measurement to `TimerRecords`.

    Returns:
        A decorator that returns a wrapped function preserving the original return value.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            with GpuTimer(message, warmup=warmup, enable=enable, verbose=verbose, record=record):
                return func(*args, **kwargs)

        return wrapper

    return decorator


class CpuMemoryProfiler:

    def __init__(
        self,
        message,
        warmup: int = 0,
        enable: bool = True,
        verbose: bool = True,
        record: bool = True,
    ):
        """Configure a CPU memory profiler context manager.

        Tracks two values per invocation, both in bytes:
            peak_bytes  -- `max(0, ru_maxrss_at_exit - ru_maxrss_at_entry)`. `ru_maxrss` is monotonic per process,
                            so the difference captures the peak growth caused by this block (0 if the block did not
                            push the process to a new high).
            delta_bytes -- current RSS at exit minus current RSS at entry (positive = memory not yet freed,
                            negative = freed).

        Both metrics include all process memory (Python objects, numpy arrays, torch CPU tensors, native
        libraries) -- not just Python-side allocations.

        Nesting is safe: each profiler samples OS counters independently.

        Args:
            message: Label printed alongside the measured memory.
            warmup: Number of initial enter/exit cycles to skip when computing the running average.
            enable: When False, the context manager is a no-op.
            verbose: When True, print the memory summary on every successful exit.
            record: When True, append each post-warmup measurement to `MemoryRecords` keyed by `message`.
        """
        self.message = message
        self.warmup = warmup
        self.enable = enable
        self.verbose = verbose
        self.record = record
        self.cnt = 0
        self.peak_bytes = 0
        self.delta_bytes = 0
        self._average_peak = 0.0
        self._average_delta = 0.0
        self._total_peak = 0
        self._total_delta = 0
        self._start_rss = 0
        self._start_maxrss = 0

    def __enter__(self):
        if not self.enable:
            return self
        self._start_rss = _process_rss_bytes()
        self._start_maxrss = _process_maxrss_bytes()
        return self

    def __exit__(self, *args):
        if not self.enable:
            return
        end_rss = _process_rss_bytes()
        end_maxrss = _process_maxrss_bytes()
        if self.cnt < self.warmup:
            self.cnt += 1
            return
        self.cnt += 1
        self.peak_bytes = max(0, end_maxrss - self._start_maxrss)
        self.delta_bytes = end_rss - self._start_rss
        self._total_peak += self.peak_bytes
        self._total_delta += self.delta_bytes
        denom = self.cnt - self.warmup
        self._average_peak = self._total_peak / denom
        self._average_delta = self._total_delta / denom
        if self.record:
            MemoryRecords.add_record(
                self.message,
                self.peak_bytes,
                self.delta_bytes,
                raw={"end_rss": end_rss, "end_maxrss": end_maxrss},
            )
        if self.verbose:
            mib = 1024.0 * 1024.0
            tqdm.write(
                f"{self.message} [cpu]: peak +{self.peak_bytes / mib:.3f}(cur)/"
                f"{self._average_peak / mib:.3f}(avg) MiB, "
                f"delta {self.delta_bytes / mib:+.3f}(cur)/{self._average_delta / mib:+.3f}(avg) MiB"
            )


def cpu_memory_profiler(message, warmup=0, enable=True, verbose=True, record=True):
    """Decorator factory that wraps the decorated function in a `CpuMemoryProfiler` context manager.

    Args:
        message: Label passed to the underlying `CpuMemoryProfiler`.
        warmup: Number of initial calls to ignore when averaging.
        enable: When False, profiling is disabled and the function runs untouched.
        verbose: When True, print a per-call summary line.
        record: When True, append each post-warmup measurement to `MemoryRecords`.

    Returns:
        A decorator that returns a wrapped function preserving the original return value.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            with CpuMemoryProfiler(message, warmup=warmup, enable=enable, verbose=verbose, record=record):
                return func(*args, **kwargs)

        return wrapper

    return decorator


class GpuMemoryProfiler:

    def __init__(
        self,
        message,
        device: int | torch.device | None = None,
        warmup: int = 0,
        enable: bool = True,
        verbose: bool = True,
        record: bool = True,
    ):
        """Configure a CUDA memory profiler context manager.

        Tracks two values per invocation, both in bytes, read from the torch caching allocator after a device
        synchronize at exit:
            peak_bytes  -- `max_memory_allocated() - allocated_at_entry`. Peak stats are reset on `__enter__` so the
                            value reflects allocations made by this block alone.
            delta_bytes -- currently-allocated bytes at exit minus at entry (positive = tensors still alive,
                            negative = block net-freed).

        Caveat -- `torch.cuda` keeps one peak counter per device, so two `GpuMemoryProfiler` instances must NOT be
        nested on the same device: the inner profiler resets the peak and the outer reading becomes meaningless.
        Multiple sibling (non-overlapping) profilers, or profilers on different devices, compose fine.

        Args:
            message: Label printed alongside the measured memory.
            device: CUDA device index or `torch.device`; `None` means the current device.
            warmup: Number of initial enter/exit cycles to skip when computing the running average.
            enable: When False (or CUDA is unavailable), the context manager is a no-op.
            verbose: When True, print the memory summary on every successful exit.
            record: When True, append each post-warmup measurement to `MemoryRecords` keyed by `message`.
        """
        self.message = message
        self.device = device
        self.warmup = warmup
        self.enable = enable and torch.cuda.is_available()
        self.verbose = verbose
        self.record = record
        self.cnt = 0
        self.peak_bytes = 0
        self.delta_bytes = 0
        self._average_peak = 0.0
        self._average_delta = 0.0
        self._total_peak = 0
        self._total_delta = 0
        self._start_allocated = 0

    def __enter__(self):
        if not self.enable:
            return self
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        self._start_allocated = int(torch.cuda.memory_allocated(self.device))
        return self

    def __exit__(self, *args):
        if not self.enable:
            return
        torch.cuda.synchronize(self.device)
        end_allocated = int(torch.cuda.memory_allocated(self.device))
        end_peak = int(torch.cuda.max_memory_allocated(self.device))
        if self.cnt < self.warmup:
            self.cnt += 1
            return
        self.cnt += 1
        self.peak_bytes = max(0, end_peak - self._start_allocated)
        self.delta_bytes = end_allocated - self._start_allocated
        self._total_peak += self.peak_bytes
        self._total_delta += self.delta_bytes
        denom = self.cnt - self.warmup
        self._average_peak = self._total_peak / denom
        self._average_delta = self._total_delta / denom
        if self.record:
            MemoryRecords.add_record(
                self.message,
                self.peak_bytes,
                self.delta_bytes,
                raw={"end_allocated": end_allocated, "end_peak": end_peak},
            )
        if self.verbose:
            mib = 1024.0 * 1024.0
            tqdm.write(
                f"{self.message} [gpu]: peak +{self.peak_bytes / mib:.3f}(cur)/"
                f"{self._average_peak / mib:.3f}(avg) MiB, "
                f"delta {self.delta_bytes / mib:+.3f}(cur)/{self._average_delta / mib:+.3f}(avg) MiB"
            )


def gpu_memory_profiler(message, device=None, warmup=0, enable=True, verbose=True, record=True):
    """Decorator factory that wraps the decorated function in a `GpuMemoryProfiler` context manager.

    Args:
        message: Label passed to the underlying `GpuMemoryProfiler`.
        device: CUDA device index or `torch.device`; `None` means the current device.
        warmup: Number of initial calls to ignore when averaging.
        enable: When False, profiling is disabled and the function runs untouched.
        verbose: When True, print a per-call summary line.
        record: When True, append each post-warmup measurement to `MemoryRecords`.

    Returns:
        A decorator that returns a wrapped function preserving the original return value.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            with GpuMemoryProfiler(
                message, device=device, warmup=warmup, enable=enable, verbose=verbose, record=record
            ):
                return func(*args, **kwargs)

        return wrapper

    return decorator
