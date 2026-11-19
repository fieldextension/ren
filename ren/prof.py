import atexit
import contextlib
import decimal
import json
import tempfile
import time
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any, Generator

from ren.context import PROFILE, NVTX


def profiling_enabled() -> bool:
    """Return True when profiling is enabled via the PROFILE context var."""
    return bool(PROFILE.value)


def _nvtx():
    try:
        return import_module("nvtx")
    except ModuleNotFoundError as exc:
        if exc.name != "nvtx":
            raise
        raise ImportError("NVTX tracing requires the ren[profiling] extra") from exc


def _nvtx_ctx(name: str):
    return _nvtx().annotate(name) if NVTX else contextlib.nullcontext()


def _perf_counter_us() -> decimal.Decimal:
    return decimal.Decimal(time.perf_counter_ns()) / 1000


class ProfileEvent:
    pass


@dataclass
class ProfileRangeEvent(ProfileEvent):
    device: str
    name: str
    st: decimal.Decimal
    en: decimal.Decimal | None = None
    args: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProfilePointEvent(ProfileEvent):
    device: str
    name: str
    ts: decimal.Decimal = field(default_factory=_perf_counter_us)
    arg: dict[str, Any] = field(default_factory=dict)


cpu_events: list[ProfileEvent] = []


@contextlib.contextmanager
def trace(name: str, device: str = "REN", **args) -> Generator[ProfileRangeEvent | None, None, None]:
    """
    Record a timed span if profiling is enabled. When disabled, this is a no-op.

    If NVTX=1, also emits NVTX markers for nsys/Nsight visualization.
    """
    ev = None
    if profiling_enabled():
        ev = ProfileRangeEvent(device=device, name=name, st=_perf_counter_us(), args=args)

    with _nvtx_ctx(name):
        try:
            yield ev
        finally:
            if ev is not None:
                ev.en = _perf_counter_us()
                cpu_events.append(ev)


def profile_marker(name: str, device: str = "REN", arg: dict[str, Any] | None = None) -> None:
    """Emit a point-in-time marker when profiling is enabled."""
    if NVTX:
        _nvtx().mark(name)
    if not profiling_enabled():
        return
    cpu_events.append(ProfilePointEvent(device=device, name=name, arg=arg or {}))


def clear_events() -> None:
    """Clear all collected profiling events."""
    cpu_events.clear()


def print_summary() -> None:
    """Print aggregated timing summary grouped by event name."""
    from collections import defaultdict

    stats: dict[str, list[float]] = defaultdict(list)
    for ev in cpu_events:
        if isinstance(ev, ProfileRangeEvent) and ev.en is not None:
            dur_ms = float(ev.en - ev.st) / 1000  # us -> ms
            stats[ev.name].append(dur_ms)

    if not stats:
        print("[ren.prof] No profiling events collected")
        return

    print(f"\n{'Event':<40} {'Count':>8} {'Total':>10} {'Min':>10} {'Max':>10} {'Avg':>10}")
    print("-" * 90)

    # Sort by total time descending
    sorted_stats = sorted(stats.items(), key=lambda x: sum(x[1]), reverse=True)

    for name, times in sorted_stats:
        count = len(times)
        total = sum(times)
        min_t = min(times)
        max_t = max(times)
        avg_t = total / count
        print(f"{name:<40} {count:>8} {total:>9.3f}ms {min_t:>9.3f}ms {max_t:>9.3f}ms {avg_t:>9.3f}ms")

    print()


def export_chrome_trace(path: str | Path | None = None) -> Path:
    """
    Export collected events to a Chrome Trace / Perfetto compatible JSON file.
    """
    if path is None:
        path = Path(tempfile.gettempdir()) / "ren_profile.json"
    else:
        path = Path(path)

    device_to_tid: dict[str, int] = {}

    def _tid(dev: str) -> int:
        tid = device_to_tid.get(dev)
        if tid is None:
            tid = len(device_to_tid)
            device_to_tid[dev] = tid
        return tid

    trace_events: list[dict[str, Any]] = []
    for ev in cpu_events:
        if isinstance(ev, ProfileRangeEvent):
            if ev.en is None:
                continue
            trace_events.append({
                "name": ev.name,
                "cat": ev.device,
                "ph": "X",
                "ts": float(ev.st),
                "dur": float(ev.en - ev.st),
                "pid": 0,
                "tid": _tid(ev.device),
                "args": ev.args,
            })
        elif isinstance(ev, ProfilePointEvent):
            trace_events.append({
                "name": ev.name,
                "cat": ev.device,
                "ph": "i",
                "s": "g",
                "ts": float(ev.ts),
                "pid": 0,
                "tid": _tid(ev.device),
                "args": ev.arg,
            })

    path.write_text(json.dumps({"traceEvents": trace_events}, indent=2))
    return path


def finalize_profile() -> None:
    """Flush collected events at process exit, if any were collected."""
    if not cpu_events:
        return
    try:
        out = export_chrome_trace()
        print(f"[ren.prof] profile written to {out}", flush=True)
    except Exception as exc:  # pragma: no cover - best-effort
        print(f"[ren.prof] failed to write profile: {exc}", flush=True)


atexit.register(finalize_profile)


__all__ = [
    "trace",
    "profile_marker",
    "profiling_enabled",
    "clear_events",
    "print_summary",
    "ProfileEvent",
    "ProfileRangeEvent",
    "ProfilePointEvent",
    "export_chrome_trace",
    "finalize_profile",
]
