"""Small, line-oriented initialization events for the trainctl terminal monitor."""

import json
import os
import resource
import sys
import time


PREFIX = "@@TAF_PROGRESS@@"
_PLAIN_SAMPLER_STATE: dict[str, tuple[str, float]] = {}


def emit(stage: str, status: str = "Running", *, current: int | None = None,
         total: int | None = None, detail: str = "", phase: str = "") -> None:
    if os.environ.get("TAF_TERMINAL_MONITOR") != "1":
        # Plain trainctl logs still expose sampler startup; no JSON/ANSI spam.
        if stage in {"Training Sampler Initialization", "Validation Sampler Initialization"}:
            now = time.monotonic()
            previous_phase, previous_time = _PLAIN_SAMPLER_STATE.get(stage, ("", -float("inf")))
            if (status != "Running" or phase != previous_phase or now - previous_time >= 5
                    or (current is not None and current == total)):
                count = f"{current:,}/{total:,}" if current is not None and total is not None else ""
                print(f"[{stage}] {status} {phase} {count} {detail}".strip(), flush=True)
                _PLAIN_SAMPLER_STATE[stage] = (phase, now)
        return
    import torch

    with open("/proc/self/statm", encoding="ascii") as statm_file:
        pages = int(statm_file.read().split()[1])
    event = {
        "stage": stage, "status": status, "current": current, "total": total,
        "detail": detail, "phase": phase, "time": time.time(),
        "rss_gib": pages * os.sysconf("SC_PAGE_SIZE") / 2**30,
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
    }
    if torch.cuda.is_initialized():
        event["gpu_allocated_gib"] = torch.cuda.memory_allocated() / 2**30
        event["gpu_reserved_gib"] = torch.cuda.memory_reserved() / 2**30
    print(PREFIX + json.dumps(event, ensure_ascii=False), file=sys.stdout, flush=True)
