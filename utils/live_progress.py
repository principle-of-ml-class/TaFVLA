"""Small, line-oriented initialization events for the trainctl terminal monitor."""

import json
import os
import resource
import sys
import time


PREFIX = "@@TAF_PROGRESS@@"


def emit(stage: str, status: str = "Running", *, current: int | None = None,
         total: int | None = None, detail: str = "") -> None:
    if os.environ.get("TAF_TERMINAL_MONITOR") != "1":
        return
    import torch

    with open("/proc/self/statm", encoding="ascii") as status:
        pages = int(status.read().split()[1])
    event = {
        "stage": stage, "status": status, "current": current, "total": total,
        "detail": detail, "time": time.time(),
        "rss_gib": pages * os.sysconf("SC_PAGE_SIZE") / 2**30,
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
    }
    if torch.cuda.is_initialized():
        event["gpu_allocated_gib"] = torch.cuda.memory_allocated() / 2**30
        event["gpu_reserved_gib"] = torch.cuda.memory_reserved() / 2**30
    print(PREFIX + json.dumps(event, ensure_ascii=False), file=sys.stdout, flush=True)
