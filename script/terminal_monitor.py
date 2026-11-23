"""Display the exact trainctl run log and TaFVLA metrics while trainctl supervises it."""

from __future__ import annotations

import argparse
import json
import os
import re
import selectors
import signal
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

from rich.console import Console, Group
from rich.live import Live
from rich.table import Table

from utils.live_progress import PREFIX


STAGES = (
    "Dataset Initialization", "Temporal Alignment Audit", "Manifest Loading & Validation",
    "Normalization", "Model / Optimizer Initialization", "Training Sampler Initialization",
    "Validation Sampler Initialization", "W&B / trainctl Initialization", "Training Ready",
)


def _duration(seconds: float) -> str:
    return time.strftime("%H:%M:%S", time.gmtime(max(0, seconds)))


def _descendants(pid: int) -> set[int]:
    found: set[int] = set()
    pending = [pid]
    while pending:
        parent = pending.pop()
        try:
            children = Path(f"/proc/{parent}/task/{parent}/children").read_text().split()
        except OSError:
            continue
        for child in map(int, children):
            if child not in found:
                found.add(child)
                pending.append(child)
    return found


def _stop_tree(process: subprocess.Popen, signum: int) -> None:
    # trainctl starts its supervised child in a separate session. Capture its
    # descendants before stopping trainctl, otherwise that session is orphaned.
    groups = set()
    for pid in _descendants(process.pid):
        try:
            groups.add(os.getpgid(pid))
        except ProcessLookupError:
            pass
    groups.discard(process.pid)
    for group in groups:
        try:
            os.killpg(group, signal.SIGTERM)
        except ProcessLookupError:
            pass
    # Let trainctl observe the child's exit and send its normal final status.
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
    groups.add(process.pid)
    for group in groups:
        try:
            os.killpg(group, 0)
        except ProcessLookupError:
            continue
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            pass


class Monitor:
    def __init__(self, metrics_path: Path, total_steps: int, *, tty: bool | None = None):
        self.metrics_path = metrics_path
        self.metrics_offset = metrics_path.stat().st_size if metrics_path.exists() else 0
        self.total_steps = total_steps
        self.tty = sys.stdout.isatty() if tty is None else tty
        self.console = Console(force_terminal=self.tty)
        self.stages: dict[str, dict] = {}
        self.metrics: dict = {}
        self.recent = deque(maxlen=4)
        self.log_path: Path | None = None
        self.log_offset = 0
        self.run_id: str | None = None
        self.started = time.monotonic()
        self.last_refresh = 0.0
        self.phase = "Initializing"

    def _render(self):
        table = Table(title=f"TaFVLA  |  {self.phase}", expand=True)
        for name in ("Stage", "Status", "Progress", "Elapsed", "ETA", "CPU RSS / peak", "GPU alloc / reserved"):
            table.add_column(name, no_wrap=True)
        now = time.time()
        for name in STAGES:
            item = self.stages.get(name, {})
            status = item.get("status", "Pending")
            current, total = item.get("current"), item.get("total")
            progress = f"{current}/{total}" if current is not None and total else (str(current) if current is not None else "—")
            elapsed = now - item.get("time", now) if status == "Running" else item.get("elapsed", 0)
            eta = elapsed * (total - current) / current if status == "Running" and current and total else None
            cpu = (f"{item['rss_gib']:.2f} / {item['peak_rss_gib']:.2f} GiB"
                   if "rss_gib" in item else "—")
            gpu = (f"{item['gpu_allocated_gib']:.2f} / {item['gpu_reserved_gib']:.2f} GiB"
                   if "gpu_allocated_gib" in item else "—")
            table.add_row(name, status, progress, _duration(elapsed) if status != "Pending" else "—",
                          _duration(eta) if eta is not None else "—", cpu, gpu)
        m = self.metrics
        training = Table(title="Training", expand=True)
        training.add_column("Step"); training.add_column("Loss"); training.add_column("Val loss")
        training.add_column("Retrieval t2f / f2t R@1"); training.add_column("Throughput")
        training.add_column("Elapsed / ETA"); training.add_column("CPU RSS / peak"); training.add_column("GPU alloc / reserved")
        def value(key):
            raw = m.get(key)
            return f"{raw:.4g}" if isinstance(raw, (int, float)) else "—"
        step = int(m.get("step", 0))
        elapsed = time.monotonic() - self.started
        rate = m.get("perf/steps_per_second", 0)
        eta = (self.total_steps - step) / rate if rate else None
        training.add_row(f"{step}/{self.total_steps}", value("train/loss_total"), value("val/loss_total"),
                         f"{value('alignment/t2f/recall_at_1')} / {value('alignment/f2t/recall_at_1')}",
                         f"{rate:.2f} steps/s" if rate else "—",
                         f"{_duration(elapsed)} / {_duration(eta) if eta is not None else '—'}",
                         f"{value('perf/cpu_rss_gib')} / {value('perf/cpu_peak_rss_gib')} GiB",
                         f"{value('perf/gpu_allocated_gib')} / {value('perf/gpu_reserved_gib')} GiB")
        notes = Table(title="Current activity / recent messages", expand=True, show_header=False)
        current = next((self.stages[n] for n in reversed(STAGES)
                        if self.stages.get(n, {}).get("status") == "Running"), {})
        notes.add_row(current.get("detail") or self.phase)
        for line in self.recent:
            notes.add_row(line[-160:])
        return Group(table, training, notes)

    def _event(self, line: str) -> bool:
        if not line.startswith(PREFIX):
            return False
        try:
            event = json.loads(line[len(PREFIX):])
        except ValueError:
            return False
        name = event.get("stage")
        if name == "Activity":
            self.phase = event.get("detail", "Training")
            if not self.tty:
                print(f"[{self.phase}] {event.get('status', 'Running')}", flush=True)
            return True
        if name not in STAGES:
            return False
        previous = self.stages.get(name, {})
        if event.get("status") == "Completed":
            for key in ("current", "total"):
                if event.get(key) is None and previous.get(key) is not None:
                    event[key] = previous[key]
            if not event.get("detail") and previous.get("detail"):
                event["detail"] = previous["detail"]
        if event.get("status") != "Running" and previous.get("time"):
            event["elapsed"] = event["time"] - previous["time"] + previous.get("elapsed", 0)
        elif previous.get("status") == "Running":
            event["elapsed"] = event["time"] - previous["time"] + previous.get("elapsed", 0)
        self.stages[name] = event
        if name == "Training Ready":
            self.phase = "Training"
        if not self.tty and (event.get("status") != previous.get("status")
                             or event.get("current") != previous.get("current")
                             or event.get("detail") != previous.get("detail")):
            print(f"[{name}] {event['status']} {event.get('current') or ''}/{event.get('total') or ''} "
                  f"{event.get('detail', '')}", flush=True)
        return True

    def _log_line(self, line: str) -> None:
        if self._event(line):
            return
        if self.tty:
            if "Traceback (most recent call last)" in line or self.phase == "Traceback":
                self.phase = "Traceback"
                self.console.print(line, markup=False, highlight=False)
            elif line.strip() and not line.startswith(("Train step ", "Val step ", "Alignment ")):
                self.recent.append(line.rstrip())
        else:
            print(line, flush=True)

    def _read_file(self, path: Path, offset: int, line_handler) -> int:
        if not path.exists():
            return offset
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            handle.seek(offset)
            while True:
                start = handle.tell()
                line = handle.readline()
                if not line or not line.endswith("\n"):
                    return start
                line_handler(line.rstrip("\r\n"))

    def poll(self) -> None:
        if self.log_path:
            self.log_offset = self._read_file(self.log_path, self.log_offset, self._log_line)
        def on_metric(line):
            try:
                record = json.loads(line)
            except ValueError:
                return
            if "step" in record:
                self.metrics["step"] = max(int(record["step"]), self.metrics.get("step", 0))
            self.metrics.update({key: value for key, value in record.items()
                                 if key.startswith(("train/", "val/", "alignment/", "perf/"))
                                 and isinstance(value, (int, float))})
            if record.get("record_type") == "performance_event":
                for key, value in record.items():
                    if key.startswith("perf/") and key.endswith("_seconds"):
                        summary = f"{key[5:-8]} completed at step {record.get('step', '?')} in {value:.1f}s"
                        self.recent.append(summary)
                        if not self.tty:
                            print(summary, flush=True)
            if "seconds" in record:
                self.metrics["seconds"] = record["seconds"]
        self.metrics_offset = self._read_file(self.metrics_path, self.metrics_offset, on_metric)

    def refresh(self, live: Live | None, *, force: bool = False) -> None:
        now = time.monotonic()
        if self.tty and live is not None and force and now - self.last_refresh < 0.5:
            time.sleep(0.5 - (now - self.last_refresh))
            now = time.monotonic()
        if self.tty and live is not None and now - self.last_refresh >= 0.5:
            live.update(self._render(), refresh=True)
            self.last_refresh = now


def run_monitor(command: list[str], metrics_path: Path, total_steps: int, *, tty: bool | None = None) -> int:
    monitor = Monitor(metrics_path, total_steps, tty=tty)
    env = os.environ.copy()
    env["TAF_TERMINAL_MONITOR"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1, start_new_session=True, env=env)
    assert process.stdout is not None
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    interrupted = None
    def on_signal(signum, _frame):
        nonlocal interrupted
        interrupted = signum
    old_int = signal.signal(signal.SIGINT, on_signal)
    old_term = signal.signal(signal.SIGTERM, on_signal)
    try:
        with Live(monitor._render(), console=monitor.console, auto_refresh=False,
                  screen=False, transient=False) if monitor.tty else _NullLive() as live:
            while True:
                if interrupted is not None:
                    _stop_tree(process, interrupted)
                    monitor.poll()
                    monitor.refresh(live, force=True)
                    return 128 + interrupted
                for key, _ in selector.select(timeout=0.5):
                    line = key.fileobj.readline()
                    if not line:
                        selector.unregister(key.fileobj)
                        continue
                    line = line.rstrip("\r\n")
                    match = re.search(r"\(([0-9a-f]{12})\):", line)
                    if line.startswith("Starting ") and match:
                        monitor.run_id = match.group(1)
                    elif line.startswith("Log: ") and monitor.run_id:
                        candidate = Path(line[5:]).expanduser().resolve()
                        if candidate.name.endswith(f"-{monitor.run_id}.log"):
                            monitor.log_path = candidate
                            monitor.log_offset = 0
                    if not monitor.tty:
                        print(line, flush=True)
                    elif "Traceback (most recent call last)" in line or monitor.phase == "Traceback":
                        monitor.phase = "Traceback"
                        monitor.console.print(line, markup=False, highlight=False)
                    elif not line.startswith("Log: "):
                        monitor.recent.append(line)
                monitor.poll()
                monitor.refresh(live)
                if process.poll() is not None and not selector.get_map():
                    monitor.poll()
                    if process.returncode and monitor.phase != "Traceback":
                        for name in reversed(STAGES):
                            if monitor.stages.get(name, {}).get("status") == "Running":
                                monitor.stages[name]["status"] = "Failed"
                                break
                        monitor.phase = "Failed"
                    monitor.refresh(live, force=True)
                    return 128 - process.returncode if process.returncode < 0 else process.returncode
    finally:
        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGTERM, old_term)
        selector.close()


class _NullLive:
    def __enter__(self):
        return None
    def __exit__(self, *_args):
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--total-steps", type=int, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("trainctl command required after --")
    return run_monitor(command, args.metrics, args.total_steps)


if __name__ == "__main__":
    sys.exit(main())
