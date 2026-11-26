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
    "Normalization", "Model / Optimizer Initialization", "Codebook Initialization", "Training Sampler Initialization",
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
        # A resumed run or an attached dashboard must show the existing step
        # immediately, even before the next metrics record is appended.
        self.metrics_offset = 0
        self._metrics_initialized = False
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
            progress = f"{current:,}/{total:,}" if current is not None and total else (str(current) if current is not None else "—")
            if status == "Running" and item.get("phase"):
                progress = f"{item['phase']}: {progress}"
            elapsed = item.get("elapsed", 0) + (max(0, now - item.get("time", now)) if status == "Running" else 0)
            phase_elapsed = max(0, now - item.get("phase_started", item.get("time", now)))
            eta = phase_elapsed * max(0, total - current) / current if status == "Running" and current and total else None
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
        elapsed = m.get("perf/elapsed_seconds", m.get("seconds", time.monotonic() - self.started))
        rate = m.get("perf/steps_per_second", 0)
        eta = (self.total_steps - step) / rate if rate else None
        cpu_rss_key = "perf/total_rss_gib" if "perf/total_rss_gib" in m else "perf/cpu_rss_gib"
        cpu_peak_key = ("perf/total_peak_sampled_rss_gib" if "perf/total_peak_sampled_rss_gib" in m
                        else "perf/cpu_peak_rss_gib")
        step_label = str(step) if "step" in m else "—"
        training.add_row(f"{step_label}/{self.total_steps}", value("train/loss_total"), value("val/loss_total"),
                         f"{value('alignment/t2f/recall_at_1')} / {value('alignment/f2t/recall_at_1')}",
                         f"{rate:.2f} steps/s" if rate else "—",
                         f"{_duration(elapsed)} / {_duration(eta) if eta is not None else '—'}",
                         f"{value(cpu_rss_key)} / {value(cpu_peak_key)} GiB",
                         f"{value('perf/gpu_allocated_gib')} / {value('perf/gpu_reserved_gib')} GiB")
        stability = Table(title="Training stability", expand=True)
        for title in ("LR / phase", "Gradient before / after", "Clip coefficient / fraction", "Pressure / Force PPL", "AMP skips"):
            stability.add_column(title)
        stability.add_row(f"{value('train/learning_rate')} / {m.get('train/scheduler_phase', '—')}",
                          f"{value('train/gradient_norm_before_clipping')} / {value('train/gradient_norm_after_clipping')}",
                          f"{value('train/gradient_clipping_coefficient')} / {value('train/gradient_clipping_fraction')}",
                          f"{value('train/pressure_codebook_perplexity')} / {value('train/force_codebook_perplexity')}",
                          value('train/amp_skipped_update_count'))
        pipeline = Table(title="Data pipeline", expand=True)
        pipeline.add_column("Wait ratio / ms per batch")
        pipeline.add_column("Parquet / video ms per call")
        pipeline.add_column("Sensor / decoder cache hit rate")
        pipeline.add_row(f"{value('perf/data_wait_ratio')} / {value('perf/data_wait_ms')}",
                         f"{value('perf/parquet_read_ms')} / {value('perf/video_decode_ms')}",
                         f"{value('perf/parquet_cache_hit_rate')} / {value('perf/video_cache_hit_rate')}")
        notes = Table(title="Current activity / recent messages", expand=True, show_header=False)
        notes.add_row(f"Metrics: {self.metrics_path}")
        if not self.metrics_path.is_file():
            notes.add_row("Metrics file not created yet; check the path if training is already running.")
        current = next((self.stages[n] for n in reversed(STAGES)
                        if self.stages.get(n, {}).get("status") == "Running"), {})
        notes.add_row(current.get("detail") or self.phase)
        for line in self.recent:
            notes.add_row(line[-160:])
        return Group(table, training, stability, pipeline, notes)

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
        if (previous.get("status") == "Running" and previous.get("phase", "") == event.get("phase", "")):
            event["phase_started"] = previous.get("phase_started", previous.get("time", event["time"]))
        else:
            event["phase_started"] = event["time"]
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
                             or event.get("phase") != previous.get("phase")
                             or event.get("detail") != previous.get("detail")):
            print(f"[{name}] {event['status']} {event.get('phase', '')} "
                  f"{event.get('current') if event.get('current') is not None else ''}/{event.get('total') or ''} "
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
                                 and (isinstance(value, (int, float)) or key == 'train/scheduler_phase')})
            if record.get("record_type") == "performance_event":
                for key, value in record.items():
                    if key.startswith("perf/") and key.endswith("_seconds"):
                        summary = f"{key[5:-8]} completed at step {record.get('step', '?')} in {value:.1f}s"
                        self.recent.append(summary)
                        if not self.tty and self._metrics_initialized:
                            print(summary, flush=True)
            if "seconds" in record:
                self.metrics["seconds"] = record["seconds"]
        self.metrics_offset = self._read_file(self.metrics_path, self.metrics_offset, on_metric)
        self._metrics_initialized = True

    def refresh(self, live: Live | None, *, force: bool = False) -> None:
        now = time.monotonic()
        if self.tty and live is not None and force and now - self.last_refresh < 0.5:
            time.sleep(0.5 - (now - self.last_refresh))
            now = time.monotonic()
        if self.tty and live is not None and now - self.last_refresh >= 0.5:
            live.update(self._render(), refresh=True)
            self.last_refresh = now


def validate_monitor_paths(command: list[str], metrics_path: Path) -> None:
    """Fail before spawning training when shell expansion lost the run directory."""
    if metrics_path.resolve().parent == Path("/"):
        raise ValueError("--metrics points to the filesystem root; check the run directory shell variable")
    for index, token in enumerate(command):
        if Path(token).name != "trainctl" or command[index+1:index+2] != ["run"]:
            continue
        options = command[index+2:]
        if "--" in options:
            options = options[:options.index("--")]
        for position, option in enumerate(options):
            if option == "--checkpoint-dir":
                directory = options[position+1] if position+1 < len(options) else ""
            elif option.startswith("--checkpoint-dir="):
                directory = option.split("=", 1)[1]
            else:
                continue
            if not directory.strip() or directory.startswith("--"):
                raise ValueError("trainctl --checkpoint-dir is empty; set the run directory before launching")
            expected = Path(directory).expanduser().resolve() / "metrics.jsonl"
            if metrics_path.expanduser().resolve() != expected:
                raise ValueError(f"--metrics must match trainctl --checkpoint-dir: {expected}")


def watch_metrics(metrics_path: Path, total_steps: int, *, log_path: Path | None = None,
                  once: bool = False) -> int:
    """Read an existing run without starting or controlling any training process."""
    if not metrics_path.is_file():
        raise ValueError(f"Existing metrics file required for --watch: {metrics_path}")
    if log_path is not None and not log_path.is_file():
        raise ValueError(f"Existing log file required for --log: {log_path}")
    monitor = Monitor(metrics_path, total_steps)
    monitor.log_path = log_path
    monitor.phase = "Watching metrics"
    monitor.poll()
    if int(monitor.metrics.get("step", 0)) >= total_steps:
        monitor.phase = "Completed"
    if once:
        monitor.console.print(monitor._render())
        return 0
    try:
        with Live(monitor._render(), console=monitor.console, auto_refresh=False) as live:
            while True:
                monitor.poll()
                if int(monitor.metrics.get("step", 0)) >= total_steps:
                    monitor.phase = "Completed"
                monitor.refresh(live)
                time.sleep(0.5)
    except KeyboardInterrupt:
        return 0


def run_monitor(command: list[str], metrics_path: Path, total_steps: int, *, tty: bool | None = None) -> int:
    validate_monitor_paths(command, metrics_path)
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
    parser.add_argument("--watch", action="store_true", help="Only view an existing run; never start or stop training")
    parser.add_argument("--once", action="store_true", help="With --watch, print one dashboard snapshot and exit")
    parser.add_argument("--log", type=Path, help="With --watch, read this existing training log")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if args.total_steps < 1:
        parser.error("--total-steps must be positive")
    if args.watch and command:
        parser.error("--watch cannot start a command")
    if not args.watch and (args.once or args.log):
        parser.error("--once and --log require --watch")
    if not args.watch and not command:
        parser.error("trainctl command required after --")
    try:
        if args.watch:
            return watch_metrics(args.metrics, args.total_steps, log_path=args.log, once=args.once)
        return run_monitor(command, args.metrics, args.total_steps)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    sys.exit(main())
