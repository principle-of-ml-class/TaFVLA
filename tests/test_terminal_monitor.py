"""Synthetic trainctl log relay tests; no dataset or GPU is used."""

import json
import contextlib
import io
import os
import pty
import select
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from script.terminal_monitor import Monitor, run_monitor, validate_monitor_paths, watch_metrics
from utils.live_progress import PREFIX


ROOT = Path(__file__).resolve().parents[1]


class TerminalMonitorTests(unittest.TestCase):
    def test_existing_metrics_are_loaded_and_continue_updating(self):
        with tempfile.TemporaryDirectory() as directory:
            metrics = Path(directory) / "metrics.jsonl"
            metrics.write_text(json.dumps({"step": 2620, "seconds": 4253.6, "train/loss_total": 14.0}) + "\n")
            monitor = Monitor(metrics, 10000, tty=False)
            monitor.poll()
            self.assertEqual(monitor.metrics["step"], 2620)
            self.assertEqual(monitor.metrics["seconds"], 4253.6)
            with metrics.open("a") as stream:
                stream.write(json.dumps({"step": 2640, "train/loss_total": 13.0}) + "\n")
            monitor.poll()
            self.assertEqual(monitor.metrics["step"], 2640)
            self.assertEqual(monitor.metrics["train/loss_total"], 13.0)

    def test_bad_paths_fail_before_starting_any_process(self):
        with tempfile.TemporaryDirectory() as directory:
            metrics = Path(directory) / "metrics.jsonl"
            for path, checkpoint_dir in ((Path("/metrics.jsonl"), ""), (metrics, ""), (metrics, directory + "/wrong")):
                with self.subTest(path=path, checkpoint_dir=checkpoint_dir), \
                        patch("script.terminal_monitor.subprocess.Popen") as launch:
                    with self.assertRaises(ValueError):
                        run_monitor(["uv", "run", "trainctl", "run", "--checkpoint-dir", checkpoint_dir,
                                     "--", "python", "fake.py"], path, 10000)
                    launch.assert_not_called()
            validate_monitor_paths(["uv", "run", "trainctl", "run", "--checkpoint-dir", directory,
                                    "--", "python", "fake.py"], metrics)
            validate_monitor_paths(["trainctl", "run", f"--checkpoint-dir={directory}", "--", "fake"], metrics)

    def test_watch_snapshot_never_starts_or_stops_training(self):
        with tempfile.TemporaryDirectory() as directory:
            metrics = Path(directory) / "metrics.jsonl"
            metrics.write_text(json.dumps({"step": 10000, "train/loss_total": 1.5}) + "\n" +
                               json.dumps({"step": 10000, "record_type": "performance_event",
                                           "perf/validation_seconds": 42.0}) + "\n")
            with contextlib.redirect_stdout(io.StringIO()) as output, \
                    patch("script.terminal_monitor.subprocess.Popen") as launch, \
                    patch("script.terminal_monitor._stop_tree") as stop:
                self.assertEqual(watch_metrics(metrics, 10000, once=True), 0)
                launch.assert_not_called(); stop.assert_not_called()
            self.assertIn("Completed", output.getvalue())
            # Historical events may appear inside the dashboard's recent rows,
            # but must not be printed as a stream before the dashboard itself.
            self.assertTrue(output.getvalue().lstrip().startswith("TaFVLA"))
            with self.assertRaisesRegex(ValueError, "Existing metrics"):
                watch_metrics(metrics.with_name("missing.jsonl"), 10000, once=True)

    def test_plain_relay_and_run_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "test-abcdef123456.log"
            metrics = root / "metrics.jsonl"
            fake = root / "fake.py"
            fake.write_text(
                "import json,sys,time\n"
                "from pathlib import Path\n"
                "log, metrics = map(Path, sys.argv[1:])\n"
                "print('Starting \\\'test\\\' (abcdef123456): python fake.py', flush=True)\n"
                "print('Log: ' + str(log), flush=True)\n"
                "time.sleep(0.1)\n"
                "with log.open('w') as f:\n"
                " f.write('hello from training\\n')\n"
                " f.write('" + PREFIX + "' + json.dumps(dict(stage='Training Ready',status='Completed',time=time.time())) + '\\n')\n"
                " f.write('Traceback (most recent call last):\\nValueError: fake\\n')\n"
                "with metrics.open('w') as f: f.write(json.dumps({'step':2,'train/loss_total':1.5})+'\\n')\n",
                encoding="utf-8",
            )
            result = subprocess.run(
                [sys.executable, "-m", "script.terminal_monitor", "--metrics", str(metrics),
                 "--total-steps", "10", "--", sys.executable, str(fake), str(log), str(metrics)],
                cwd=ROOT, capture_output=True, text=True, timeout=5,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.count("hello from training"), 1)
            self.assertIn("Traceback (most recent call last):", result.stdout)
            self.assertNotIn(PREFIX, result.stdout)
            self.assertEqual(log.read_text().count("hello from training"), 1)
            self.assertNotIn("\x1b", log.read_text())
            monitor = Monitor(metrics, 10, tty=True)
            self.assertTrue(monitor.tty)
            self.assertIsNotNone(monitor._render())

    def test_sigint_stops_supervised_child_group(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pidfile = root / "child.pid"
            fake = root / "supervisor.py"
            fake.write_text(
                "import subprocess,sys,time\n"
                "from pathlib import Path\n"
                "child = subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'], start_new_session=True)\n"
                "Path(sys.argv[1]).write_text(str(child.pid))\n"
                "print('Starting \\\'test\\\' (abcdef123456): fake', flush=True)\n"
                "print('Log: ' + sys.argv[2], flush=True)\n"
                "time.sleep(30)\n", encoding="utf-8",
            )
            process = subprocess.Popen(
                [sys.executable, "-m", "script.terminal_monitor", "--metrics", str(root / "metrics.jsonl"),
                 "--total-steps", "10", "--", sys.executable, str(fake), str(pidfile),
                 str(root / "test-abcdef123456.log")],
                cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            try:
                deadline = time.monotonic() + 3
                while not pidfile.exists() and time.monotonic() < deadline:
                    time.sleep(0.05)
                self.assertTrue(pidfile.exists())
                child_pid = int(pidfile.read_text())
                process.send_signal(signal.SIGINT)
                self.assertEqual(process.wait(timeout=10), 130)
                with self.assertRaises(ProcessLookupError):
                    os.kill(child_pid, 0)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
                process.stdout.close()
                process.stderr.close()

    def test_tty_dashboard_keeps_log_plain(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "test-abcdef123456.log"
            fake = root / "fake.py"
            fake.write_text(
                "import sys,time\n"
                "from pathlib import Path\n"
                "print('Starting test (abcdef123456): fake', flush=True)\n"
                "print('Log: ' + sys.argv[1], flush=True)\n"
                "time.sleep(.1)\n"
                "Path(sys.argv[1]).write_text('plain log line\\n')\n",
                encoding="utf-8",
            )
            master, slave = pty.openpty()
            try:
                process = subprocess.Popen(
                    [sys.executable, "-m", "script.terminal_monitor", "--metrics", str(root / "metrics.jsonl"),
                     "--total-steps", "10", "--", sys.executable, str(fake), str(log)],
                    cwd=ROOT, stdout=slave, stderr=slave,
                )
                os.close(slave)
                output = bytearray()
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    ready, _, _ = select.select([master], [], [], 0.1)
                    if ready:
                        try:
                            output.extend(os.read(master, 65536))
                        except OSError:
                            break
                    if process.poll() is not None and not ready:
                        break
                self.assertEqual(process.wait(timeout=1), 0)
                self.assertIn(b"TaFVLA", output)
                self.assertIn(b"\x1b[", output)
                self.assertEqual(log.read_text(), "plain log line\n")
            finally:
                os.close(master)
                if process.poll() is None:
                    process.kill()
                    process.wait()
