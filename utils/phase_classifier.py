"""Offline raw-Fz short-window classifier, adapted from the cited teammate sources.

See docs/phase4_sources.json. No contact rejection, smoothing or post-fusion filter.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

PHASES = ("loading", "holding", "unloading", "unknown")
PHASE_COLORS = dict(zip(PHASES, ("#e66101", "#1b9e77", "#7570b3", "#999999")))
VERSION = "raw-fz-fusion-phase4-v1"
SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def identity(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


@dataclass
class PhaseConfig:
    range_fraction: float = .02
    noise_multiplier: float = 3.
    min_run_frames: int = 2
    baseline_pressure_quantile: float = .1
    timestamp_relative_tolerance: float = .1

    def __post_init__(self):
        if self.range_fraction <= 0 or self.noise_multiplier <= 0 or self.min_run_frames != 2:
            raise ValueError("Invalid phase thresholds; upstream min_run_frames must be 2")
        if not 0 <= self.baseline_pressure_quantile <= 1 or not 0 < self.timestamp_relative_tolerance < 1:
            raise ValueError("Invalid preprocessing/timestamp options")

    @property
    def configuration_id(self):
        return identity({"version": VERSION, "source_sha256": SOURCE_SHA256, **asdict(self), "windows_seconds": [.2, .3], "smoothing": "none"})


def validate_signals(timestamps, force, pressure, config: PhaseConfig):
    t, f, p = (np.asarray(v, dtype=np.float64) for v in (timestamps, force, pressure))
    if t.ndim != 1 or len(t) < 1 or f.shape != (len(t), 6) or p.shape != (len(t), 12, 12):
        raise ValueError("Episode arrays must be timestamps[F], force[F,6], pressure[F,12,12]")
    if not all(np.isfinite(v).all() for v in (t, f, p)):
        raise ValueError("Non-finite episode timestamp/force/pressure")
    dt = np.diff(t)
    if np.any(dt <= 0):
        raise ValueError("Episode timestamps must be strictly increasing")
    if len(dt) and np.max(np.abs(dt-np.median(dt))) > config.timestamp_relative_tolerance*np.median(dt):
        raise ValueError("Irregular timestamps exceed fixed-window regression tolerance")
    return t, f, p


def episode_preprocess(timestamps, force, pressure, config: PhaseConfig = PhaseConfig()):
    t, f, p = validate_signals(timestamps, force, pressure, config)
    raw = f[:, 2].copy(); total = p.reshape(len(p), -1).sum(1)
    sign = 1
    if len(raw) >= 3 and np.std(raw) > 0 and np.std(total) > 0:
        correlation = float(np.corrcoef(raw, total)[0, 1])
        sign = -1 if correlation < 0 else 1
    signed = sign*raw
    mask = total <= np.quantile(total, config.baseline_pressure_quantile)
    baseline = float(np.median(signed[mask]))
    return signed-baseline, {"applied_sign": sign, "episode_baseline": baseline,
                             "baseline_mode": "low_pressure", "baseline_pressure_quantile": config.baseline_pressure_quantile,
                             "offline_episode_wide": True, "smoothing": "none"}


def fit_sensor_calibration(episodes, config: PhaseConfig = PhaseConfig()):
    """episodes yields (split, timestamps, original wrench, pressure), once per episode."""
    forces, noises = [], []
    for split, t, f, p in episodes:
        if split != "train":
            raise ValueError("Calibration may only consume Train episodes")
        signed, _ = episode_preprocess(t, f, p, config)
        forces.append(signed)
        if len(signed) >= 3:
            second = np.diff(signed, n=2)  # Deliberately inside each episode.
            noises.append(float(1.4826*np.median(np.abs(second-np.median(second)))/np.sqrt(6)))
    if not forces or not noises:
        raise ValueError("Sensor has no usable Train episodes for calibration")
    pooled = np.concatenate(forces)
    result = {"noise_sigma": float(np.median(noises)), "force_iqr": float(np.quantile(pooled, .75)-np.quantile(pooled, .25)),
              "train_frames": len(pooled), "train_episodes": len(forces), "version": VERSION,
              "configuration_id": config.configuration_id,
              "noise_estimator": "median episode MAD(second difference of raw signed Fz)/sqrt(6)"}
    result["calibration_id"] = identity(result)
    return result


def classify_window(timestamps, signed_force, seconds: float, calibration: dict, config: PhaseConfig = PhaseConfig()):
    t = np.asarray(timestamps, dtype=np.float64); f = np.asarray(signed_force, dtype=np.float64)
    if t.ndim != 1 or f.shape != t.shape or not len(t) or not np.isfinite(t).all() or not np.isfinite(f).all():
        raise ValueError("Invalid classification arrays")
    dt_values = np.diff(t)
    if len(dt_values) and (np.any(dt_values <= 0) or np.max(np.abs(dt_values-np.median(dt_values))) > config.timestamp_relative_tolerance*np.median(dt_values)):
        raise ValueError("Invalid classification timestamps")
    if calibration["configuration_id"] != config.configuration_id:
        raise ValueError("Stale calibration configuration")
    dt = float(np.median(dt_values)) if len(dt_values) else None
    labels = np.full(len(f), "unknown", dtype=object)
    slope = np.full(len(f), np.nan); change = slope.copy()
    valid = np.zeros(len(f), dtype=bool)
    if dt is None:
        return labels, slope, change, valid, {"requested_seconds": seconds, "reason": "insufficient_timestamps", "points": None}
    intervals = int(round(seconds/dt)); points = intervals+1
    if intervals < 1:
        raise ValueError("Sampling interval is too long for classifier windows")
    left, right = intervals//2, intervals-intervals//2
    x = np.arange(points)*dt; x -= x.mean(); duration = intervals*dt
    floor = config.range_fraction*calibration["force_iqr"]
    noise = calibration["noise_sigma"]
    change_thr = max(config.noise_multiplier*np.sqrt(2)*noise, floor)
    slope_thr = max(config.noise_multiplier*noise/np.sqrt(x@x), floor/duration)
    details = dict(requested_seconds=seconds, points=points, actual_span_seconds=duration,
                   median_dt=dt, min_dt=float(dt_values.min()), max_dt=float(dt_values.max()),
                   left_frames=left, right_frames=right, center_offset_seconds=(right-left)*dt/2,
                   change_threshold=change_thr, slope_threshold=slope_thr,
                   reason="short_episode" if len(f) < points else "boundary_support_unavailable")
    if len(f) >= points:
        windows = np.lib.stride_tricks.sliding_window_view(f, points)
        inner_slope = windows@x/(x@x); inner_change = windows[:, -1]-windows[:, 0]
        inner = np.full(len(windows), "unknown", object)
        inner[(inner_slope >= slope_thr)&(inner_change >= change_thr)] = "loading"
        inner[(inner_slope <= -slope_thr)&(inner_change <= -change_thr)] = "unloading"
        inner[(abs(inner_slope) <= slope_thr*.5)&(abs(inner_change) <= change_thr*.5)] = "holding"
        support = slice(left, len(f)-right)
        labels[support] = inner; slope[support] = inner_slope; change[support] = inner_change; valid[support] = True
    cuts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1])+1, len(labels)]
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b-a < config.min_run_frames:
            labels[a:b] = "unknown"
    return labels, slope, change, valid, details


def fuse_phases(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or not set(np.unique(a)) <= set(PHASES) or not set(np.unique(b)) <= set(PHASES):
        raise ValueError("Invalid phase arrays")
    loading = (a == "loading") | (b == "loading")
    unloading = (a == "unloading") | (b == "unloading")
    result = np.full(a.shape, "unknown", object)
    result[loading & ~unloading] = "loading"; result[unloading & ~loading] = "unloading"
    result[(a == "holding") & (b == "holding")] = "holding"
    return result, loading & unloading


def classify_episode(timestamps, force, pressure, calibration, config: PhaseConfig = PhaseConfig()):
    signed, preprocessing = episode_preprocess(timestamps, force, pressure, config)
    a = classify_window(timestamps, signed, .2, calibration, config)
    b = classify_window(timestamps, signed, .3, calibration, config)
    fused, conflict = fuse_phases(a[0], b[0])
    supports = {}
    positions = np.arange(len(signed)); t = np.asarray(timestamps)
    for name, values in (("02", a), ("03", b)):
        details = values[4]; mask = values[3]
        starts = np.full(len(signed), -1, dtype=np.int64); ends = starts.copy()
        spans = np.full(len(signed), np.nan)
        starts[mask] = positions[mask]-details.get("left_frames", 0)
        ends[mask] = positions[mask]+details.get("right_frames", 0)
        spans[mask] = t[ends[mask]]-t[starts[mask]]
        supports.update({f"window_start_{name}": starts, f"window_end_{name}": ends,
                         f"actual_span_{name}": spans})
    return {**supports, "raw_fz": np.asarray(force)[:, 2].copy(), "signed_fz": signed,
            "phase_02": a[0], "phase_03": b[0], "phase": fused, "direction_conflict": conflict,
            "slope_02": a[1], "change_02": a[2], "supported_02": a[3],
            "slope_03": b[1], "change_03": b[2], "supported_03": b[3]}, {
                **preprocessing, "window_02": a[4], "window_03": b[4], "version": VERSION,
                "calibration_id": calibration["calibration_id"], "configuration_id": config.configuration_id}
