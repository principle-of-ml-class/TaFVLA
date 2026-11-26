"""Adapted episode heatmap/timeline ideas; source details in docs/phase4_sources.json."""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.phase_classifier import PHASES, PHASE_COLORS


def pressure_heatmap(pressure, title="Pressure (raw values)"):
    figure = go.Figure(go.Heatmap(z=np.asarray(pressure).reshape(12, 12), colorscale="Viridis", zsmooth=False))
    figure.update_yaxes(autorange="reversed", scaleanchor="x", scaleratio=1)
    figure.update_layout(title=title, height=330)
    return figure


def signal_plot(timestamps, force, selected):
    figure = make_subplots(rows=2, cols=1, shared_xaxes=True)
    step = max(1, len(timestamps)//6000)
    for axis, name in enumerate(("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")):
        figure.add_trace(go.Scattergl(x=timestamps[::step], y=force[::step, axis], name=name), row=1 if axis < 3 else 2, col=1)
    figure.add_vline(x=float(timestamps[selected]), line_dash="dash", row="all", col=1)
    figure.update_layout(height=450); figure.update_xaxes(title_text="Timestamp (s)", row=2, col=1)
    return figure


def phase_plot(labels, selected, n):
    times = np.asarray(labels["timestamp"]); figure = make_subplots(rows=2, cols=1, shared_xaxes=True)
    step = max(1, len(times)//6000)
    for name in ("raw_fz", "signed_fz"):
        figure.add_trace(go.Scattergl(x=times[::step], y=np.asarray(labels[name])[::step], name=name), row=1, col=1)
    for track, field in enumerate(("phase_02", "phase_03", "phase")):
        for phase in PHASES:
            indices = np.flatnonzero(np.asarray(labels[field]) == phase)
            indices = indices[::max(1, len(indices)//4000)]
            figure.add_trace(go.Scattergl(x=times[indices], y=np.full(len(indices), track), mode="markers",
                                         marker=dict(color=PHASE_COLORS[phase], symbol="square", size=6),
                                         name=phase.title(), legendgroup=phase, showlegend=track == 0), row=2, col=1)
    conflict = np.flatnonzero(labels["direction_conflict"])
    figure.add_trace(go.Scattergl(x=times[conflict], y=np.full(len(conflict), 2), mode="markers",
                                 marker=dict(symbol="x", color="black", size=9), name="Direction conflict"), row=2, col=1)
    figure.add_vrect(x0=float(times[max(0, selected-n+1)]), x1=float(times[selected]), fillcolor="gold", opacity=.15,
                    annotation_text=f"Model history N={n}", row=1, col=1)
    for suffix, color in (("02", "blue"), ("03", "red")):
        lo, hi = labels[f"window_start_{suffix}"][selected], labels[f"window_end_{suffix}"][selected]
        if lo >= 0:
            figure.add_vrect(x0=float(times[lo]), x1=float(times[hi]), fillcolor=color, opacity=.08,
                            annotation_text=f"Offline classifier {suffix}", row=1, col=1)
    figure.update_yaxes(tickvals=[0, 1, 2], ticktext=["0.2s", "0.3s", "Fused"], row=2, col=1)
    figure.add_vline(x=float(times[selected]), line_dash="dash", row="all", col=1)
    figure.update_layout(height=570)
    return figure
