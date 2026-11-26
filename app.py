"""TaFVLA prepared-data explorer. Run via bash scripts/phase4.sh ui."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import threading

import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from utils.phase4_data import load_prepared, load_settings, source_for_settings, file_hash
from utils.phase_classifier import PHASE_COLORS
from visualization.data import episode_labels, prediction_episode, result_catalog
from visualization.plots import pressure_heatmap, signal_plot, phase_plot


@st.cache_resource(max_entries=2)
def source_resource(config_path, overrides, config_stamp):
    return source_for_settings(load_settings(config_path, overrides)), threading.RLock()


@st.cache_data(max_entries=2)
def preparation(path, modified):
    report = load_prepared(path, verify=False)
    # The UI does not scan all dense label/manifest files on page load.
    for name in ("episodes.json", "calibration.json", "episode_split.json", "manifest_identity.json"):
        if file_hash(Path(path)/name) != report["artifacts"][name]:
            raise ValueError(f"Prepared metadata changed: {name}")
    return report, json.loads((Path(path)/"episodes.json").read_text())


@st.cache_data(max_entries=3)
def episode_data(config_path, overrides, prepared_dir, episode_json, manifest_id, config_stamp):
    episode = json.loads(episode_json); source, lock = source_resource(config_path, overrides, config_stamp)
    with lock:
        matches = [ep for ep in source.episodes if (ep.sensor_type, ep.sequence_id, ep.episode_index) ==
                   (episode["sensor_configuration"], episode["sequence_id"], episode["episode_index"])]
        if len(matches) != 1:
            raise ValueError("Prepared episode cannot be matched to raw dataset")
        ep = matches[0]
        for name in ("parquet", "video"):
            expected = episode["source_provenance"].get(name)
            if expected is not None:
                file = Path(getattr(ep, name)); stat = file.stat()
                if (str(file.resolve()), stat.st_size, stat.st_mtime_ns) != (expected["path"], expected["bytes"], expected["mtime_ns"]):
                    raise ValueError(f"Raw episode {name} changed since preparation")
        force, pressure = source._sensor_rows(ep, np.arange(ep.length))
    expected = load_prepared(prepared_dir, verify=False)["artifacts"][episode["label_file"]]
    if file_hash(Path(prepared_dir)/episode["label_file"]) != expected:
        raise ValueError("Selected episode label artifact changed")
    labels = episode_labels(prepared_dir, episode)
    if not np.array_equal(labels["source_frame_index"], ep.frames) or not np.allclose(labels["timestamp"], ep.timestamps, atol=1e-6):
        raise ValueError("Raw timestamps/frames changed since preparation")
    return force, pressure, labels


@st.cache_data(max_entries=4)
def stored_predictions(directory, episode_json, method):
    return prediction_episode(directory, json.loads(episode_json), method)


def preparation_failures(prepared_dir):
    failures = []
    for path in Path(prepared_dir).parent.glob(".phase4_prepare_*/failure.json"):
        row = json.loads(path.read_text())
        if row.get("intended_output") == str(Path(prepared_dir).resolve()):
            failures.append({"file": str(path), **row})
    return failures[-20:]


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="config/phase4_pipeline.yaml")
    parser.add_argument("--set", action="append", default=[])
    args, _ = parser.parse_known_args(); cfg = load_settings(args.config, args.set)
    config_stamp = Path(args.config).stat().st_mtime_ns
    st.set_page_config(page_title="TaFVLA Phase4", layout="wide")
    st.title("TaFVLA · 네 상황과 Temporal Context")
    summary_path = Path(cfg.prepared_dir)/"summary.json"
    if not summary_path.is_file():
        st.info("준비된 데이터가 없습니다. bash scripts/phase4.sh prepare 를 먼저 실행하세요.")
        for failure in preparation_failures(cfg.prepared_dir):
            st.error(f"준비 실패: {failure['error']}"); st.json(failure)
        return
    summary, episodes = preparation(cfg.prepared_dir, summary_path.stat().st_mtime_ns)
    page = st.sidebar.radio("Page", ["Dataset Explorer", "Phase Inspector", "Dataset Statistics", "Model Analysis"])
    sensor = st.sidebar.selectbox("Sensor", sorted({r["sensor_configuration"] for r in episodes}))
    sequence = st.sidebar.selectbox("Sequence", sorted({r["sequence_id"] for r in episodes if r["sensor_configuration"] == sensor}))
    selected_episodes = [r for r in episodes if r["sensor_configuration"] == sensor and r["sequence_id"] == sequence]
    episode_index = st.sidebar.selectbox("Episode", [r["episode_index"] for r in selected_episodes])
    episode = next(r for r in selected_episodes if r["episode_index"] == episode_index)
    st.sidebar.write(f"Split: **{episode['split']}**")
    st.sidebar.caption(f"Manifest {summary['manifest_id'][:12]}")
    n = st.sidebar.selectbox("Model N (classifier window와 독립)", cfg.windows, index=2)
    episode_json = json.dumps(episode, sort_keys=True)
    if page == "Dataset Statistics":
        st.subheader("공통 endpoint / 전체 frame 통계를 구분합니다")
        st.write("Episode 수", {s:sum(r["split"] == s for r in episodes) for s in ("train", "validation", "test")})
        anchor_rows = [{"split": s, **counts} for s, counts in summary["anchor_support"].items()]
        st.dataframe(anchor_rows); st.dataframe(summary["frame_support"])
        st.plotly_chart(px.bar(summary["frame_support"], x="split", y="frames", color="phase", facet_col="sensor", color_discrete_map=PHASE_COLORS))
        st.write("제외 사유", summary["exclusions"]); st.write("준비 실패", summary["failures"])
        history = preparation_failures(cfg.prepared_dir)
        if history:
            st.write("이전 준비 시도의 실패 기록", history)
        st.caption("Unknown도 학습·평가에 포함됩니다. No Contact 제외와 sudden-event 우선순위는 사용하지 않습니다.")
        return
    if page == "Model Analysis":
        catalog = result_catalog(cfg.results_dir, summary["manifest_id"])
        if not catalog:
            st.info("이 manifest의 저장된 모델 평가 결과가 아직 없습니다."); return
        families = sorted({r["summary"]["experiment_id"] for r in catalog})
        family = st.selectbox("Experiment configuration", families)
        catalog = [r for r in catalog if r["summary"]["experiment_id"] == family]
        metrics = [m for record in catalog for m in record["summary"]["metrics"]]
        st.dataframe(metrics)
        phase = st.selectbox("Phase", ["all", *PHASE_COLORS])
        metric = st.selectbox("Metric", ["endpoint_standardized_wrench_rmse", "endpoint_raw_pressure_rmse"])
        plot_rows = [r for r in metrics if r["phase"] == phase and r["method"] != "gt_vq_reconstruction"]
        if plot_rows:
            st.plotly_chart(px.line(plot_rows, x="N", y=metric, color="method", line_dash="split", markers=True))
        else:
            st.info("이 상황에 해당하는 평가 endpoint가 없습니다.")
        cohort = Path(catalog[0]["directory"]).parents[2]
        analysis = cohort/"analysis"
        if (analysis/"summary.json").is_file():
            routing = json.loads((analysis/"summary.json").read_text()); st.write("Fixed-N / oracle routing", routing["policy"])
            st.dataframe([r for r in routing["metrics"] if r["phase"] == phase])
            st.caption("Offline force-derived oracle routing이며 tactile-only 실시간 routing이 아닙니다. 평균 N은 latency가 아닙니다.")
        if (analysis/"retrieval.parquet").is_file():
            import pyarrow.parquet as pq
            st.dataframe(pq.read_table(analysis/"retrieval.parquet").to_pylist())
        else:
            retrieval = []
            for record in catalog:
                for split in ("validation", "test"):
                    path = Path(record["directory"]).parent/f"retrieval_{split}.json"
                    if path.is_file():
                        report = json.loads(path.read_text())
                        retrieval.extend({"N": report["N"], "split": split, "phase": group, **values}
                                         for group, values in report["groups"].items())
            if retrieval:
                st.dataframe(retrieval)
        candidates = [r for r in catalog if r["summary"]["N"] == n]
        if not candidates:
            st.info("선택한 N의 저장된 예측이 없습니다."); return
        options = {m: (candidates[0]["directory"], m) for m in ("ridge", "vq_decoding", "training_mean", "gt_vq_reconstruction")}
        if (analysis/"prediction_index.parquet").is_file():
            for m in ("ridge", "vq_decoding"):
                options[f"oracle/{m}"] = (str(analysis), f"oracle_phase_routing/{m}")
        choice = st.selectbox("Saved prediction method", list(options)); directory, method = options[choice]
        st.caption("VQ decoding = train-observed code-pair decoding. 정답 Force/Pressure gallery를 예측 후보로 사용하지 않습니다.")
        rows = stored_predictions(directory, episode_json, method)
        if not rows:
            st.info("이 episode/split에는 저장된 평가 예측이 없습니다. Train은 probe fitting에 사용됩니다."); return
        axis = st.selectbox("Wrench axis (raw values)", ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"], index=2)
        axis_index = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"].index(axis)
        figure = go.Figure()
        for field in ("actual_wrench", "predicted_wrench"):
            figure.add_trace(go.Scattergl(x=[r["timestamp_anchor"] for r in rows], y=[r[field][axis_index] for r in rows], name=field))
        st.plotly_chart(figure)
        index = st.slider("Saved endpoint", 0, len(rows)-1, 0) if len(rows) > 1 else 0
        left, right = st.columns(2)
        left.plotly_chart(pressure_heatmap(rows[index]["actual_pressure"], "Actual endpoint pressure"))
        right.plotly_chart(pressure_heatmap(rows[index]["predicted_pressure"], "Predicted endpoint pressure"))
        st.caption("GT VQ reconstruction은 정답 센서를 입력한 별도 진단이며 tactile-only 예측이 아닙니다.")
        return
    force, pressure, labels = episode_data(args.config, tuple(args.set), cfg.prepared_dir, episode_json, summary["manifest_id"], config_stamp)
    selected = st.slider("Frame position", 0, len(force)-1, 0) if len(force) > 1 else 0
    st.write(f"Frame {labels['source_frame_index'][selected]} · timestamp {labels['timestamp'][selected]:.6f}s · {labels['phase'][selected].title()}")
    if selected < n-1:
        st.info("선택한 모델 N의 history가 부족합니다. 공통 학습 endpoint는 position 19부터입니다.")
    if page == "Dataset Explorer":
        source, lock = source_resource(args.config, tuple(args.set), config_stamp)
        with lock:
            ep = next(e for e in source.episodes if (e.sensor_type, e.sequence_id, e.episode_index) == (sensor, sequence, episode_index))
            image, pts = source._images(ep, np.asarray([selected]))
        left, right = st.columns(2)
        left.image(image[0], caption=f"검증된 timestamp 매칭 · video PTS {float(pts[0]):.6f}")
        right.plotly_chart(pressure_heatmap(pressure[selected]))
        st.plotly_chart(signal_plot(np.asarray(labels["timestamp"]), force, selected))
    else:
        st.plotly_chart(phase_plot(labels, selected, n))
        st.write("Episode sign/baseline과 창별 공통 threshold", episode["preprocessing"])
        st.dataframe([{key: labels[key][selected] for key in ("phase_02", "phase_03", "phase", "direction_conflict", "slope_02", "slope_03", "change_02", "change_03", "supported_02", "supported_03", "actual_span_02", "actual_span_03")}])
        st.caption("0.2/0.3초 창은 미래 force도 사용하는 offline 라벨입니다. 노란 모델 입력 구간은 현재까지의 tactile history입니다.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        st.error(f"데이터/설정 오류: {error}")
        st.exception(error)
