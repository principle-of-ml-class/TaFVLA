"""Small end-to-end preparation/evaluation/UI checks; no network, tracking or training."""
from __future__ import annotations

from dataclasses import asdict
import contextlib
import io
import itertools
import json
from pathlib import Path
import tempfile
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow.parquet as pq
import torch
from omegaconf import OmegaConf

from model.taf_adapter import TaFAdapter
from utils.phase_classifier import PHASES, PhaseConfig, classify_episode, classify_window, episode_preprocess, fit_sensor_calibration, fuse_phases
from utils.phase4_data import PhasePipelineConfig, load_prepared, prepare_phase4
from utils.dynamics import ManifestDataset
from utils.endpoint_eval import MetricSums, RidgeStatistics, endpoint_error, evaluate_endpoints, ridge_predict
from utils.phase_routing import aggregate_and_route, select_policy
from tests.test_dynamics_comparison import FakeSource
from tests.phase4_reference import classify as reference_classify, fuse as reference_fuse


class TinyTactile(torch.nn.Module):
    def __init__(self, output_dim, **options):
        super().__init__(); self.linear = torch.nn.Linear(2, output_dim)

    def forward(self, tactile):
        values = tactile.float().mean((2, 3, 4))
        return self.linear(torch.stack((values.mean(1), values[:, -1]), dim=1))


def tiny_model(n):
    with patch("model.taf_adapter.TactileEncoder", TinyTactile):
        return TaFAdapter(window_size=n, force_encoder={"hidden_dim": 8, "pressure_latent_dim": 3, "force_latent_dim": 3},
                          vq={"pressure_codebook_size": 4, "force_codebook_size": 4})


def synthetic_source(root, length=64):
    source = FakeSource(); source.episodes = []
    for seq in range(4):
        for index in range(3):
            path = root/f"raw_{seq}_{index}.parquet"; path.write_text("synthetic raw fixture")
            source.episodes.append(SimpleNamespace(sensor_type="sensor", sequence_id=f"seq{seq}", episode_index=index,
                length=length, frames=np.arange(100, 100+length), timestamps=np.arange(length)/30.,
                video_timestamps=np.arange(length)/30., video_frame_indices=np.arange(length), parquet=path))
    return source


def write_lerobot_fixture(root, length=40):
    """Tiny real Parquet+AV1 MP4 layout for the unmocked production loader/UI."""
    import pyarrow as pa
    for seq in range(4):
        sequence = Path(root)/"taf_dataset"/"sensor"/f"sensor_obj{seq+1}"
        data = sequence/"data/chunk-000/file-000.parquet"
        video = sequence/"videos/observation.image/chunk-000/file-000.mp4"
        meta = sequence/"meta/episodes/chunk-000/file-000.parquet"
        for path in (data,video,meta):
            path.parent.mkdir(parents=True,exist_ok=True)
        frames = np.tile(np.arange(length),3); total = len(frames)
        values = np.minimum(frames, length-frames).astype(np.float32)/10 + seq*.1
        force = np.stack((values*.2,values*.1,values,values*.03,values*.02,values*.01),axis=1)
        pressure = np.broadcast_to(values[:,None]+2,(total,144)).copy()
        table = pa.table({"index":np.arange(total),"episode_index":np.repeat(np.arange(3),length),
                          "frame_index":frames,"timestamp":frames/30.,
                          "observation.force_torque":pa.array(force.tolist(),type=pa.list_(pa.float32(),6)),
                          "observation.pressure_matrix":pa.array(pressure.tolist(),type=pa.list_(pa.float32(),144))})
        pq.write_table(table,data)
        rows = [{"episode_index":ep,"length":length,"data/chunk_index":0,"data/file_index":0,
                 "dataset_from_index":ep*length,"dataset_to_index":(ep+1)*length,
                 "videos/observation.image/chunk_index":0,"videos/observation.image/file_index":0,
                 "videos/observation.image/from_timestamp":ep*length/30.,
                 "videos/observation.image/to_timestamp":(ep+1)*length/30.} for ep in range(3)]
        pq.write_table(pa.Table.from_pylist(rows),meta)
        info = {"total_episodes":3,"total_frames":total,"fps":30,
                "data_path":"data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
                "video_path":"videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"}
        (sequence/"meta/info.json").write_text(json.dumps(info))
        pixels = np.broadcast_to((frames*4+seq).astype(np.uint8)[:,None,None,None],(total,16,16,3)).copy()
        subprocess.run(["ffmpeg","-hide_banner","-loglevel","error","-f","rawvideo","-pixel_format","rgb24",
                        "-video_size","16x16","-framerate","30","-i","pipe:0","-c:v","libaom-av1",
                        "-cpu-used","8","-threads","1","-crf","30","-g","16","-pix_fmt","yuv420p",str(video)],
                       input=pixels.tobytes(),capture_output=True,check=True,timeout=5)


class ClassifierTests(unittest.TestCase):
    def calibration(self):
        cfg = PhaseConfig()
        return {"noise_sigma": .01, "force_iqr": 1., "configuration_id": cfg.configuration_id,
                "calibration_id": "synthetic"}, cfg

    def test_linear_constant_offset_and_upstream_alignment(self):
        t = np.arange(120)/30; calibration, cfg = self.calibration()
        for seconds, points, left, right in ((.2, 7, 3, 3), (.3, 10, 4, 5)):
            for f, phase in ((20*t, "loading"), (-20*t, "unloading"), (t*0, "holding")):
                result = classify_window(t, f, seconds, calibration, cfg)
                self.assertTrue((result[0][10:-10] == phase).all())
                self.assertTrue(np.array_equal(result[0], classify_window(t, f+100, seconds, calibration, cfg)[0]))
                self.assertEqual((result[4]["points"], result[4]["left_frames"], result[4]["right_frames"]), (points, left, right))
                self.assertTrue((~result[3][:left]).all()); self.assertTrue((~result[3][-right:]).all())

    def test_fusion_all_16_cases_and_symmetry(self):
        upstream_names = dict(zip(PHASES, ("Loading", "Holding", "Unloading", "Transition / Unknown")))
        for a, b in itertools.product(PHASES, repeat=2):
            expected = "unknown" if {a,b} == {"loading","unloading"} else "loading" if "loading" in (a,b) else "unloading" if "unloading" in (a,b) else "holding" if a == b == "holding" else "unknown"
            fused, conflict = fuse_phases([a], [b]); other, reverse = fuse_phases([b], [a])
            self.assertEqual(fused[0], expected); self.assertEqual(fused[0], other[0]); self.assertEqual(conflict[0], reverse[0])
            self.assertEqual(conflict[0], {a,b} == {"loading","unloading"})
            upstream, upstream_conflict = reference_fuse(np.array([upstream_names[a]]), np.array([upstream_names[b]]))
            self.assertEqual(upstream_names[fused[0]], upstream[0])
            np.testing.assert_array_equal(conflict, upstream_conflict)

    def test_regular_and_irregular_timestamps_match_pinned_upstream(self):
        calibration, cfg = self.calibration()
        upstream_cfg = {**calibration, "range_fraction": cfg.range_fraction, "min_run_frames": cfg.min_run_frames}
        names = {"Loading":"loading", "Holding":"holding", "Unloading":"unloading", "Transition / Unknown":"unknown"}
        rng = np.random.default_rng(42)
        for fps in (15, 30, 60):
            regular = np.arange(180)/fps
            jitter = np.r_[0, np.cumsum((1+.35*np.sin(np.arange(179)*1.3))/fps)]
            dropped = regular.copy(); dropped[90:] += 2
            signals = (20*regular, -20*regular, np.zeros(180),
                       np.r_[np.arange(60), np.full(60,59), np.arange(59,-1,-1)],
                       rng.normal(size=180))
            for kind, t in (("regular",regular), ("jitter",jitter), ("dropped_frames",dropped)):
                for number, force in enumerate(signals):
                    with self.subTest(fps=fps, timestamps=kind, signal=number):
                        actual, reference = [], []
                        for seconds in (.2, .3):
                            result = classify_window(t,force,seconds,calibration,cfg)
                            expected, details = reference_classify(t,force,seconds,upstream_cfg)
                            np.testing.assert_array_equal(result[0], [names[label] for label in expected])
                            # The upstream key describes intervals * median dt, not measured timestamp spans.
                            self.assertEqual(result[4]["nominal_span_seconds"],details["actual_span_seconds"])
                            for key in ("points","left_frames","right_frames","slope_threshold","change_threshold"):
                                self.assertEqual(result[4][key],details[key])
                            actual.append(result[0]); reference.append(expected)
                        fused, conflict = fuse_phases(*actual)
                        expected, expected_conflict = reference_fuse(*reference)
                        np.testing.assert_array_equal(fused,[names[label] for label in expected])
                        np.testing.assert_array_equal(conflict,expected_conflict)

    def test_irregular_audit_distinguishes_nominal_and_measured_spans(self):
        calibration, cfg = self.calibration()
        t = np.arange(60)/30; t[30:] += 1
        force = np.zeros((60,6)); force[:,2] = np.arange(60)
        pressure = np.ones((60,12,12))
        result, audit = classify_episode(t,force,pressure,calibration,cfg)
        timing = audit["timestamp_diagnostics"]; dt = np.diff(t); median = np.median(dt)
        self.assertTrue(timing["exceeds_relative_tolerance"])
        self.assertEqual(timing["tolerance_role"],"diagnostic_only")
        self.assertEqual(timing["median_dt"],median)
        self.assertEqual(timing["min_dt"],dt.min()); self.assertEqual(timing["max_dt"],dt.max())
        self.assertEqual(timing["max_relative_deviation"],float(max(abs(dt-median))/median))
        for suffix in ("02","03"):
            window = audit[f"window_{suffix}"]; mask = result[f"supported_{suffix}"]
            spans = t[result[f"window_end_{suffix}"][mask]]-t[result[f"window_start_{suffix}"][mask]]
            np.testing.assert_array_equal(result[f"actual_span_{suffix}"][mask],spans)
            self.assertEqual(window["nominal_span_seconds"],(window["points"]-1)*median)
            self.assertEqual(window["actual_span_seconds_min"],float(spans.min()))
            self.assertEqual(window["actual_span_seconds_median"],float(np.median(spans)))
            self.assertEqual(window["actual_span_seconds_max"],float(spans.max()))
            self.assertGreater(spans.max(),window["nominal_span_seconds"]+.9)
            # A missing-frame gap is diagnostic, never an additional Unknown mask.
            self.assertTrue((result[f"phase_{suffix}"][mask] == "loading").all())
        json.dumps(audit,allow_nan=False)

    def test_irregular_train_episodes_are_included_in_calibration(self):
        cfg = PhaseConfig(); t = np.arange(60)/30
        force = np.zeros((60,6)); force[:,2] = np.sin(np.arange(60)) + np.arange(60)
        pressure = np.ones((60,12,12))
        irregular = t.copy(); irregular[30:] += 2
        uniform = fit_sensor_calibration([("train",t,force,pressure)]*2,cfg)
        actual = fit_sensor_calibration([("train",t,force,pressure),("train",irregular,force,pressure)],cfg)
        self.assertEqual(actual,uniform)
        self.assertEqual(actual["train_episodes"],2); self.assertEqual(actual["train_frames"],120)

    def test_invalid_data_is_not_unknown_and_short_support_is_explicit(self):
        calibration, cfg = self.calibration()
        for length in (1, 2, 6):
            t = np.arange(length)/30; f = np.ones((length,6)); p = np.ones((length,12,12))
            result, audit = classify_episode(t, f, p, calibration, cfg)
            self.assertTrue((result["phase"] == "unknown").all())
            self.assertFalse(result["supported_02"].any())
            self.assertFalse(result["supported_03"].any())
            self.assertIsNone(audit["window_02"]["actual_span_seconds_max"])
            json.dumps(audit,allow_nan=False)
        t = np.arange(20)/30; f = np.ones((20,6)); p = np.ones((20,12,12))
        for bad in (t[::-1], np.r_[t[:10], t[9], t[11:]],
                    *[np.r_[t[:10], value, t[11:]] for value in (np.nan,np.inf,-np.inf)]):
            with self.assertRaises(ValueError):
                classify_episode(bad, f, p, calibration, cfg)
            with self.assertRaises(ValueError):
                classify_window(bad,f[:,2],.2,calibration,cfg)
            with self.assertRaises(ValueError):
                fit_sensor_calibration([("train",bad,f,p)],cfg)
        for value in (np.nan,np.inf,-np.inf):
            for array in (f,p):
                invalid = array.copy(); invalid.flat[2] = value
                with self.assertRaises(ValueError):
                    classify_episode(t,invalid if array is f else f,invalid if array is p else p,calibration,cfg)
            bad_signal = f[:,2].copy(); bad_signal[2] = value
            with self.assertRaises(ValueError):
                classify_window(t,bad_signal,.2,calibration,cfg)
        for bad_t,bad_f,bad_p in ((t[:0],f[:0],p[:0]),(t[:,None],f,p),(t,f[:,:5],p),(t,f,p[:,:,:11])):
            with self.assertRaises(ValueError):
                classify_episode(bad_t,bad_f,bad_p,calibration,cfg)
        with self.assertRaisesRegex(ValueError,"Sampling interval"):
            classify_window(np.arange(20),f[:,2],.2,calibration,cfg)
        with self.assertRaisesRegex(ValueError,"sampling intervals"):
            classify_episode([-np.finfo(float).max,np.finfo(float).max],f[:2],p[:2],calibration,cfg)

    def test_calibration_identity_remains_enforced(self):
        calibration, cfg = self.calibration(); t = np.arange(20)/30
        calibration["configuration_id"] = "previous-classifier-configuration"
        with self.assertRaises(ValueError):
            classify_window(t,np.arange(20),.2,calibration,cfg)

    def test_episode_sign_baseline_and_no_cross_episode_noise(self):
        cfg = PhaseConfig(); t = np.arange(30)/30.
        force = np.zeros((30,6)); force[:,2] = -np.arange(30)
        pressure = np.broadcast_to(np.arange(30)[:,None,None], (30,12,12)).copy()
        old = force.copy(); signed, audit = episode_preprocess(t, force, pressure, cfg)
        self.assertEqual(audit["applied_sign"], -1); self.assertTrue(np.array_equal(force, old))
        calibration = fit_sensor_calibration([("train",t,force,pressure), ("train",t,force-10000,pressure)], cfg)
        self.assertEqual(calibration["noise_sigma"], 0.)
        short = np.zeros((3,6)); short[:,2] = np.arange(3)
        separated = fit_sensor_calibration([("train",t[:3],short,pressure[:3]),
                                             ("train",t[:3],short+10000,pressure[:3])],cfg)
        self.assertEqual(separated["noise_sigma"],0.)
        wrong = np.diff(np.r_[short[:,2],short[:,2]+10000],n=2)
        self.assertGreater(np.median(abs(wrong-np.median(wrong))),0.)
        with self.assertRaisesRegex(ValueError, "Train"):
            fit_sensor_calibration([("validation",t,force,pressure)], cfg)


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_streaming_ridge_and_evaluation_truth_independence(self):
        rng = np.random.default_rng(42); x = rng.normal(size=(50,4)); y = x@rng.normal(size=(4,150))+3
        stats = RidgeStatistics(4)
        for chunk in range(0,50,7):
            stats.add(x[chunk:chunk+7], y[chunk:chunk+7])
        probe = stats.fit(.01); prediction = ridge_predict(x[:3], probe)
        _ = endpoint_error(prediction, y[:3]+10000, np.ones(6))
        self.assertTrue(np.array_equal(prediction, ridge_predict(x[:3], probe)))
        standard = (x-x.mean(0))/x.std(0)
        target = (y-y.mean(0))/y.std(0)
        direct = np.linalg.solve(standard.T@standard+.01*np.eye(4), standard.T@target)
        np.testing.assert_allclose(probe["weights"], direct, rtol=1e-10, atol=1e-10)

    def test_vq_full_concat_original_vectors_and_ties(self):
        model = tiny_model(1).eval()
        with torch.no_grad():
            model.force_encoder.pressure.quantizer.codebook.weight.copy_(torch.tensor([[2.,0,0],[0,3.,0],[2.,0,0],[0,0,4.]]))
            model.force_encoder.force.quantizer.codebook.weight.copy_(torch.tensor([[0,5.,0],[6.,0,0],[0,5.,0],[0,0,7.]]))
        embedding = torch.tensor([[2.,0,0,0,5.,0]])
        bank = torch.tensor([[2,2],[1,1],[0,0]])
        a = model.decode_tactile_embedding(embedding, bank, chunk_size=1)
        b = model.decode_tactile_embedding(embedding, bank, chunk_size=8)
        self.assertEqual(a["code_pairs"].tolist(), [[0,0]])
        torch.testing.assert_close(a["force"], b["force"])
        expected = model.force_encoder.force.decoder(model.force_encoder.force.quantizer.codebook(torch.tensor([0]))).view(1,1,6)
        torch.testing.assert_close(a["force"], expected)
        # Tactile-only API has no evaluation force/pressure parameter at all.
        images = torch.randn(2,1,3,2,2)
        first = model.decode_tactile(images, bank)
        _ = torch.randn(2,1,6)*1000
        second = model.decode_tactile(images, bank)
        torch.testing.assert_close(first["force"], second["force"])

    def test_pooled_rmse_not_mean_of_rmses(self):
        sums = MetricSums()
        for error in (1.,3.):
            pred = np.zeros((1,150)); pred[:,:6] = error
            errors = endpoint_error(pred, np.zeros((1,150)), np.ones(6))
            row = {k:v[0].tolist() for k,v in errors.items()}
            sums.add({**row, "N":3, "sensor_configuration":"s", "sequence_id":"a", "episode_index":0})
        self.assertAlmostEqual(sums.result()["endpoint_standardized_wrench_rmse"], np.sqrt(5))

    def test_policy_five_percent_and_unknown_fallback(self):
        metrics = []
        for method in ("ridge", "vq_decoding"):
            for phase in ("all",*PHASES):
                for n,error in zip((1,3,5,7,10,20),(1.1,1.04,1.,1.01,1.03,1.1)):
                    metrics.append(dict(method=method,split="validation",phase=phase,N=n,anchors=20 if phase=="unloading" else 100,
                                        episodes=3,endpoint_standardized_wrench_rmse=error))
        policy = select_policy(metrics)
        decision = policy["methods"]["ridge"]
        self.assertEqual(decision["global_N"], 3); self.assertEqual(decision["global_minimum_error_N"],5)
        self.assertTrue(decision["phases"]["unknown"]["fallback"])
        self.assertTrue(decision["phases"]["unloading"]["fallback"])
        with self.assertRaises(ValueError):
            select_policy([{**r,"split":"test"} for r in metrics])


class IntegratedPhase4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_irregular_episode_summary_and_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = synthetic_source(root)
            for ep in (source.episodes[0],source.episodes[-1]):
                ep.timestamps[30:] += 1
                ep.video_timestamps = ep.timestamps.copy()
            cfg = PhasePipelineConfig(data_root=str(root),sensors=["sensor"],split_reference=None,
                                      prepared_dir=str(root/"prepared"),gallery_size=4)
            with patch.object(source,"_images",side_effect=AssertionError("No decoding for timestamp audit")):
                report = prepare_phase4(source,cfg)
            episodes = json.loads((Path(cfg.prepared_dir)/"episodes.json").read_text())
            timing = report["timestamp_diagnostics"]
            self.assertEqual(timing["irregular_episodes"],2)
            self.assertEqual(timing["irregular_episodes_by_sensor"],{"sensor":2})
            self.assertEqual(sum(timing["irregular_episodes_by_split"].values()),2)
            self.assertEqual(report,load_prepared(cfg.prepared_dir))
            for ep in episodes:
                audit = json.loads((Path(cfg.prepared_dir)/ep["label_file"]).with_suffix(".json").read_text())
                self.assertEqual(audit["timestamp_diagnostics"],ep["preprocessing"]["timestamp_diagnostics"])
                self.assertEqual(audit["timestamp_diagnostics"]["exceeds_relative_tolerance"],
                                 ep["preprocessing"]["window_02"]["exceeds_relative_tolerance"])
            cfg.classifier = PhaseConfig(timestamp_relative_tolerance=.2)
            with self.assertRaisesRegex(FileExistsError,"new prepared_dir"):
                prepare_phase4(source,cfg)

    def test_zero_signal_is_retained_and_failed_sensor_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = synthetic_source(root)
            cfg = PhasePipelineConfig(data_root=str(root), sensors=["sensor"], split_reference=None,
                                      prepared_dir=str(root/"zero"), gallery_size=4)
            def zeros(ep, offsets):
                return np.zeros((len(offsets),6),np.float32), np.zeros((len(offsets),12,12),np.float32)
            with patch.object(source,"_sensor_rows",side_effect=zeros):
                report = prepare_phase4(source,cfg)
            self.assertEqual(sum(s["count"] for s in report["anchor_support"].values()),12*(64-19))
            dataset = ManifestDataset(source,root/"zero/manifest.parquet",5)
            self.assertEqual({row["phase"] for row in dataset.rows},{"holding","unknown"})
            assignment = []
            for ep in source.episodes:
                if ep.sequence_id == "seq3":
                    ep.sensor_type = "heldout_only"
                assignment.append(dict(sensor_configuration=ep.sensor_type,sequence_id=ep.sequence_id,
                                       episode_index=ep.episode_index,split="test" if ep.sequence_id=="seq3" else
                                       "validation" if ep.episode_index==1 else "train"))
            reference = root/"assignment.json"; reference.write_text(json.dumps({"episode_assignments":assignment}))
            cfg.prepared_dir = str(root/"missing_train"); cfg.split_reference = str(reference)
            cfg.sensors = ["sensor", "heldout_only"]
            with self.assertRaisesRegex(RuntimeError,"Preparation failed"):
                prepare_phase4(source,cfg)
            self.assertFalse(Path(cfg.prepared_dir).exists())
            failures = [json.loads(path.read_text()) for path in root.glob(".phase4_prepare_*/failure.json")]
            self.assertEqual(failures[0]["episode"]["sensor"],"heldout_only")
            self.assertIn("no usable Train",failures[0]["error"])

    def test_prepare_all_n_evaluate_route_and_ui_readers(self):
        from visualization.data import episode_labels, prediction_episode
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = synthetic_source(root)
            cfg = PhasePipelineConfig(data_root=str(root), sensors=["sensor"], split_reference=None,
                                      prepared_dir=str(root/"prepared"), results_dir=str(root/"results"), gallery_size=4)
            with patch.object(source,"_images",side_effect=AssertionError("No decoding in preparation")):
                report = prepare_phase4(source, cfg)
            self.assertEqual(load_prepared(cfg.prepared_dir)["manifest_id"], report["manifest_id"])
            self.assertEqual(report["exclusions"], {"insufficient_history":12*19})
            episodes = json.loads((Path(cfg.prepared_dir)/"episodes.json").read_text())
            self.assertEqual(sum(ep["frames"] for ep in episodes), 12*64)
            normal = torch.load(Path(cfg.prepared_dir)/report["normalization_file"],weights_only=True)["normalization"]
            signature = None; paths = {}; retrieval = []
            for n in (1,3,5,7,10,20):
                dataset = ManifestDataset(source, Path(cfg.prepared_dir)/"manifest.parquet", n)
                current = [(r["sample_id"],r["split"],r["phase"]) for r in dataset.rows]
                self.assertEqual(current, signature or current); signature = current
                self.assertTrue(any(r["phase"] == "unknown" for r in dataset.rows))
                model = tiny_model(n).eval(); model.set_normalization(**normal)
                paths[n] = root/f"results/cohort/N{n}_seed42/evaluation/endpoint"
                provenance = dict(manifest_id=report["manifest_id"], checkpoint_id=f"fresh-synthetic-N{n}", experiment_id="synthetic", normalization_id="train-only",split_id=report["split_id"])
                result = evaluate_endpoints(model,dataset,paths[n],provenance,batch_size=32,alphas=(.1,1.))
                with patch.object(model,"encode_tactile",side_effect=AssertionError("Completed evaluation must be reused")):
                    reused = evaluate_endpoints(model,dataset,paths[n],provenance,batch_size=32,alphas=(.1,1.))
                self.assertEqual(result["evaluation_id"],reused["evaluation_id"])
                self.assertEqual(result["counts"]["validation"], sum(r["split"]=="validation" for r in dataset.rows))
                bank = json.loads((paths[n]/"code_pair_bank.json").read_text())
                self.assertEqual(bank["source_split"],"train")
            routed = aggregate_and_route(paths,root/"analysis")
            self.assertFalse(routed["mean_N_is_latency"])
            self.assertEqual(routed["policy"]["methods"]["ridge"]["phases"]["unknown"]["selected_N"], routed["policy"]["methods"]["ridge"]["global_N"])
            episode = next(ep for ep in episodes if ep["split"] == "validation")
            labels = episode_labels(cfg.prepared_dir,episode)
            self.assertEqual(len(labels["phase"]),64)
            saved = prediction_episode(paths[5],episode,"vq_decoding")
            self.assertEqual(len(saved),45); self.assertEqual(len(saved[0]["predicted_pressure"]),144)
            # Only held-out raw values/files change; Train calibration and normalization stay identical.
            original = source._sensor_rows
            heldout = {(ep["sequence_id"],ep["episode_index"]) for ep in episodes if ep["split"] != "train"}
            for ep in source.episodes:
                if (ep.sequence_id,ep.episode_index) in heldout:
                    ep.parquet.write_text("changed held-out fixture")
            def modified(ep, offsets):
                f,p = original(ep,offsets)
                return (f+1000,p+1000) if (ep.sequence_id,ep.episode_index) in heldout else (f,p)
            cfg.prepared_dir = str(root/"modified")
            with patch.object(source,"_sensor_rows",side_effect=modified):
                changed = prepare_phase4(source,cfg)
            other = torch.load(Path(cfg.prepared_dir)/changed["normalization_file"],weights_only=True)["normalization"]
            for key in normal:
                torch.testing.assert_close(normal[key],other[key],rtol=0,atol=0)
            a = json.loads((root/"prepared/calibration.json").read_text()); b = json.loads((root/"modified/calibration.json").read_text())
            self.assertEqual(a,b)

    def test_runner_dry_run_frozen_config_and_failure_continuation(self):
        from script.phase4 import run_windows, experiment, training_config
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = synthetic_source(root)
            cfg = PhasePipelineConfig(data_root=str(root), sensors=["sensor"], split_reference=None,
                                      prepared_dir=str(root/"prepared"), results_dir=str(root/"runs"), gallery_size=4,
                                      training_config="taf_adapter_sparsh_episode_stable")
            prepare_phase4(source,cfg)
            with contextlib.redirect_stdout(io.StringIO()) as printed, patch("script.phase4.subprocess.run") as launch:
                run_windows(cfg,[1,3],dry_run=True); launch.assert_not_called()
            commands = json.loads(printed.getvalue())["commands"]
            self.assertEqual(set(commands),{"1","3"})
            base = training_config(cfg)
            self.assertEqual(base.vq.initialization,"kmeans")
            self.assertEqual(base.training.scheduler.type,"cosine")
            with patch("script.phase4.select_gpu_zero",return_value="not-used"), \
                    patch("script.phase4.source_for_settings",return_value=source), \
                    patch("script.phase4.subprocess.run",side_effect=subprocess.CalledProcessError(7,["never-launched"])) as launch:
                with self.assertRaisesRegex(RuntimeError,"Some stages failed"):
                    run_windows(cfg,[1,3])
                self.assertEqual(launch.call_count,2)
            _,_,_,out = experiment(cfg)
            state = json.loads((out/"state.json").read_text())
            self.assertEqual(len(state["failures"]),2)
            self.assertTrue(all(r["training"] == "failed" for r in state["N"].values()))
            for n in (1,3):
                frozen = OmegaConf.load(out/f"N{n}_seed42/resolved_training.yaml")
                from wandb import Settings
                # Validate with the actual SDK without init(), login or remote runs.
                Settings(run_tags=tuple(frozen.wandb.tags))
                self.assertTrue(all(1 <= len(tag) <= 64 for tag in frozen.wandb.tags))
                self.assertIn(f"phase4-experiment:{state['experiment_id'][:40]}",frozen.wandb.tags)
                self.assertEqual(frozen.training.batch_size,base.training.batch_size)
                self.assertEqual(frozen.training.gradient_accumulation_steps,base.training.gradient_accumulation_steps)
                self.assertEqual(frozen.training.scheduler,base.training.scheduler)
                self.assertEqual(frozen.vq,base.vq)
                self.assertEqual(frozen.data.window_size,n)
                from hydra import initialize_config_dir, compose
                from utils.common import validate_config
                with initialize_config_dir(config_dir=str((out/f"N{n}_seed42").resolve()), version_base="1.3"):
                    composed = compose(config_name="resolved_training")
                validate_config(composed, require_data_root=True)
                self.assertEqual(composed.data.window_size,n)

    def test_old_manifest_checkpoint_cannot_be_relabelled(self):
        from hydra import compose, initialize_config_dir
        from utils.config import register_config_schema
        from script.phase4 import _assert_checkpoint
        register_config_schema()
        with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1]/"config"),version_base="1.3"):
            cfg = compose(config_name="taf_adapter")
        checkpoint = {"config":OmegaConf.to_container(cfg,resolve=True),
                      "comparison_identity":{"manifest_id":"old-sequence-manifest","window_size":5},"step":10000}
        with self.assertRaisesRegex(ValueError,"manifest/N mismatch"):
            _assert_checkpoint(checkpoint,cfg,"new-phase4-manifest")


if __name__ == "__main__":
    unittest.main()
