"""Hydra structured schema for the TaF-Adapter experiment configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from hydra.core.config_store import ConfigStore


_REGISTERED = False


@dataclass
class DataConfig:
    root_dir: Optional[str] = None
    window_size: int = 5
    stride: int = 1
    image_size: Optional[list[int]] = field(default_factory=lambda: [224, 224])
    max_sequences: Optional[int] = None
    # Dataset accepts either one sensor name or a list; common validation checks it.
    sensor_types: Any = None
    split_seed: int = 42
    split_unit: str = "sequence"
    validation_fraction: float = 0.1
    skip_invalid_sequences: bool = False
    timestamp_cache_enabled: bool = True
    timestamp_cache_dir: Optional[str] = None
    sensor_arrow_conversion: bool = True
    video_cache_max_entries: int = 2
    batch_group_reads: bool = True
    frame_cache_max_open_files: int = 8
    sensor_cache_max_mb: int = 256
    sensor_mmap_enabled: bool = True
    sensor_mmap_dir: str = "outputs/sensor_mmap"
    sensor_mmap_rebuild: bool = False
    sensor_mmap_max_open_files: int = 16
    sensor_mmap_batch_rows: int = 8192
    frame_cache_enabled: bool = False
    frame_cache_dir: Optional[str] = None
    frame_cache_rebuild: bool = False
    frame_cache_batch_size: int = 32
    show_initialization_progress: bool = True
    alignment_decode_windows_per_episode: int = 1
    normalization_eps: float = 1.0e-6
    normalization_chunk_frames: int = 8192


@dataclass
class ForceEncoderConfig:
    hidden_dim: int = 512
    pressure_latent_dim: int = 128
    force_latent_dim: int = 128
    mlp_layers: int = 2


@dataclass
class VQConfig:
    pressure_codebook_size: int = 512
    force_codebook_size: int = 512
    commitment_beta: float = 0.25


@dataclass
class SparshConfig:
    repo_id: str = "facebook/sparsh-dino-small"
    revision: str = "1aa732e94c1dd58294d4553e139c4bbe90e1ba35"
    filename: str = "dino_vitsmall.safetensors"
    sha256: str = "4126ba669f0fbe62b50893f46185d021f86d7a3dc671ac6397b2e9e3b9aeb8b8"
    checkpoint_path: Optional[str] = None
    temporal_stride: int = 5
    pairing_mode: str = "window_clamp"
    background_subtraction: bool = False
    pooling: str = "attentive"
    pooling_heads: int = 12


@dataclass
class TactileEncoderConfig:
    backbone_type: str = "imagenet"
    backbone: str = "vit_small_patch16_224.augreg_in21k_ft_in1k"
    pretrained: bool = True
    freeze_backbone: bool = True
    transformer_dim: int = 256
    transformer_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1
    max_window_size: int = 20
    backbone_chunk_frames: int = 0
    sparsh: SparshConfig = field(default_factory=SparshConfig)


@dataclass
class LossConfig:
    lambda_quant: float = 1.0
    lambda_recon: float = 1.0
    temperature: float = 0.07
    nce_mode: str = "paper"


@dataclass
class TrainingConfig:
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-5
    max_steps: int = 10000
    num_workers: int = 4
    eval_num_workers: int = 0
    prefetch_factor: int = 1
    seed: int = 42
    amp: bool = True
    grad_clip_norm: float = 1.0
    log_every: int = 20
    validate_every: int = 500
    validation_batches: Optional[int] = 50
    save_every: int = 1000
    output_dir: str = "outputs/taf_adapter"
    resume: Optional[str] = None
    min_same_episode_window_gap: int = 5
    # auto preserves legacy comparison and standard training behavior.
    batch_sampler: str = "auto"


@dataclass
class WandbConfig:
    enabled: bool = False
    project: str = "TaF-Adapter"
    entity: Optional[str] = None
    name: Optional[str] = None
    group: Optional[str] = None
    tags: list[str] = field(default_factory=lambda: ["taf-vla", "pretraining"])
    mode: str = "online"
    log_every: int = 20
    watch_model: bool = False


@dataclass
class AlignmentEvalConfig:
    enabled: bool = True
    retrieval_every_n_steps: int = 1000
    retrieval_every_n_epochs: int = 1
    visualize_every_n_epochs: int = 5
    retrieval_max_pairs: int = 256
    visualization_max_pairs: int = 64
    min_window_start_gap: int = 5
    random_seed: int = 42
    run_on_training_end: bool = True
    save_local_figures: bool = True


@dataclass
class CheckConfig:
    overfit_steps: int = 100
    skip_overfit: bool = False
    no_pretrained: bool = False
    window_sizes: list[int] = field(default_factory=lambda: [1, 3, 10, 5])


@dataclass
class ComparisonConfig:
    enabled: bool = False
    manifest: Optional[str] = None
    sampling_seed: int = 42
    gallery_seed: int = 42
    gallery_max_pairs: int = 256
    min_anchor_gap: int = 20
    window_sizes: list[int] = field(default_factory=lambda: [1, 3, 5, 7, 10, 20])
    training_seeds: list[int] = field(default_factory=lambda: [42, 43, 44])


@dataclass
class PerformanceWarningsConfig:
    enabled: bool = False
    data_wait_ratio_threshold: float = 0.5
    stall_timeout_seconds: float = 600.0
    warning_cooldown_seconds: float = 1800.0
    cpu_rss_warning_gib: Optional[float] = None
    gpu_reserved_warning_gib: Optional[float] = None
    cache_hit_rate_threshold: float = 0.1
    send_to_trainctl: bool = False


@dataclass
class PerformanceConfig:
    enabled: bool = True
    detailed_profiling: bool = False
    log_every: int = 20
    monitor_cpu_memory: bool = True
    monitor_gpu_memory: bool = True
    monitor_throughput: bool = True
    monitor_data_loading: bool = True
    monitor_cache: bool = False
    monitor_gpu_utilization: bool = False
    warnings: PerformanceWarningsConfig = field(default_factory=PerformanceWarningsConfig)


@dataclass
class TrainctlConfig:
    enabled: bool = True
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)


@dataclass
class TaFConfig:
    data: DataConfig = field(default_factory=DataConfig)
    force_encoder: ForceEncoderConfig = field(default_factory=ForceEncoderConfig)
    vq: VQConfig = field(default_factory=VQConfig)
    tactile_encoder: TactileEncoderConfig = field(default_factory=TactileEncoderConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    alignment_eval: AlignmentEvalConfig = field(default_factory=AlignmentEvalConfig)
    check: CheckConfig = field(default_factory=CheckConfig)
    comparison: ComparisonConfig = field(default_factory=ComparisonConfig)
    trainctl: TrainctlConfig = field(default_factory=TrainctlConfig)


def register_config_schema() -> None:
    """Register once per process so YAML values are type checked and struct-locked."""
    global _REGISTERED
    if _REGISTERED:
        return
    ConfigStore.instance().store(name="taf_adapter_schema", node=TaFConfig)
    _REGISTERED = True

