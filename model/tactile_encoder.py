"""ImageNet ViT or window-paired Sparsh, with the shared causal temporal head."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class TactileEncoder(nn.Module):
    def __init__(self, output_dim: int = 256, transformer_dim: int = 256,
                 transformer_layers: int = 2, num_heads: int = 4, dropout: float = 0.1,
                 backbone: str = "vit_small_patch16_224.augreg_in21k_ft_in1k",
                 pretrained: bool = True, freeze_backbone: bool = True,
                 max_window_size: int = 20, backbone_chunk_frames: int = 0,
                 backbone_type: str = "imagenet", sparsh: dict | None = None):
        super().__init__()
        self.backbone_type = backbone_type
        self.supports_frame_cache = backbone_type == "imagenet"
        self.sparsh_options = dict(sparsh or {})
        if backbone_type == "sparsh":
            if not pretrained or backbone != "sparsh-dino-small":
                raise ValueError("Sparsh requires pretrained=true and backbone=sparsh-dino-small; no random fallback")
            from .sparsh import load_sparsh_backbone, SpatialAttentivePool
            if self.sparsh_options.get("pairing_mode", "window_clamp") != "window_clamp":
                raise ValueError("Sparsh only supports window_clamp temporal pairing")
            if self.sparsh_options.get("pooling", "attentive") != "attentive":
                raise ValueError("Sparsh requires trainable attentive spatial pooling")
            self.backbone, self.pretrained_identity = load_sparsh_backbone(self.sparsh_options)
            self.backbone_dim = 384
            self.input_size = (320, 240)
            self.interpolation = "bilinear"
            self.spatial_pool = SpatialAttentivePool(heads=int(self.sparsh_options.get("pooling_heads", 12)))
            mean, std = (0., 0., 0.), (1., 1., 1.)
        elif backbone_type == "imagenet":
            try:
                import timm
            except ImportError as exc:
                raise ImportError("TactileEncoder requires timm") from exc
            self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
            self.backbone_dim = int(self.backbone.num_features)
            cfg = getattr(self.backbone, "pretrained_cfg", {}) or {}
            self.input_size = tuple(cfg.get("input_size", (3, 224, 224))[-2:])
            self.interpolation = str(cfg.get("interpolation", "bicubic"))
            mean, std = cfg.get("mean", (0.485, 0.456, 0.406)), cfg.get("std", (0.229, 0.224, 0.225))
            self.pretrained_identity = {"backbone": backbone, "pretrained": pretrained}
        else:
            raise ValueError(f"Unsupported tactile backbone_type: {backbone_type}")
        self.backbone_name = backbone
        self.freeze_backbone = freeze_backbone
        self.projection = nn.Linear(self.backbone_dim, transformer_dim)
        self.register_buffer("pixel_mean", torch.tensor(mean).view(1, 1, 3, 1, 1))
        self.register_buffer("pixel_std", torch.tensor(std).view(1, 1, 3, 1, 1))
        if freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()
        self.summary_token = nn.Parameter(torch.zeros(1, 1, transformer_dim))
        nn.init.trunc_normal_(self.summary_token, std=0.02)
        self.position = nn.Embedding(max_window_size + 1, transformer_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim, nhead=num_heads, dim_feedforward=4 * transformer_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=transformer_layers, enable_nested_tensor=False)
        self.output = nn.Linear(transformer_dim, output_dim) if transformer_dim != output_dim else nn.Identity()
        self.max_window_size = max_window_size
        if backbone_chunk_frames < 0:
            raise ValueError("backbone_chunk_frames must be nonnegative")
        self.backbone_chunk_frames = backbone_chunk_frames

    @staticmethod
    def causal_mask(window_size: int, device: torch.device | None = None) -> torch.Tensor:
        """True means masked; final summary position can read every image token."""
        length = window_size + 1
        mask = torch.triu(torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1)
        return mask

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, tactile: torch.Tensor) -> torch.Tensor:
        features = self.encode_frames(tactile)
        return self.aggregate_features(features)

    def encode_frames(self, tactile: torch.Tensor) -> torch.Tensor:
        """Return projected per-frame ViT features with shape ``(B,T,D)``."""
        return self.project_backbone_features(self.encode_backbone(tactile))

    def encode_backbone(self, tactile: torch.Tensor) -> torch.Tensor:
        """Return only frozen ViT outputs, before the trainable projection."""
        if tactile.ndim != 5 or tactile.shape[2] != 3:
            raise ValueError(f"Expected tactile (B,T,3,H,W), got {tuple(tactile.shape)}")
        batch, steps = tactile.shape[:2]
        if steps > self.max_window_size:
            raise ValueError(f"T={steps} exceeds max_window_size={self.max_window_size}")
        if self.backbone_type == "sparsh":
            from .sparsh import preprocess_frame_pairs
            stride = int(self.sparsh_options.get("temporal_stride", 5))
            chunk = self.backbone_chunk_frames if self.freeze_backbone and self.backbone_chunk_frames else batch * steps
            parts = []
            for offset in range(0, batch * steps, chunk):
                positions = torch.arange(offset, min(offset + chunk, batch * steps), device=tactile.device)
                batches, current = positions // steps, positions % steps
                reference = (current - stride).clamp_min(0)
                background = tactile[batches, 0] if self.sparsh_options.get("background_subtraction", False) else None
                pairs = preprocess_frame_pairs(tactile[batches, current], tactile[batches, reference], background)
                if self.freeze_backbone:
                    with torch.no_grad():
                        parts.append(self.backbone(pairs))
                else:
                    parts.append(self.backbone(pairs))
            return torch.cat(parts).reshape(batch, steps, 300, self.backbone_dim)
        flat = tactile.reshape(batch * steps, 3, *tactile.shape[-2:])
        chunk = self.backbone_chunk_frames if self.freeze_backbone and self.backbone_chunk_frames else len(flat)
        parts = []
        for offset in range(0, len(flat), chunk):
            # Convert only one chunk after transfer, avoiding a full float32 image batch.
            x = flat[offset:offset + chunk]
            x = x.float().div(255) if x.dtype == torch.uint8 else x
            if tuple(x.shape[-2:]) != self.input_size:
                x = F.interpolate(x, size=self.input_size, mode=self.interpolation,
                                  align_corners=False, antialias=True)
            x = (x - self.pixel_mean[:, 0]) / self.pixel_std[:, 0]
            if self.freeze_backbone:
                with torch.no_grad():
                    parts.append(self.backbone(x))
            else:
                parts.append(self.backbone(x))
        features = torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]
        return features.reshape(batch, steps, -1)

    def project_backbone_features(self, features: torch.Tensor) -> torch.Tensor:
        if self.backbone_type == "sparsh":
            if features.ndim != 4 or features.shape[-2:] != (300, self.backbone_dim):
                raise ValueError("Sparsh requires window-paired spatial tokens (B,N,300,384)")
            batch, steps = features.shape[:2]
            if steps > self.max_window_size:
                raise ValueError(f"T={steps} exceeds max_window_size={self.max_window_size}")
            pooled = self.spatial_pool(features.reshape(batch * steps, 300, self.backbone_dim))
            return self.projection(pooled).reshape(batch, steps, -1)
        if features.ndim != 3 or features.shape[-1] != self.backbone_dim:
            raise ValueError(f"Expected backbone features (B,T,{self.backbone_dim})")
        if features.shape[1] > self.max_window_size:
            raise ValueError(f"T={features.shape[1]} exceeds max_window_size={self.max_window_size}")
        return self.projection(features)

    def checkpoint_identity(self) -> dict:
        identity = {"backbone_type": self.backbone_type, "backbone": self.backbone_name,
                    "pretrained": self.pretrained_identity}
        if self.backbone_type == "sparsh":
            identity["input_protocol"] = {
                "version": 1, "pairing_mode": "window_clamp", "channel_order": "current_RGB,reference_RGB",
                "temporal_stride": int(self.sparsh_options.get("temporal_stride", 5)),
                "background_subtraction": bool(self.sparsh_options.get("background_subtraction", False)),
                "background_reference": "window_first_frame", "pooling": "attentive_patch_only",
                "pooling_heads": int(self.sparsh_options.get("pooling_heads", 12)),
                "resolution": self.input_size, "range": [0, 1], "normalization": "none",
                "geometry": "official_orientation_crop_tensor_bilinear_antialias"}
        return identity

    def aggregate_features(self, features: torch.Tensor, return_all_tokens: bool = False) -> torch.Tensor:
        """Causally aggregate ``(B,T,D)`` frame features; the summary token is last."""
        if features.ndim != 3 or features.shape[-1] != self.projection.out_features:
            raise ValueError(
                f"Expected frame features (B,T,{self.projection.out_features}), "
                f"got {tuple(features.shape)}"
            )
        batch, steps = features.shape[:2]
        if steps > self.max_window_size:
            raise ValueError(f"T={steps} exceeds max_window_size={self.max_window_size}")
        summary = self.summary_token.expand(batch, -1, -1)
        tokens = torch.cat((features, summary), dim=1)
        positions = torch.arange(steps + 1, device=tokens.device)
        tokens = tokens + self.position(positions).unsqueeze(0)
        encoded = self.temporal(tokens, mask=self.causal_mask(steps, tokens.device))
        if return_all_tokens:
            return encoded
        return self.output(encoded[:, -1])
