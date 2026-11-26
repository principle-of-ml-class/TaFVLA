"""Official Sparsh DINO Small loading, bounded RGB pairing and spatial pooling.

Upstream: facebookresearch/sparsh@fee6a05a97330eca014b59c71ac41596fe3974a7.
The pretrained backbone is strict and immutable in identity; pooling is newly
trainable and follows upstream AttentivePooler(depth=1, complete_block=True).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .sparsh_backbone import SparshVisionTransformer, Mlp

REPO_ID = "facebook/sparsh-dino-small"
REVISION = "1aa732e94c1dd58294d4553e139c4bbe90e1ba35"
FILENAME = "dino_vitsmall.safetensors"
SHA256 = "4126ba669f0fbe62b50893f46185d021f86d7a3dc671ac6397b2e9e3b9aeb8b8"
UPSTREAM_REVISION = "fee6a05a97330eca014b59c71ac41596fe3974a7"


def build_sparsh_pairs(images: torch.Tensor, stride: int = 5) -> torch.Tensor:
    """Only local indices: [current RGB, max(0, current-stride) RGB]."""
    if images.ndim != 5 or images.shape[2] != 3 or images.shape[1] < 1:
        raise ValueError("Sparsh requires nonempty (B,N,3,H,W) windows")
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise ValueError("Sparsh temporal stride must be a positive integer")
    reference = (torch.arange(images.shape[1], device=images.device) - stride).clamp_min(0)
    return torch.cat((images, images[:, reference]), dim=2)


def _preprocess_rgb(images: torch.Tensor, background: torch.Tensor | None = None) -> torch.Tensor:
    """Official deterministic geometry/range, with no ImageNet normalization.

    Upstream PIL bilinear resize is expressed as tensor bilinear antialias resize.
    Optional background is explicitly the first image of THIS window, not an
    upstream external no-contact background. Marker experiments leave it off.
    """
    if images.ndim != 5 or images.shape[2] != 3 or images.shape[1] < 1:
        raise ValueError("Sparsh requires (B,N,3,H,W)")
    if images.dtype == torch.uint8:
        value = images.float().div(255)
    elif images.is_floating_point():
        if not torch.isfinite(images).all() or images.amin() < 0 or images.amax() > 1:
            raise ValueError("Sparsh float RGB pixels must be finite and in [0,1]")
        value = images.float()
    else:
        raise TypeError("Sparsh accepts uint8 or floating RGB pixels in [0,1]")
    if background is not None:
        bg = background.float().div(255) if background.dtype == torch.uint8 else background.float()
        value = ((value - bg + .5).clamp(0, 1) * 255).to(torch.uint8).float().div(255)
    # Exact upstream deterministic load_sample_from_buf orientation/crop rule.
    if value.shape[-2] < value.shape[-1]:
        value = torch.rot90(value, -1, (-2, -1))
    height, width = value.shape[-2:]
    if height / width != 4 / 3:
        crop_height = int(height / (4 / 3))
        top = (height - crop_height) // 2
        value = value[..., top:top + crop_height, :]
    batch, steps = value.shape[:2]
    value = F.interpolate(value.reshape(batch * steps, 3, *value.shape[-2:]),
                          size=(320, 240), mode="bilinear", align_corners=False, antialias=True)
    value = value.reshape(batch, steps, 3, 320, 240)
    return value


def preprocess_sparsh(images: torch.Tensor, *, stride=5, background_subtraction=False) -> torch.Tensor:
    background = images[:, :1] if background_subtraction else None
    return build_sparsh_pairs(_preprocess_rgb(images, background), stride)


def preprocess_frame_pairs(current, reference, background=None):
    """Chunked equivalent of preprocess_sparsh without a full native FP32 window."""
    rgb = torch.stack((current, reference), dim=1)
    prepared = _preprocess_rgb(rgb, background[:, None] if background is not None else None)
    return torch.cat((prepared[:, 0], prepared[:, 1]), dim=1)


def load_sparsh_backbone(options: dict) -> tuple[nn.Module, dict]:
    identity = {"repo_id": options.get("repo_id", REPO_ID), "revision": options.get("revision", REVISION),
                "filename": options.get("filename", FILENAME), "sha256": options.get("sha256", SHA256)}
    expected = dict(repo_id=REPO_ID, revision=REVISION, filename=FILENAME, sha256=SHA256)
    if identity != expected:
        raise ValueError("Only the pinned official facebook/sparsh-dino-small checkpoint is supported")
    default = Path(__file__).resolve().parents[1] / "outputs/pretrained/sparsh-dino-small" / REVISION / FILENAME
    supplied = options.get("checkpoint_path")
    path = Path(supplied).expanduser().resolve() if supplied else default
    if supplied and not path.is_file():
        raise FileNotFoundError(f"Explicit Sparsh checkpoint does not exist: {path}")
    if not path.is_file():
        from huggingface_hub import hf_hub_download
        path = Path(hf_hub_download(repo_id=REPO_ID, filename=FILENAME, revision=REVISION,
                                   local_dir=str(default.parent))).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != SHA256:
        raise ValueError(f"Official Sparsh checkpoint SHA-256 mismatch: {path}")
    from safetensors.torch import load_file
    weights = load_file(str(path), device="cpu")
    backbone = SparshVisionTransformer()
    loaded = backbone.load_state_dict(weights, strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise RuntimeError(f"Sparsh load mismatch: {loaded}")
    if not all(torch.isfinite(v).all() for v in backbone.state_dict().values()):
        raise ValueError("Official Sparsh checkpoint contains nonfinite weights")
    identity.update(path=str(path), upstream_revision=UPSTREAM_REVISION,
                    state_keys=len(weights), missing_keys=loaded.missing_keys,
                    unexpected_keys=loaded.unexpected_keys, embed_dim=384,
                    input_channels=6, patch_size=16, patch_tokens=300, register_tokens=1)
    return backbone, identity


class SpatialAttentivePool(nn.Module):
    """One learned query, cross-attention residual and MLP residual; no registers."""
    def __init__(self, dim=384, heads=12):
        super().__init__()
        self.heads = heads
        self.query_tokens = nn.Parameter(torch.empty(1, 1, dim))
        self.norm1 = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim=dim, hidden_dim=dim * 4)
        self.apply(self._initialize)
        nn.init.trunc_normal_(self.query_tokens, std=.02)
        with torch.no_grad():
            self.proj.weight.div_(2 ** .5)
            self.mlp.fc2.weight.div_(2 ** .5)

    @staticmethod
    def _initialize(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=.02)
            nn.init.zeros_(module.bias)

    def forward(self, patches):
        batch, count, dim = patches.shape
        query = self.query_tokens.expand(batch, -1, -1)
        q = self.q(query).reshape(batch, 1, self.heads, dim // self.heads).transpose(1, 2)
        kv = self.kv(self.norm1(patches)).reshape(batch, count, 2, self.heads, dim // self.heads)
        k, v = kv.permute(2, 0, 3, 1, 4).unbind(0)
        pooled = F.scaled_dot_product_attention(q, k, v, dropout_p=0.).transpose(1, 2).reshape(batch, 1, dim)
        query = query + self.proj(pooled)
        return (query + self.mlp(self.norm2(query))).squeeze(1)
