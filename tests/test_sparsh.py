"""Bounded pairing/config/cache/gradient contract checks; no downloads."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from model.sparsh import build_sparsh_pairs, preprocess_sparsh, load_sparsh_backbone
from model.tactile_encoder import TactileEncoder
from model.taf_adapter import TaFAdapter
from utils.config import register_config_schema
from utils.common import validate_config
from utils.checkpoint import assert_tactile_checkpoint_compatible
from utils.frame_cache import FrozenFrameCache

ROOT = Path(__file__).resolve().parents[1]


def sparsh_config(*overrides):
    register_config_schema()
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "config")):
        return compose(config_name="taf_adapter_sparsh", overrides=list(overrides))


class TinySpatialBackbone(torch.nn.Module):
    """Explicit unit-test double; production never permits random Sparsh."""
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(6, 384) * .01)

    def forward(self, value):
        result = value.mean((-2, -1)) @ self.weight
        return result[:, None].expand(-1, 300, -1)


class SparshTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_all_contexts_pair_exact_local_indices_and_preserve_input(self):
        for n in [1, 3, 5, 7, 10, 20]:
            images = torch.arange(n).view(1, n, 1, 1, 1).expand(2, n, 3, 4, 6).clone()
            before = images.clone()
            paired = build_sparsh_pairs(images)
            self.assertEqual(tuple(paired.shape), (2, n, 6, 4, 6))
            self.assertTrue(torch.equal(images, before))
            for i in range(n):
                reference = max(0, i - 5)
                self.assertLessEqual(reference, i)
                self.assertTrue(torch.equal(paired[:, i, :3], images[:, i]))
                self.assertTrue(torch.equal(paired[:, i, 3:], images[:, reference]))
            if n == 1:
                self.assertTrue(torch.equal(paired[:, :, :3], paired[:, :, 3:]))
            if n == 7:
                self.assertEqual(int(paired[0, -1, 0, 0, 0]), 6)
                self.assertEqual(int(paired[0, -1, 3, 0, 0]), 1)

    def test_preprocessing_range_channels_and_bounded_background(self):
        x = torch.full((1, 3, 3, 32, 24), 64, dtype=torch.uint8)
        x[:, 1] = 128; x[:, 2] = 192
        normal = preprocess_sparsh(x)
        self.assertEqual(tuple(normal.shape), (1, 3, 6, 320, 240))
        self.assertTrue(torch.allclose(normal[:, 1, :3].mean(), torch.tensor(128/255)))
        self.assertTrue(torch.allclose(normal[:, 1, 3:].mean(), torch.tensor(64/255)))
        subtracted = preprocess_sparsh(x, background_subtraction=True)
        self.assertTrue(torch.allclose(subtracted[:, 0].mean(), torch.tensor(127/255)))
        with self.assertRaisesRegex(ValueError, "finite"):
            preprocess_sparsh(x.float())
        with self.assertRaises(ValueError):
            build_sparsh_pairs(x, 0)

    def test_schema_and_incompatible_options_fail_explicitly(self):
        for n in [1, 3, 5, 7, 10, 20]:
            validate_config(sparsh_config(f"data.window_size={n}"))
        for override in ["data.frame_cache_enabled=true", "data.image_size=[224,224]",
                         "tactile_encoder.pretrained=false", "tactile_encoder.sparsh.temporal_stride=0",
                         "tactile_encoder.sparsh.pairing_mode=global", "tactile_encoder.sparsh.pooling=cls"]:
            with self.subTest(override=override), self.assertRaises(ValueError):
                validate_config(sparsh_config(override))

    def test_chunked_pairing_matches_explicit_pairs_across_batch_boundaries(self):
        with patch("model.sparsh.load_sparsh_backbone", return_value=(TinySpatialBackbone(), {})):
            encoder = TactileEncoder(backbone_type="sparsh", backbone="sparsh-dino-small",
                                      backbone_chunk_frames=3)
        value = torch.randint(0, 256, (2, 7, 3, 32, 24), dtype=torch.uint8)
        with torch.no_grad():
            actual = encoder.encode_backbone(value)
            pairs = preprocess_sparsh(value)
            expected = encoder.backbone(pairs.flatten(0, 1)).reshape(2, 7, 300, 384)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_checksum_and_official_identity_fail_without_random_fallback(self):
        with self.assertRaisesRegex(ValueError, "pinned official"):
            load_sparsh_backbone({"repo_id": "facebook/dinov2-small"})
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "bad.safetensors"
            file.write_bytes(b"not official weights")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_sparsh_backbone({"checkpoint_path": str(file)})

    def test_frozen_backbone_keeps_trainable_pool_projection_and_temporal(self):
        with patch("model.sparsh.load_sparsh_backbone", return_value=(TinySpatialBackbone(), {})):
            encoder = TactileEncoder(backbone_type="sparsh", backbone="sparsh-dino-small",
                                      transformer_dim=16, output_dim=16, num_heads=4, dropout=0.)
        encoder.train()
        self.assertFalse(encoder.backbone.training)
        self.assertTrue(encoder.spatial_pool.training)
        value = torch.rand(2, 3, 3, 32, 24)
        output = encoder(value)
        self.assertEqual(tuple(output.shape), (2, 16))
        output.square().mean().backward()
        self.assertTrue(all(p.grad is None for p in encoder.backbone.parameters()))
        for module in [encoder.spatial_pool, encoder.projection, encoder.temporal, encoder.position]:
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()))
        self.assertGreater(float(encoder.summary_token.grad.abs().sum()), 0)
        with self.assertRaisesRegex(ValueError, "incompatible"):
            FrozenFrameCache(SimpleNamespace(), SimpleNamespace(tactile_encoder=encoder), "/tmp/unused_sparsh_cache", device=torch.device("cpu"), amp_enabled=False)

    def test_unfreeze_backbone_and_original_vq_gradient_routing(self):
        with patch("model.sparsh.load_sparsh_backbone", return_value=(TinySpatialBackbone(), {})):
            model = TaFAdapter(window_size=3, force_encoder={"hidden_dim": 32, "pressure_latent_dim": 8, "force_latent_dim": 8},
                               vq={"pressure_codebook_size": 32, "force_codebook_size": 32},
                               tactile_encoder={"backbone_type": "sparsh", "backbone": "sparsh-dino-small",
                                 "freeze_backbone": False, "transformer_dim": 16, "num_heads": 4, "dropout": 0.})
        model.train()
        self.assertTrue(model.tactile_encoder.backbone.training)
        output = model(torch.rand(4,3,3,32,24), torch.randn(4,3,6), torch.randn(4,3,12,12))
        books = [model.force_encoder.pressure.quantizer.codebook.weight, model.force_encoder.force.quantizer.codebook.weight]
        gradients = torch.autograd.grad(output["loss_nce"], books, allow_unused=True, retain_graph=True)
        self.assertTrue(all(g is None for g in gradients))
        output["loss_total"].backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.tactile_encoder.backbone.parameters()))
        self.assertTrue(all(p.grad is not None and p.grad.abs().sum() > 0 for p in books))

    def test_cross_backbone_resume_is_blocked(self):
        config = sparsh_config()
        imagenet = OmegaConf.to_container(config, resolve=True)
        imagenet["tactile_encoder"].update(backbone_type="imagenet", backbone="vit_small_patch16_224.augreg_in21k_ft_in1k")
        with self.assertRaisesRegex(ValueError, "mismatch"):
            assert_tactile_checkpoint_compatible({"config": imagenet}, config)
        with self.assertRaisesRegex(ValueError, "missing pretrained"):
            assert_tactile_checkpoint_compatible({"config": config}, config)


if __name__ == "__main__":
    unittest.main()
