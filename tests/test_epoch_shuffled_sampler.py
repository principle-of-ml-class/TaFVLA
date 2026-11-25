"""Regression tests for epoch-varying strict comparison batches."""

import unittest

from utils.data import EpochShuffledBatchSampler, FixedValidationBatchSampler, temporal_batch_violations


class FakeCommonAnchors:
    def __init__(self, window_size):
        self.episodes = [object() for _ in range(8)]
        anchors = (19, 20, 39, 40, 59, 60)
        self.locations = [(ep, a - window_size + 1) for ep in self.episodes for a in anchors]

    def _locate(self, index):
        return self.locations[index]


class EpochShuffledSamplerTests(unittest.TestCase):
    def test_reproducible_strict_full_batches_across_epochs_and_windows(self):
        plans = {}
        for n in (1, 5, 20):
            dataset = FakeCommonAnchors(n)
            indices = list(range(len(dataset.locations)))
            sampler = EpochShuffledBatchSampler(dataset, indices, batch_size=4, min_gap=20, seed=42)
            validation = FixedValidationBatchSampler(dataset, indices, batch_size=4, min_gap=20, seed=42)
            fixed_plan = list(validation)
            for epoch in (0, 1, 2, 1, 0):
                sampler.set_epoch(epoch)
                plan = list(sampler)
                key = epoch
                if key in plans:
                    self.assertEqual(plan, plans[key])
                else:
                    plans[key] = plan
                self.assertTrue(plan)
                self.assertEqual(len(sampler), len(plan))
                self.assertTrue(all(len(batch) == 4 for batch in plan))
                self.assertEqual(temporal_batch_violations(dataset, indices, plan, 20), [])
                used = [i for batch in plan for i in batch]
                self.assertEqual(len(used), len(set(used)))
                self.assertEqual(sampler.report.used_samples, len(used))
                self.assertEqual(sampler.report.dropped_samples, len(indices)-len(used))
                self.assertEqual(sampler.report.seed, 42 + epoch)
                validation.set_epoch(epoch)
                self.assertEqual(list(validation), fixed_plan)
        self.assertNotEqual(plans[0], plans[1])
        self.assertNotEqual(plans[1], plans[2])


if __name__ == "__main__":
    unittest.main()
