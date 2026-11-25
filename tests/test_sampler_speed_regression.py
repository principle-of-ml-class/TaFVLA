"""Check that the fast strict packer preserves the exact legacy batch plan."""
import random
import unittest

from utils.data import (
    FixedValidationBatchSampler,
    EpochShuffledBatchSampler,
    temporal_batch_violations,
)


class Anchors:
    def __init__(self, items, n=5):
        self.episodes = [object() for _ in range(max(e for _, e, _ in items) + 1)]
        self.items = items
        self.n = n

    def _locate(self, index):
        _, e, anchor = self.items[index]
        return self.episodes[e], anchor - self.n + 1


def legacy_plan(items, batch_size, gap, seed):
    remaining = list(items)
    random.Random(seed).shuffle(remaining)
    batches, dropped = [], []
    while len(remaining) >= batch_size:
        selected, deferred = [], []
        for pos, item in enumerate(remaining):
            if all(item[1] != other[1] or abs(item[2] - other[2]) >= gap
                   for other in selected):
                selected.append(item)
            else:
                deferred.append(item)
            if len(selected) == batch_size:
                deferred.extend(remaining[pos + 1:])
                break
        if len(selected) == batch_size:
            batches.append([item[0] for item in selected])
        else:
            dropped.extend(selected)
        remaining = deferred
    dropped.extend(remaining)
    return batches, [item[0] for item in dropped]


class SamplerSpeedTests(unittest.TestCase):
    def test_identical_legacy_batches_and_drops(self):
        for seed in range(10):
            rng = random.Random(seed)
            items = [(i, rng.randrange(12), rng.randrange(19, 200)) for i in range(300)]
            for n in (1, 5, 20):
                dataset = Anchors(items, n)
                indices = list(range(len(items)))
                for size in (4, 16, 32):
                    expected, dropped = legacy_plan(items, size, 20, seed)
                    if not expected:
                        continue
                    sampler = FixedValidationBatchSampler(dataset, indices, size, 20, seed)
                    self.assertEqual(list(sampler), expected)
                    self.assertEqual(sampler.dropped_local_indices, dropped)
                    self.assertFalse(temporal_batch_violations(dataset, indices, expected, 20))

    def test_epoch_sampler_matches_seeded_legacy(self):
        items = [(i, i // 40, 19 + i % 40) for i in range(400)]
        dataset = Anchors(items)
        sampler = EpochShuffledBatchSampler(dataset, list(range(len(items))), 8, 20, 42)
        for epoch in (0, 1, 2, 1):
            sampler.set_epoch(epoch)
            expected, dropped = legacy_plan(items, 8, 20, 42 + epoch)
            self.assertEqual(list(sampler), expected)
            self.assertEqual(sampler.dropped_local_indices, dropped)

    def test_audit_checks_interleaved_episodes(self):
        items = [(0, 0, 20), (1, 1, 20), (2, 0, 21), (3, 0, 40), (4, 1, 22)]
        dataset = Anchors(items)
        violations = temporal_batch_violations(dataset, list(range(len(items))), [list(range(len(items)))], 20)
        self.assertEqual([(x['left_local_index'], x['right_local_index']) for x in violations],
                         [(0, 2), (1, 4), (2, 3)])


if __name__ == '__main__':
    unittest.main()
