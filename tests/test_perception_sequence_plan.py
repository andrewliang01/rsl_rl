"""Lazy replay preserves eager sequence contents, identity and reset boundaries."""
from dataclasses import fields
import unittest
import torch
from rsl_rl.storage.perception_storage import PerceptionStorage


class SequencePlanTests(unittest.TestCase):
    def test_exact_eager_equivalence_including_history_and_multiple_passes(self):
        torch.set_num_threads(1)
        torch.manual_seed(4)
        storage = PerceptionStorage()
        for step, ids in enumerate(([3, 0, 1, 2], [1, 3, 0], [2, 0, 3, 1], [3, 2])):
            n = len(ids)
            image = torch.randn(n, 2, 8, 16)
            env = torch.tensor(ids)
            episode = ((env == 3) & (step >= 2)).long()
            storage.add(image, image + 1, image + 2, torch.randn(n, 9), env, episode,
                        torch.full((n,), step, dtype=torch.long))
            storage._batches[-1]['history_ids'] = torch.randint(-1, 30, (n, 3))
            storage._batches[-1]['history_age'] = torch.rand(n, 3)
        for steps in (1, 2, 8):
            for batch_size in (1, 3, 32):
                expected = storage.sequence_batches(steps, batch_size)
                plan = storage.sequence_plan(steps, batch_size)
                self.assertEqual(len(plan), len(expected))
                self.assertEqual(plan.anchor_count, storage.num_anchors)
                for _ in range(2):
                    for actual, reference in zip(plan, expected, strict=True):
                        for field in fields(actual):
                            a, b = getattr(actual, field.name), getattr(reference, field.name)
                            if isinstance(a, torch.Tensor):
                                torch.testing.assert_close(a, b, rtol=0, atol=0)
                                self.assertFalse(torch.is_inference(a))
                            else:
                                self.assertEqual(a, b)

    def test_empty_and_invalid_plan(self):
        storage = PerceptionStorage()
        self.assertEqual(list(storage.sequence_plan(8, 128)), [])
        self.assertEqual(storage.sequence_plan(8, 128).anchor_count, 0)
        with self.assertRaises(ValueError):
            storage.sequence_plan(0, 128)

    def test_plan_owns_snapshots_after_storage_cleared(self):
        storage = PerceptionStorage()
        image = torch.ones(2, 2, 8, 16)
        storage.add(image, image, image, torch.zeros(2, 3), torch.arange(2), torch.zeros(2,dtype=torch.long), torch.zeros(2,dtype=torch.long))
        plan = storage.sequence_plan(8, 128)
        storage.clear()
        batch = next(iter(plan))
        torch.testing.assert_close(batch.observed[0], image)


if __name__ == '__main__':
    unittest.main()
