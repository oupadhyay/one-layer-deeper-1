from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import torch

from benchmark.s1_c1 import (
    FORBIDDEN_PROJECTED_FIELDS, OneStepHarness, ProjectedDataset, S1GRUTransducer,
    load_eval_rows, load_projected_train,
)


class S1C1Tests(unittest.TestCase):
    def test_exact_architecture_and_shared_modules(self) -> None:
        model = S1GRUTransducer()
        self.assertEqual(model.embedding.num_embeddings, 10)
        self.assertEqual(model.embedding.embedding_dim, 32)
        self.assertEqual(model.cell.input_size, 64)
        self.assertEqual(model.cell.hidden_size, 64)
        self.assertEqual(model.digit_head.in_features, 64)
        self.assertEqual(model.digit_head.out_features, 10)
        self.assertEqual(sum(module is model.cell for module in model.modules()), 1)
        self.assertEqual(sum(module is model.digit_head for module in model.modules()), 1)
        self.assertEqual(sum(module is model.step for module in model.modules()), 1)

    def test_only_fixed_input_slices_reach_shared_embedding(self) -> None:
        model = S1GRUTransducer()
        ids = torch.tensor([[0, 11, 10, 9, 8, 1, 5, 2, 2, 2]])
        state, context = model.canonical_from_input_ids(ids)
        self.assertEqual(state.tolist(), [[9, 8, 7, 6]])
        self.assertEqual(context.tolist(), [[3, 0, 0, 0]])
        logits = model(ids)
        self.assertEqual(logits.shape, (1, 4, 10))
        logits.sum().backward()
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))

    def test_projected_training_items_cannot_expose_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = {"input_ids": [0, 2, 2, 2, 2, 1, 3, 2, 2, 2], "labels": [1, 0, 0, 0],
                   "x0": 0, "a": 1, "y": 1, "T": 1, "split": "train", "context_id": "secret",
                   "trajectory": [0, 1], "carry_length": 0, "carry_lengths": [0], "seed": 1}
            (root / "train.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            opened: list[str] = []
            dataset = load_projected_train(root, opened)
            self.assertEqual(opened, ["train.jsonl"])
            self.assertEqual(set(dataset[0]), {"input_ids", "labels"})
            self.assertFalse(set(dataset[0]) & FORBIDDEN_PROJECTED_FIELDS)
            self.assertEqual(set(vars(dataset)), {"input_ids", "labels"})

    def test_optimization_loader_cannot_open_eval_or_depth(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = {"input_ids": [0, 2, 2, 2, 2, 1, 3, 2, 2, 2], "labels": [1, 0, 0, 0]}
            (root / "train.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            for name in ("test_state", "test_context", "depth_t_2"):
                (root / f"{name}.jsonl").write_text("forbidden", encoding="utf-8")
            opened: list[str] = []
            load_projected_train(root, opened)
            self.assertEqual(opened, ["train.jsonl"])
            with self.assertRaises(ValueError):
                load_eval_rows(root, "depth_t_2", [])

    def test_direct_call_matches_harness_and_eval_is_pure(self) -> None:
        torch.manual_seed(76)
        model = S1GRUTransducer().eval()
        ids = torch.tensor([[0, 11, 11, 11, 11, 1, 3, 2, 2, 2]])
        before = {name: value.clone() for name, value in model.state_dict().items()}
        direct = model(ids)
        harness = OneStepHarness(model).run(ids)
        repeated = model(ids)
        self.assertTrue(torch.equal(direct, harness))
        self.assertTrue(torch.equal(direct, repeated))
        self.assertTrue(all(torch.equal(value, before[name]) for name, value in model.state_dict().items()))


if __name__ == "__main__":
    unittest.main()
