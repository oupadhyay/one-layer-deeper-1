import hashlib
import importlib.util
import py_compile
import unittest
from pathlib import Path

import torch

from benchmark import ModelSpec, OptimizerSpec, count_model_state_elements
from submission_validation import validate_submission_source

ROOT = Path(__file__).resolve().parents[1]
V1_PATH = ROOT / "submissions/neural_gpu_e5_remote_v1_t1/submission.py"
PATH = ROOT / "submissions/neural_gpu_e5_remote_v3_t1_batch64/submission.py"
loader = importlib.util.spec_from_file_location("neural_gpu_e5_remote_v3_t1_batch64", PATH)
candidate = importlib.util.module_from_spec(loader)
loader.loader.exec_module(candidate)


def spec(length=64):
    return ModelSpec(17, length, 250_000)


def row(n=(1, 2, 3, 4), x=(5, 6), t=(1,), length=16):
    values = [candidate.N, *(v + candidate.DIGIT for v in n), candidate.X,
              *(v + candidate.DIGIT for v in x), candidate.T,
              *(v + candidate.DIGIT for v in t)]
    return torch.tensor([values + [candidate.PAD] * (length - len(values))])


class NeuralGpuE5RemoteV3T1Batch64Tests(unittest.TestCase):
    def test_exact_source_diff_is_only_batch_size(self):
        source = V1_PATH.read_text(encoding="utf-8")
        expected = source.replace("batch_size=256,", "batch_size=64,")
        self.assertNotEqual(source, expected)
        self.assertEqual(source.count("batch_size=256,"), 1)
        self.assertEqual(PATH.read_text(encoding="utf-8"), expected)

    def test_contract_state_optimizer_and_source(self):
        model = candidate.build_model(spec())
        self.assertEqual(count_model_state_elements(model), 126682)
        self.assertLess(count_model_state_elements(model), 250000)
        bundle = candidate.build_optimizer(model, OptimizerSpec(1, "cpu"))
        self.assertEqual(bundle.optimizer.defaults["lr"], 1e-4)
        self.assertEqual(bundle.optimizer.defaults["betas"], (.9, .95))
        self.assertEqual([g["weight_decay"] for g in bundle.optimizer.param_groups], [.01, 0.])
        self.assertAlmostEqual(bundle.scheduler.get_last_lr()[0], 1e-4 / 30)
        self.assertEqual((candidate.SUBMISSION.batch_size, candidate.SUBMISSION.eval_batch_size,
                          candidate.SUBMISSION.max_steps), (64, 512, None))
        source = PATH.read_text(encoding="utf-8")
        validate_submission_source(PATH.name, source, 256 * 1024, required_filename=None)
        py_compile.compile(str(PATH), doraise=True)
        for forbidden in ("einsum", "remainder", "carry", "anti_diagonal", "lookup", "pair_grid"):
            self.assertNotIn(forbidden, source.lower())

    def test_parser_grid_phases_and_feedback(self):
        model = candidate.build_model(spec()).train()
        mixed = torch.cat((row(), row(n=(8, 7), x=(9,), t=(3,))))
        mask, nd, xd, nm, xm, steps = model.prepare(mixed)
        self.assertEqual(nd[0, :4].tolist(), [4, 3, 2, 1])
        self.assertEqual(xd[0, :4].tolist(), [6, 5, 0, 0])
        self.assertEqual((nm.sum(1).tolist(), xm.sum(1).tolist(), steps.tolist()),
                         ([4, 2], [2, 1], [1, 3]))
        calls = []
        hooks = [model.phase_a.register_forward_hook(lambda *args: calls.append("A")),
                 model.phase_b.register_forward_hook(lambda *args: calls.append("B"))]
        logits, aux = model(mixed)
        for hook in hooks:
            hook.remove()
        self.assertEqual(aux["grid_shape"], (2, 4, 4, 48))
        self.assertEqual(calls, ["A", "B"] * 6)
        self.assertEqual((aux["macrosteps"], aux["microticks_per_macrostep"]), (1, 12))
        logits[0, :, candidate.DIGIT:].sum().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        model.zero_grad(set_to_none=True)
        model(row(t=(3,)))[0].sum().backward()
        self.assertTrue(all(torch.count_nonzero(p.grad) == 0 for p in model.parameters()))

    def test_eval_t64_deterministic_and_pure(self):
        torch.manual_seed(8)
        model = candidate.build_model(spec()).eval()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        ids = row(n=(1, 2, 3, 4), x=(5,), t=(6, 4))
        with torch.no_grad():
            first, aux = model(ids)
            second, _ = model(ids)
        self.assertEqual((aux["parsed_steps"].item(), aux["macrosteps"],
                          aux["phase_a_calls"], aux["scratch_resets"]), (64, 64, 384, 64))
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.isfinite(first).all())
        self.assertTrue(all(torch.equal(v, before[k]) for k, v in model.state_dict().items()))

    def test_source_hash_is_reportable(self):
        self.assertEqual(len(hashlib.sha256(PATH.read_bytes()).hexdigest()), 64)


if __name__ == "__main__":
    unittest.main()
