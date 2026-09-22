import hashlib
import importlib.util
import py_compile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

import competition_loop6_biscan_t1grad as frozen
from benchmark import ModelSpec, OptimizerSpec, count_model_state_elements
from submission_validation import validate_submission_source

ROOT = Path(__file__).resolve().parents[1]
V1_PATH = ROOT / "submissions/biscan_e1_remote_v1_large/submission.py"
PATH = ROOT / "submissions/biscan_e1_remote_v2_batch64/submission.py"
module_spec = importlib.util.spec_from_file_location("biscan_e1_remote_v2_batch64", PATH)
candidate = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(candidate)


def model_spec(length=16):
    return ModelSpec(17, length, 20_000_000)


def row(t=1):
    td = [7 + int(d) for d in str(t)]
    return torch.tensor([[2, 10, 9, 10, 3, 8, 12, 4, *td]])


class BiScanE1RemoteV2Batch64Tests(unittest.TestCase):
    def test_exact_diff_from_v1_is_only_official_batches(self):
        source = V1_PATH.read_text(encoding="utf-8")
        expected = source.replace("batch_size=512, eval_batch_size=512",
                                  "batch_size=64, eval_batch_size=128")
        self.assertNotEqual(source, expected)
        self.assertEqual(PATH.read_text(encoding="utf-8"), expected)

    def test_self_contained_legal_source_and_exact_state(self):
        source = PATH.read_text(encoding="utf-8")
        validate_submission_source(PATH.name, source, 256 * 1024)
        py_compile.compile(str(PATH), doraise=True)
        self.assertNotIn("competition_", source)
        self.assertNotIn("einsum", source)
        model = candidate.build_model(model_spec())
        self.assertEqual(count_model_state_elements(model), 63_178)
        self.assertEqual(set(dict(model.named_modules())),
                         {"", "embedding", "scan", "scan.down_cell", "scan.up_cell", "readout"})

    def test_frozen_identity_parser_alignment_and_feedback(self):
        torch.manual_seed(74)
        reference = frozen.Model(model_spec())
        actual = candidate.Model(model_spec())
        actual.load_state_dict(reference.state_dict())
        ids = torch.cat((row(1), row(2), row(3)))
        for training in (True, False):
            reference.train(training); actual.train(training)
            expected, ea = reference(ids)
            result, ra = actual(ids)
            torch.testing.assert_close(result, expected, rtol=0, atol=0)
            for key in ("parsed_steps", "initial_state_digits", "context_digits", "width_mask",
                        "digit_probabilities"):
                torch.testing.assert_close(ra[key], ea[key], rtol=0, atol=0)
            if not training:
                self.assertTrue(((ra["digit_probabilities"] == 0) |
                                 (ra["digit_probabilities"] == 1)).all())

    def test_gradient_gate_scan_order_and_max_t_purity(self):
        model = candidate.Model(model_spec()).train()
        output, aux = model(row(2))
        target = torch.tensor([1, 2])
        gated = torch.autograd.grad(F.cross_entropy(output[0, -2:, 7:], target),
                                    aux["ungated_logits"], retain_graph=True)[0]
        plain = torch.autograd.grad(F.cross_entropy(aux["ungated_logits"][0, -2:, 7:], target),
                                    aux["ungated_logits"])[0]
        torch.testing.assert_close(gated, plain * .01, rtol=2e-6, atol=2e-8)
        places = []
        original = model.scan.down_cell.forward
        model.scan.down_cell.forward = lambda value, hidden: (places.append(int(value[0, 0])) or
                                                               original(value, hidden))
        state = torch.arange(4.).view(1, 4, 1).expand(-1, -1, 32)
        model.scan(state, torch.zeros_like(state), torch.ones(1, 4, dtype=torch.bool))
        self.assertEqual(places, [3, 2, 1, 0])
        model.eval()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        with torch.no_grad():
            logits, info = model(row(64))
        self.assertEqual(info["executed_macrosteps"].item(), 64)
        self.assertTrue(torch.isfinite(logits).all())
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_optimizer_scheduler_and_official_batches(self):
        model = candidate.Model(model_spec())
        cpu = candidate.build_optimizer(model, OptimizerSpec(60, "cpu"))
        self.assertEqual([(g["lr"], g["betas"], g["weight_decay"], g["capturable"])
                          for g in cpu.optimizer.param_groups],
                         [(4e-5, (.9, .98), .02, False), (4e-5, (.9, .98), 0., False)])
        self.assertEqual(cpu.scheduler.lr_lambdas[0](0), .05)
        self.assertEqual(cpu.scheduler.lr_lambdas[0](19), 1.)
        cuda = candidate.build_optimizer(model, OptimizerSpec(60, "cuda"))
        self.assertTrue(all(group["capturable"] for group in cuda.optimizer.param_groups))
        self.assertEqual((candidate.SUBMISSION.batch_size, candidate.SUBMISSION.eval_batch_size,
                          candidate.SUBMISSION.max_steps), (64, 128, None))

    def test_source_hash_is_reportable(self):
        self.assertEqual(len(hashlib.sha256(PATH.read_bytes()).hexdigest()), 64)


if __name__ == "__main__":
    unittest.main()
