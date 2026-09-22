import importlib.util
import py_compile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from benchmark import ModelSpec, OptimizerSpec, count_model_state_elements
from submission_validation import validate_submission_source

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "submissions/autoregressive_digit_decoder_easy_v1/submission.py"
spec = importlib.util.spec_from_file_location("autoregressive_digit_decoder_easy_v1", PATH)
candidate = importlib.util.module_from_spec(spec); spec.loader.exec_module(candidate)


def model_spec(length=13):
    return ModelSpec(17, length, 500_000_000)


def rows(t_digit=8):
    # widths two and one; answer placeholders are at each row's valid tail.
    ids = torch.tensor([[2, 8, 9, 3, 10, 11, 4, t_digit, 5, 5, 5],
                        [2, 12, 3, 13, 4, t_digit, 5, 5, 0, 0, 0]])
    return ids, ids.ne(0)


class CandidateTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_source_policy_and_compile(self):
        source = PATH.read_text(encoding="utf-8")
        validate_submission_source(PATH.name, source, 256 * 1024)
        py_compile.compile(str(PATH), doraise=True)
        for forbidden in ("argmax", "teacher_forcing", "backward(", "while ", "pow(",
                          "synthetic", "quotient", "carry", "subtraction", "reconstruct"):
            self.assertNotIn(forbidden, source.lower())

    def test_state_cap_optimizer_and_contract(self):
        a, b = candidate.build_model(model_spec(13)), candidate.build_model(model_spec(64))
        self.assertEqual(count_model_state_elements(a), 2_736_960)
        self.assertEqual(count_model_state_elements(b), 2_736_960)
        self.assertEqual((a.config.vocab_size, a.config.max_seq_len), (17, 13))
        with self.assertRaises(ValueError): candidate.build_model(model_spec(65))
        bundle = candidate.build_optimizer(a, OptimizerSpec(10, "cpu"))
        opt = bundle.optimizer
        self.assertEqual((opt.defaults["lr"], opt.defaults["betas"], opt.defaults["eps"]),
                         (6e-4, (.9, .95), 1e-8))
        self.assertEqual([g["weight_decay"] for g in opt.param_groups], [.01, 0.0])
        grouped = [p for g in opt.param_groups for p in g["params"]]
        self.assertEqual({id(p) for p in grouped}, {id(p) for p in a.parameters()})
        self.assertEqual(len(grouped), len({id(p) for p in grouped}))
        decay = {id(p) for p in opt.param_groups[0]["params"]}
        for name, p in a.named_parameters():
            self.assertEqual(id(p) in decay, p.ndim == 2 and "embedding" not in name, name)
        self.assertAlmostEqual(bundle.scheduler.lr_lambdas[0](0), 1 / 32)
        self.assertEqual(bundle.scheduler.lr_lambdas[0](31), 1)
        self.assertEqual((candidate.SUBMISSION.batch_size, candidate.SUBMISSION.eval_batch_size,
                          candidate.SUBMISSION.max_steps), (512, 1024, None))

    def test_parser_is_structural_and_digits_affect_encoding(self):
        ids, mask = rows()
        role, place, width = candidate.Model.parse(ids, mask)
        self.assertEqual(width.tolist(), [2, 1])
        self.assertEqual(role[0].tolist(), [0, 1, 1, 0, 2, 2, 0, 3, 0, 0, 0])
        self.assertEqual(place[0, [1, 2, 4, 5]].tolist(), [1, 0, 1, 0])
        changed = ids.clone(); changed[0, 1] = 16
        r2, p2, w2 = candidate.Model.parse(changed, mask)
        torch.testing.assert_close(role, r2); torch.testing.assert_close(place, p2); torch.testing.assert_close(width, w2)
        model = candidate.build_model(model_spec()).eval()
        with torch.no_grad(): out1, _ = model(ids, mask); out2, _ = model(changed, mask)
        self.assertFalse(torch.equal(out1, out2))

    def test_alignment_width_calls_t_independence_and_purity(self):
        model = candidate.build_model(model_spec()).eval(); ids, mask = rows()
        calls = []
        handles = [block.register_forward_hook(lambda *x: calls.append(1)) for block in model.decoder_blocks]
        with torch.no_grad(): out, aux = model(ids, mask)
        for h in handles: h.remove()
        self.assertEqual(sum(isinstance(x, candidate.DecoderBlock) for x in model.modules()), 2)
        self.assertEqual((aux["decoder_steps"], aux["decoder_block_calls"], len(calls)), (2, 4, 4))
        active = out[..., 7:17].ne(-1e4).any(-1)
        self.assertEqual(active[0].nonzero().flatten().tolist(), [9, 10])
        self.assertEqual(active[1].nonzero().flatten().tolist(), [7])
        changed, _ = rows(16)
        with torch.no_grad(): altered, aux2 = model(changed, mask); again, _ = model(ids, mask)
        self.assertEqual(aux2["decoder_block_calls"], 4)
        self.assertFalse(torch.equal(out, altered)); torch.testing.assert_close(out, again, rtol=0, atol=0)
        model.train(); train, _ = model(ids, mask)
        torch.testing.assert_close(train, out, rtol=0, atol=0)

    def test_endpoint_gradients_soft_feedback_padding_and_batch_purity(self):
        model = candidate.build_model(model_spec()); ids, mask = rows()
        logits, _ = model(ids, mask)
        targets = torch.full(ids.shape, -100); targets[0, 9:11] = torch.tensor([7, 8]); targets[1, 7] = 9
        F.cross_entropy(logits.reshape(-1, 17), targets.reshape(-1), ignore_index=-100).backward()
        for name, p in model.named_parameters():
            self.assertIsNotNone(p.grad, name); self.assertTrue(torch.isfinite(p.grad).all(), name)
            self.assertGreater(p.grad.abs().sum().item(), 0, name)
        model.eval()
        with torch.no_grad():
            batched, _ = model(ids, mask); single, _ = model(ids[:1], mask[:1])
            dirty = ids.clone(); dirty[1, 8:] = torch.tensor([14, 15, 16])
            padded, _ = model(dirty, mask)
        torch.testing.assert_close(batched[0], single[0], atol=2e-5, rtol=0)
        torch.testing.assert_close(batched, padded, atol=2e-5, rtol=0)
        with self.assertRaises(ValueError): model(torch.ones(1, 14, dtype=torch.long))


if __name__ == "__main__": unittest.main()
