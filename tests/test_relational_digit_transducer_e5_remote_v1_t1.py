import importlib.util
import unittest
from pathlib import Path

import torch

from benchmark import ModelSpec, OptimizerSpec, count_model_state_elements
from submission_validation import validate_submission_source


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "submissions/relational_digit_transducer_e5_remote_v1_t1/submission.py"
LOADER = importlib.util.spec_from_file_location("relational_digit_transducer", PATH)
candidate = importlib.util.module_from_spec(LOADER)
LOADER.loader.exec_module(candidate)


def spec(length=64):
    return ModelSpec(17, length, 250_000)


def row(n=(1, 2, 3), x=(4, 5, 6), t=(1,), length=16):
    values = [candidate.N, *(d + candidate.DIGIT for d in n), candidate.X,
              *(d + candidate.DIGIT for d in x), candidate.T,
              *(d + candidate.DIGIT for d in t)]
    return torch.tensor([values + [candidate.PAD] * (length - len(values))])


class RelationalDigitTransducerTests(unittest.TestCase):
    def test_fixed_state_source_and_optimizer_contract(self):
        torch.manual_seed(9)
        a = candidate.build_model(spec())
        torch.manual_seed(9)
        b = candidate.build_model(spec(128))
        self.assertEqual(count_model_state_elements(a), 160202)
        for name, value in a.state_dict().items():
            self.assertTrue(torch.equal(value, b.state_dict()[name]))
        bundle = candidate.build_optimizer(a, OptimizerSpec(1, "cpu"))
        self.assertEqual(bundle.optimizer.defaults["lr"], 8e-4)
        self.assertEqual(bundle.optimizer.defaults["betas"], (.9, .98))
        self.assertEqual([group["weight_decay"] for group in bundle.optimizer.param_groups], [.01, 0.0])
        self.assertIsNotNone(bundle.scheduler)
        self.assertEqual((candidate.SUBMISSION.batch_size, candidate.SUBMISSION.eval_batch_size,
                          candidate.SUBMISSION.max_steps), (256, 512, None))
        source = PATH.read_text()
        validate_submission_source(PATH.name, source, 256 * 1024, required_filename=None)
        for forbidden in ("einsum", "argmax", "one_hot", "softmax digit", "state_cache"):
            self.assertNotIn(forbidden, source)

    def test_parser_alignment_and_n_order(self):
        model = candidate.build_model(spec())
        ids = torch.cat((row(length=17), row(n=(3, 1, 2, 0), x=(8, 7, 6, 5), t=(3,), length=17)))
        _, nd, xd, nm, xm, steps = model.prepare(ids)
        self.assertEqual(nd[0, :3].tolist(), [3, 2, 1])
        self.assertEqual(xd[1, :4].tolist(), [5, 6, 7, 8])
        self.assertEqual((nm.sum(1).tolist(), xm.sum(1).tolist(), steps.tolist()), ([3, 4], [3, 4], [1, 3]))
        _, h1 = model.encode_n(nd, nm)
        swapped = row(n=(2, 1, 3), length=17)
        _, nd2, _, nm2, _, _ = model.prepare(swapped)
        _, h2 = model.encode_n(nd2, nm2)
        self.assertFalse(torch.equal(h1[:1], h2))

    def test_depth_calls_gate_determinism_and_gradients(self):
        model = candidate.build_model(spec()).train()
        mixed = torch.cat((row(t=(1,)), row(t=(3,))))
        logits, aux = model(mixed)
        self.assertEqual((aux["macrosteps"], aux["memory_block_calls"], aux["scratch_resets"],
                          aux["decoder_calls"]), (1, 2, 1, 3))
        loss = logits[0, :, candidate.DIGIT:candidate.DIGIT + 10].sum()
        loss.backward()
        self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                            for parameter in model.parameters()))
        model.zero_grad(set_to_none=True)
        model(row(t=(3,)))[0].sum().backward()
        self.assertTrue(all(torch.count_nonzero(parameter.grad) == 0 for parameter in model.parameters()))
        model.eval()
        with torch.no_grad():
            first, aux = model(row(t=(3,)))
            second, _ = model(row(t=(3,)))
        self.assertEqual((aux["macrosteps"], aux["memory_block_calls"], aux["decoder_calls"]), (3, 6, 9))
        self.assertTrue(torch.equal(first, second))

    def test_dynamic_modulus_width_activates_high_zero_x_places(self):
        model = candidate.build_model(spec()).eval()
        mixed = torch.cat((row(n=(1, 2, 3, 4), x=(5,), length=16),
                           row(n=(6, 7), x=(8, 9), length=16)))
        _, nd, xd, nmask, raw_xmask, _ = model.prepare(mixed)
        self.assertEqual((nmask.sum(1).tolist(), raw_xmask.sum(1).tolist()), ([4, 2], [1, 2]))
        self.assertEqual(xd[0, :4].tolist(), [5, 0, 0, 0])
        with torch.no_grad():
            _, aux = model(mixed)
        self.assertEqual(aux["active_places"], 4)
        self.assertEqual(aux["widths"].tolist(), [4, 2])
        self.assertEqual(aux["pair_tokens"], 16)
        self.assertEqual(aux["decoder_calls"], 4)
        self.assertLess(aux["pair_tokens"], candidate.MAX_PLACES ** 2)

    def test_eval_t64_finite(self):
        model = candidate.build_model(spec()).eval()
        with torch.no_grad():
            logits, aux = model(row(n=(1,), x=(2,), t=(6, 4), length=10))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertEqual((aux["parsed_steps"].item(), aux["macrosteps"]), (64, 64))
        self.assertEqual((aux["active_places"], aux["pair_tokens"], aux["decoder_calls"]), (1, 1, 64))


if __name__ == "__main__":
    unittest.main()
