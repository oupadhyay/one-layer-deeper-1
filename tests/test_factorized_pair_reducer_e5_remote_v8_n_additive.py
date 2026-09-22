import hashlib
import importlib.util
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from benchmark import ModelSpec, OptimizerSpec, count_model_state_elements
from submission_validation import validate_submission_source

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "submissions/factorized_pair_reducer_e5_remote_v8_n_additive/submission.py"
module_spec = importlib.util.spec_from_file_location("factorized_pair_reducer_v8", PATH)
candidate = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(candidate)


def spec(length=16):
    return ModelSpec(17, length, 250_000)


def row(n=(3, 2, 3), x=(1, 2, 3), step=1, padding=3):
    return torch.tensor([[
        candidate.N, *(d + candidate.DIGIT for d in n),
        candidate.X, *(d + candidate.DIGIT for d in x),
        candidate.T, *(int(d) + candidate.DIGIT for d in str(step)),
        *([candidate.PAD] * padding),
    ]])


class NConditionedPairReducerTests(unittest.TestCase):
    def test_masked_immutable_summary_is_repeated_for_every_pair(self):
        model = candidate.build_model(spec())
        ids = row()
        _, nd, xd, _, _, place_mask = model.prepare(ids)
        probabilities = F.one_hot(xd, 10).float()
        captured = []
        handle = model.n_projection.register_forward_pre_hook(
            lambda _module, args: captured.append(args[0].detach().clone()))
        _, context, context_mask = model._contexts(probabilities, nd, place_mask)
        handle.remove()
        expected_n = model.digit_embedding(nd) + model.relative_place.weight[None] + model.n_role
        expected_summary = (expected_n * place_mask[..., None]).sum(1) / place_mask.sum(1, keepdim=True)
        torch.testing.assert_close(captured[0], expected_summary)
        torch.testing.assert_close(context[:, 16:], expected_n)
        self.assertEqual(context_mask.sum().item(), 12)

        changed_padding = nd.clone()
        changed_padding[:, 3] = 9
        _, changed_context, _ = model._contexts(probabilities, changed_padding, place_mask)
        torch.testing.assert_close(context[:, :16], changed_context[:, :16])

    def test_n_changes_conditioned_pairs_and_output_with_x_fixed(self):
        torch.manual_seed(81)
        model = candidate.build_model(spec()).eval()
        a, b = row(n=(3, 2, 3)), row(n=(4, 2, 3))
        with torch.no_grad():
            out_a = model(a)[0]
            out_b = model(b)[0]
        self.assertFalse(torch.equal(out_a, out_b))

    def test_state_gradients_gate_optimizer_purity_max_t_and_validation(self):
        torch.manual_seed(82)
        model = candidate.build_model(spec())
        self.assertEqual(count_model_state_elements(model), 68_874)
        self.assertEqual(sum(p.numel() for p in model.n_projection.parameters()), 4_160)
        self.assertIsInstance(model.n_projection, torch.nn.Linear)
        self.assertNotIn("n_conditioner", dict(model.named_modules()))
        model.train()
        logits, _ = model(row(padding=3))
        logits[:, :, candidate.DIGIT:candidate.DIGIT + 10].sum().backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        model.zero_grad(set_to_none=True)
        model(row(step=2, padding=3))[0][:, :, candidate.DIGIT:candidate.DIGIT + 10].sum().backward()
        self.assertTrue(all(torch.count_nonzero(p.grad).item() == 0 for p in model.parameters()))

        model.eval()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        with torch.no_grad():
            output, aux = model(row(step=64, padding=2))
        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual((aux["parsed_steps"].item(), aux["reducer_calls"]), (64, 64))
        self.assertTrue(all(torch.equal(v, before[k]) for k, v in model.state_dict().items()))
        with self.assertRaises(ValueError):
            model(torch.zeros((1, model.max_seq_len + 1), dtype=torch.long))

        bundle = candidate.build_optimizer(model, OptimizerSpec(1, "cpu"))
        group = bundle.optimizer.param_groups[0]
        self.assertEqual((group["lr"], group["betas"], group["weight_decay"]),
                         (1e-4, (.9, .95), .1))
        self.assertIsNone(bundle.scheduler)
        self.assertEqual((candidate.D, candidate.MICROPHASES, candidate.MAX_STEPS), (64, 1, 64))
        self.assertEqual((candidate.SUBMISSION.batch_size, candidate.SUBMISSION.eval_batch_size,
                          candidate.SUBMISSION.max_steps), (512, 512, None))

        source = PATH.read_text()
        validate_submission_source(PATH.name, source, 256 * 1024, required_filename=None)
        contexts = source[source.index("    def _contexts"):source.index("    def forward")]
        for forbidden in ("remainder", "diagonal", "einsum", "lookup", "routing", "%"):
            self.assertNotIn(forbidden, contexts.lower())
        self.assertEqual(hashlib.sha256(source.encode()).hexdigest(),
                         hashlib.sha256(PATH.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
