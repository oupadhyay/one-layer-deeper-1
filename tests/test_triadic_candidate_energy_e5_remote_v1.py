import importlib.util
import unittest
from pathlib import Path

import torch

from benchmark import ModelSpec, OptimizerSpec, count_model_state_elements
from submission_validation import validate_submission_source

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "submissions/triadic_candidate_energy_e5_remote_v1/submission.py"
LOADER = importlib.util.spec_from_file_location("triadic_candidate", PATH)
candidate = importlib.util.module_from_spec(LOADER)
LOADER.loader.exec_module(candidate)


def spec(length=64):
    return ModelSpec(17, length, 250_000)


def row(n=(1, 2, 3), x=(4, 5, 6), t=(1,), length=16):
    values = [candidate.N, *(d + candidate.DIGIT for d in n), candidate.X,
              *(d + candidate.DIGIT for d in x), candidate.T,
              *(d + candidate.DIGIT for d in t)]
    return torch.tensor(values + [candidate.PAD] * (length - len(values)))


class TriadicCandidateTests(unittest.TestCase):
    def test_source_state_optimizer(self):
        source = PATH.read_text()
        validate_submission_source(PATH.name, source, 256 * 1024)
        for text in ("einsum", "argmax", "detach", "scheduler", "dropout", "remainder", "carry"):
            self.assertNotIn(text, source.lower())
        model = candidate.build_model(spec())
        self.assertEqual(count_model_state_elements(model), 118240)
        self.assertLess(118240, 250000)
        bundle = candidate.build_optimizer(model, OptimizerSpec(1, "cpu"))
        opt = bundle.optimizer
        self.assertEqual((opt.defaults["lr"], opt.defaults["betas"], opt.defaults["eps"]),
                         (5e-4, (.9, .98), 1e-8))
        self.assertEqual([g["weight_decay"] for g in opt.param_groups], [.02, 0.0])
        grouped = [id(p) for g in opt.param_groups for p in g["params"]]
        self.assertEqual(len(grouped), len(set(grouped)))
        self.assertEqual(set(grouped), {id(p) for p in model.parameters() if p.requires_grad})
        self.assertIsNone(bundle.scheduler)
        self.assertEqual((candidate.SUBMISSION.batch_size, candidate.SUBMISSION.eval_batch_size,
                          candidate.SUBMISSION.max_steps), (512, 512, None))

    def test_parser_order_sensitivity_alignment_and_cap(self):
        model = candidate.build_model(spec())
        ids = torch.stack((row(), row(n=(3, 1, 2), x=(6, 5, 4))))
        _, nd, xd, nm, _, steps = model.prepare(ids)
        self.assertEqual(nd[0, :3].tolist(), [3, 2, 1])
        self.assertEqual(xd[0, :3].tolist(), [6, 5, 4])
        probs = torch.zeros(2, 3, 10).scatter(2, xd[:, :3, None], 1)
        memory, _ = model.triadic_memory(probs, nd[:, :3], nm[:, :3])
        self.assertFalse(torch.equal(memory[0], memory[1]))
        self.assertEqual(steps.tolist(), [1, 1])
        with self.assertRaises(ValueError):
            model(torch.ones(1, 65, dtype=torch.long))

    def test_shape_alignment_purity_and_zero_energy(self):
        model = candidate.build_model(spec()).eval()
        single = row(length=18)[None]
        other = row(n=(7,), x=(8,), length=18)[None]
        with torch.no_grad():
            a, _ = model(single)
            mixed, _ = model(torch.cat((single, other)))
        self.assertEqual(a.shape, (1, 18, 17))
        self.assertTrue(torch.allclose(a, mixed[:1], atol=2e-7, rtol=2e-6))
        query = torch.randn(2, 30, candidate.D)
        self.assertTrue(torch.equal(model.candidate_energies(query, torch.zeros_like(query)),
                                    torch.zeros(2, 30)))

    def test_endpoint_backward_reaches_every_component(self):
        model = candidate.build_model(spec()).train()
        ids = row()[None]
        logits, _ = model(ids)
        torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 17), ids[:, 1:].reshape(-1)).backward()
        names = ("digit_embedding", "place_embedding", "pair_mlp", "triad_mlp",
                 "cross_attention.q", "cross_attention.kv", "cross_attention.out",
                 "query_norm", "value_norm", "query_projection", "value_projection")
        for prefix in names:
            gradients = [p.grad for n, p in model.named_parameters() if n.startswith(prefix)]
            self.assertTrue(gradients and all(g is not None and torch.isfinite(g).all() and
                                              torch.count_nonzero(g) for g in gradients), prefix)

    def test_tied_actual_t_train_eval_equality_and_t64(self):
        model = candidate.build_model(spec()).train()
        ids = torch.stack((row(t=(1,), length=18), row(t=(2,), length=18), row(t=(3,), length=18)))
        train_logits, aux = model(ids)
        self.assertEqual((aux["macrosteps"], aux["transition_calls"]), (3, 3))
        model.eval()
        before = {n: p.clone() for n, p in model.state_dict().items()}
        with torch.no_grad():
            eval_logits, eval_aux = model(ids)
            long_a, long_aux = model(row(n=(1,), x=(2,), t=(6, 4), length=10)[None])
            long_b, _ = model(row(n=(1,), x=(2,), t=(6, 4), length=10)[None])
        self.assertTrue(torch.equal(train_logits, eval_logits))
        self.assertEqual(eval_aux["macrosteps"], 3)
        self.assertEqual((long_aux["parsed_steps"].item(), long_aux["macrosteps"]), (64, 64))
        self.assertTrue(torch.isfinite(long_a).all() and torch.equal(long_a, long_b))
        self.assertTrue(all(torch.equal(v, model.state_dict()[n]) for n, v in before.items()))


if __name__ == "__main__":
    unittest.main()
