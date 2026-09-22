import importlib.util
import py_compile
import unittest
from pathlib import Path

import torch

from benchmark import ModelSpec, OptimizerSpec, count_model_state_elements
from submission_validation import validate_submission_source

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "submissions/neural_gpu_e5_remote_v3_t1_batch64/submission.py"
PATH = ROOT / "submissions/neural_gpu_e5_remote_v4_compositional_batch64/submission.py"
loader = importlib.util.spec_from_file_location("neural_gpu_e5_remote_v4", PATH)
candidate = importlib.util.module_from_spec(loader)
loader.loader.exec_module(candidate)


def spec(length=64):
    return ModelSpec(17, length, 250_000)


def row(n=(1, 2), x=(5, 6), t=2, length=16):
    td = tuple(map(int, str(t)))
    values = [candidate.N, *(v + candidate.DIGIT for v in n), candidate.X,
              *(v + candidate.DIGIT for v in x), candidate.T,
              *(v + candidate.DIGIT for v in td)]
    return torch.tensor([values + [candidate.PAD] * (length - len(values))])


class NeuralGpuE5RemoteV4CompositionalBatch64Tests(unittest.TestCase):
    def test_exact_training_contract_diff(self):
        base = BASE.read_text(encoding="utf-8")
        expected = base.replace(
            "macrosteps = 1 if self.training else int(steps.max().item())",
            "macrosteps = int(steps.max().item())",
        ).replace(
            "        if self.training:\n"
            "            gate = steps.eq(1).to(logits.dtype)[:, None, None]\n"
            "            logits = logits.detach() + gate * (logits - logits.detach())\n",
            "",
        )
        self.assertEqual(PATH.read_text(encoding="utf-8"), expected)

    def test_contract_state_optimizer_and_validation(self):
        model = candidate.build_model(spec())
        self.assertEqual(count_model_state_elements(model), 126_682)
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
        self.assertEqual((candidate.MAX_PLACES, candidate.MAX_STEPS), (16, 64))

    def test_train_eval_parity_calls_and_resets(self):
        torch.manual_seed(12)
        model = candidate.build_model(spec())
        ids = torch.cat((row(n=(1, 2, 3, 4), t=2), row(n=(8, 7), t=3)))
        calls = []
        hooks = [model.phase_a.register_forward_hook(lambda *args: calls.append("A")),
                 model.phase_b.register_forward_hook(lambda *args: calls.append("B"))]
        model.train()
        train_logits, aux = model(ids)
        for hook in hooks:
            hook.remove()
        model.eval()
        with torch.no_grad():
            eval_logits, eval_aux = model(ids)
        self.assertTrue(torch.equal(train_logits, eval_logits))
        self.assertEqual((aux["macrosteps"], aux["microticks_per_macrostep"]), (3, 12))
        self.assertEqual((aux["phase_a_calls"], aux["phase_b_calls"]), (18, 18))
        self.assertEqual(calls, ["A", "B"] * 18)
        self.assertEqual((aux["scratch_resets"], aux["n_reinitializations"]), (3, 3))
        self.assertEqual(eval_aux["macrosteps"], aux["macrosteps"])

    def test_t1_t2_t3_nonzero_plain_ce_gradients(self):
        model = candidate.build_model(spec()).train()
        for steps in (1, 2, 3):
            model.zero_grad(set_to_none=True)
            logits, aux = model(row(t=steps))
            loss = torch.nn.functional.cross_entropy(
                logits[:, -1, candidate.DIGIT:candidate.DIGIT + 10], torch.tensor([3]))
            loss.backward()
            grads = [p.grad for p in model.parameters()]
            self.assertEqual(aux["macrosteps"], steps)
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
            self.assertGreater(sum(g.abs().sum().item() for g in grads), 0.0)

    def test_max_t_deterministic_state_pure(self):
        model = candidate.build_model(spec()).eval()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        with torch.no_grad():
            first, aux = model(row(t=99))
            second, _ = model(row(t=99))
        self.assertEqual((aux["parsed_steps"].item(), aux["macrosteps"],
                          aux["phase_a_calls"], aux["scratch_resets"]), (64, 64, 256, 64))
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.isfinite(first).all())
        self.assertTrue(all(torch.equal(value, before[key])
                            for key, value in model.state_dict().items()))


if __name__ == "__main__":
    unittest.main()
