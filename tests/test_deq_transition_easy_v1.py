import importlib.util
import py_compile
import unittest
from pathlib import Path

import torch

from benchmark import ModelSpec, OptimizerSpec, count_model_state_elements
from submission_validation import validate_submission_source

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "submissions/deq_transition_easy_v1/submission.py"
loader = importlib.util.spec_from_file_location("deq_transition_easy_v1", PATH)
candidate = importlib.util.module_from_spec(loader); loader.loader.exec_module(candidate)


def spec(): return ModelSpec(17, 16, 999_999)


def row(t_digits, two=True):
    values = [2, 8, 9, 3, 10, 11, 4] + t_digits + [5, 5] if two else [2, 12, 3, 13, 4] + t_digits + [5]
    return values + [0] * (12 - len(values))


class DEQTransitionEasyTests(unittest.TestCase):
    def setUp(self): torch.manual_seed(31)

    def test_source_compile_and_policy(self):
        source = PATH.read_text(encoding="utf-8")
        validate_submission_source(PATH.name, source, 256 * 1024)
        py_compile.compile(str(PATH), doraise=True)
        for forbidden in ("product_table", "carry", "quotient", "remainder", "anti_diagonal",
                          "teacher_forcing", "custom autograd", "register_hook", "from submissions"):
            self.assertNotIn(forbidden, source.lower())

    def test_exact_state_config_optimizer_and_contract(self):
        model = candidate.build_model(spec())
        self.assertEqual(count_model_state_elements(model), 170_602)
        self.assertEqual((model.config.vocab_size, model.config.max_seq_len), (17, 16))
        bundle = candidate.build_optimizer(model, OptimizerSpec(3.0, "cpu")); opt = bundle.optimizer
        self.assertEqual((opt.defaults["lr"], opt.defaults["betas"], opt.defaults["eps"]),
                         (6e-4, (.9, .95), 1e-8))
        grouped = [p for group in opt.param_groups for p in group["params"]]
        self.assertEqual(len(grouped), len({id(p) for p in grouped}))
        self.assertEqual({id(p) for p in grouped}, {id(p) for p in model.parameters()})
        self.assertEqual([g["weight_decay"] for g in opt.param_groups], [.01, 0.0])
        self.assertEqual(bundle.scheduler.lr_lambdas[0](31), 1.0)
        self.assertEqual((candidate.SUBMISSION.batch_size, candidate.SUBMISSION.eval_batch_size,
                          candidate.SUBMISSION.max_steps), (512, 512, None))
        self.assertIs(candidate.SUBMISSION.training_loss, candidate.token_training_loss)

    def test_mixed_t_purity_and_exact_cell_calls(self):
        model = candidate.build_model(spec()).eval()
        ids = torch.tensor([row([8]), row([9], False), row([13, 11], False)])
        mask = ids.ne(0); calls = []
        handle = model.cell.register_forward_hook(lambda *args: calls.append(1))
        with torch.no_grad(): mixed, info = model(ids, mask)
        handle.remove()
        self.assertEqual(info["macrosteps"], 64)
        self.assertEqual(len(calls), 64 * 8)
        for i in range(2):
            with torch.no_grad(): alone, one_info = model(ids[i:i+1], mask[i:i+1])
            self.assertEqual(one_info["macrosteps"], i + 1)
            torch.testing.assert_close(mixed[i], alone[0], atol=2e-6, rtol=0)

    def test_alignment_padding_width_and_parity(self):
        model = candidate.build_model(spec()).eval()
        ids = torch.tensor([row([8]), row([8], False)]); mask = ids.ne(0)
        with torch.no_grad(): a, _ = model(ids, mask); again, _ = model(ids, mask)
        active = a[..., 7:17].ne(candidate.NEG).any(-1)
        self.assertEqual(active[0].nonzero().flatten().tolist(), [8, 9])
        self.assertEqual(active[1].nonzero().flatten().tolist(), [6])
        changed = ids.clone(); changed[1, 7:] = torch.tensor([16, 15, 14, 13, 12])
        with torch.no_grad(): padded, _ = model(changed, mask)
        torch.testing.assert_close(a[1], padded[1], rtol=0, atol=0)
        torch.testing.assert_close(a, again, rtol=0, atol=0)
        model.train(); train, _ = model(ids, mask)
        torch.testing.assert_close(a, train, rtol=0, atol=0)

    def test_gradient_topology_loss_and_residual(self):
        model = candidate.build_model(spec())
        ids = torch.tensor([row([8])]); mask = ids.ne(0); grad_modes = []
        handle = model.cell.register_forward_hook(lambda module, args, output: grad_modes.append(output.requires_grad))
        logits, auxiliary = model(ids, mask); handle.remove()
        self.assertEqual(grad_modes, [False] * 6 + [True, True])
        residual = auxiliary["equilibrium_residual"]
        self.assertEqual(residual.ndim, 0); self.assertTrue(torch.isfinite(residual))
        labels = torch.full_like(ids, -100); labels[0, 8:10] = torch.tensor([8, 9])
        selected = labels.ne(-100)
        loss = candidate.token_training_loss(logits[selected], labels[selected], auxiliary)
        loss.backward()
        groups = ("digit_embedding", "context_role", "state_role", "cell.self_attention",
                  "cell.cross_attention", "cell.up", "cell.down", "cell.projection", "digit_head")
        for prefix in groups:
            grads = [p.grad for name, p in model.named_parameters() if name.startswith(prefix)]
            self.assertTrue(grads and all(g is not None and torch.isfinite(g).all() for g in grads), prefix)
            self.assertGreater(sum(g.abs().sum().item() for g in grads), 0.0, prefix)
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_max_t64_tiny_forward_finite(self):
        model = candidate.build_model(spec()).eval()
        ids = torch.tensor([row([13, 11], False)]); mask = ids.ne(0)
        with torch.no_grad(): logits, auxiliary = model(ids, mask)
        self.assertEqual(auxiliary["macrosteps"], 64)
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.isfinite(auxiliary["equilibrium_residual"]))


if __name__ == "__main__": unittest.main()
