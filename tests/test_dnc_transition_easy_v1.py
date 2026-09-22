import importlib.util
import py_compile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from benchmark import ModelSpec, OptimizerSpec, count_model_state_elements
from submission_validation import validate_submission_source

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "submissions/dnc_transition_easy_v1/submission.py"
loader = importlib.util.spec_from_file_location("dnc_transition_easy_v1", PATH)
candidate = importlib.util.module_from_spec(loader); loader.loader.exec_module(candidate)


def spec(length=16): return ModelSpec(17, length, 999_999)


def rows(t=8):
    ids = torch.tensor([[2, 8, 9, 3, 10, 11, 4, t, 5, 5, 5],
                        [2, 12, 3, 13, 4, t, 5, 5, 0, 0, 0]])
    return ids, ids.ne(0)


class DNCEasyTests(unittest.TestCase):
    def setUp(self): torch.manual_seed(19)

    def test_source_compile_and_policy(self):
        source = PATH.read_text(encoding="utf-8")
        validate_submission_source(PATH.name, source, 256 * 1024)
        py_compile.compile(str(PATH), doraise=True)
        for term in ("teacher_forcing", "argmax", "straight_through", "product_table",
                     "anti_diagonal", "synthetic", "custom_loss", "label_cache"):
            self.assertNotIn(term, source.lower())
        self.assertNotIn("from submissions", source)

    def test_exact_state_config_optimizer_and_contract(self):
        model = candidate.build_model(spec())
        self.assertEqual(count_model_state_elements(model), 241_991)
        self.assertEqual((model.config.vocab_size, model.config.max_seq_len), (17, 16))
        self.assertLess(count_model_state_elements(model), 1_000_000)
        bundle = candidate.build_optimizer(model, OptimizerSpec(2.0, "cpu")); opt = bundle.optimizer
        self.assertEqual((opt.defaults["lr"], opt.defaults["betas"], opt.defaults["eps"]),
                         (3e-4, (.9, .95), 1e-8))
        grouped = [p for g in opt.param_groups for p in g["params"]]
        self.assertEqual(len(grouped), len({id(p) for p in grouped}))
        self.assertEqual({id(p) for p in grouped}, {id(p) for p in model.parameters()})
        self.assertEqual([g["weight_decay"] for g in opt.param_groups], [.01, 0.0])
        self.assertEqual(bundle.scheduler.lr_lambdas[0](49), 1.0)
        cuda = candidate.build_optimizer(model, OptimizerSpec(2.0, "cuda"))
        self.assertTrue(all(g["capturable"] for g in cuda.optimizer.param_groups))
        self.assertEqual((candidate.SUBMISSION.batch_size, candidate.SUBMISSION.eval_batch_size,
                          candidate.SUBMISSION.max_steps), (256, 512, None))

    def test_alignment_composition_parity_and_purity(self):
        model = candidate.build_model(spec()).eval(); ids, mask = rows(9)
        calls = []; hook = model.transition.register_forward_hook(lambda *args: calls.append(1))
        with torch.no_grad(): a, info = model(ids, mask)
        hook.remove()
        self.assertEqual((info["macrosteps"], len(calls)), (2, 2))
        active = a[..., 7:17].ne(candidate.NEG).any(-1)
        self.assertEqual(active[0].nonzero().flatten().tolist(), [9, 10])
        self.assertEqual(active[1].nonzero().flatten().tolist(), [7])
        self.assertTrue((info["memory_norm"] > 0).all())
        before = {k: v.clone() for k, v in model.state_dict().items()}
        with torch.no_grad(): perm, _ = model(ids.flip(0), mask.flip(0)); again, _ = model(ids, mask)
        torch.testing.assert_close(a[0], perm[1]); torch.testing.assert_close(a, again, rtol=0, atol=0)
        mixed = ids.clone(); mixed[1, 5] = 8
        with torch.no_grad(): mixed_out, mixed_info = model(mixed, mask); single, single_info = model(mixed[1:2], mask[1:2])
        self.assertEqual((mixed_info["macrosteps"], single_info["macrosteps"]), (2, 1))
        torch.testing.assert_close(mixed_out[1], single[0], atol=1e-5, rtol=0)
        model.train(); train, _ = model(ids, mask)
        torch.testing.assert_close(a, train, rtol=0, atol=0)
        for k, v in model.state_dict().items(): torch.testing.assert_close(v, before[k], rtol=0, atol=0)

    def test_endpoint_gradients_reach_dnc_components(self):
        model = candidate.build_model(spec()); ids, mask = rows()
        logits, _ = model(ids, mask)
        targets = torch.full_like(ids, -100); targets[0, 9:11] = torch.tensor([8, 9]); targets[1, 7] = 10
        F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=-100).backward()
        prefixes = ("transition.n_embedding", "transition.q_embedding", "transition.phase_embedding",
                    "transition.controller", "transition.interface", "transition.readout")
        for prefix in prefixes:
            grads = [p.grad for n, p in model.named_parameters() if n.startswith(prefix)]
            self.assertTrue(grads and all(g is not None and torch.isfinite(g).all() for g in grads), prefix)
            self.assertGreater(sum(g.abs().sum().item() for g in grads), 0, prefix)

    def test_padding_and_sixty_four_direct_applications(self):
        model = candidate.build_model(spec()).eval(); ids, mask = rows()
        changed = ids.clone(); changed[1, 8:] = torch.tensor([14, 15, 16])
        with torch.no_grad(): a, _ = model(ids, mask); b, _ = model(changed, mask)
        torch.testing.assert_close(a[1], b[1], rtol=0, atol=0)
        n = torch.zeros(1, 1, dtype=torch.long); present = torch.ones(1, 1, dtype=torch.bool)
        q = F.one_hot(torch.ones_like(n), 10).float()
        with torch.no_grad():
            for _ in range(64): q, info = model.transition(n, present, q, present)
        self.assertTrue(torch.isfinite(q).all()); self.assertGreater(info["memory_norm"].item(), 0)
        t64 = torch.tensor([[2, 8, 3, 9, 4, 13, 11]])
        with torch.no_grad(): output, run = model(t64, torch.ones_like(t64, dtype=torch.bool))
        self.assertEqual(run["macrosteps"], 64); self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__": unittest.main()
