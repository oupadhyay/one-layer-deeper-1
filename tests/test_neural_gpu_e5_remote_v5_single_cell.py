import hashlib
import importlib.util
import py_compile
import unittest
from pathlib import Path

import torch

from benchmark import ModelSpec, OptimizerSpec, count_model_state_elements
from submission_validation import validate_submission_source

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "submissions/neural_gpu_e5_remote_v5_single_cell/submission.py"
loader = importlib.util.spec_from_file_location("neural_gpu_e5_remote_v5_single_cell", PATH)
candidate = importlib.util.module_from_spec(loader)
loader.loader.exec_module(candidate)


def spec(length=64):
    return ModelSpec(17, length, 250_000)


def row(n=(1, 2, 3, 4), x=(5, 6), t=(1,), length=16):
    values = [candidate.N, *(v + candidate.DIGIT for v in n), candidate.X,
              *(v + candidate.DIGIT for v in x), candidate.T,
              *(v + candidate.DIGIT for v in t)]
    return torch.tensor([values + [candidate.PAD] * (length - len(values))])


class NeuralGpuE5RemoteV5SingleCellTests(unittest.TestCase):
    def test_one_cell_module_and_unique_parameter_identity(self):
        model = candidate.build_model(spec())
        cells = [module for module in model.modules() if isinstance(module, candidate.CGRUCell)]
        self.assertEqual(cells, [model.cell])
        parameters = list(model.parameters())
        self.assertEqual(len(parameters), len({id(parameter) for parameter in parameters}))
        self.assertFalse(hasattr(model, "phase_a"))
        self.assertFalse(hasattr(model, "phase_b"))

    def test_state_optimizer_validation_and_source(self):
        model = candidate.build_model(spec())
        self.assertEqual(count_model_state_elements(model), candidate.STATE_ELEMENTS)
        self.assertEqual(candidate.STATE_ELEMENTS, 64_330)
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
        self.assertNotIn("phase_a", source.lower())
        self.assertNotIn("phase_b", source.lower())
        self.assertEqual(source.count("self.cell = CGRUCell()"), 1)

    def test_parser_exact_calls_grid_and_training_gate(self):
        model = candidate.build_model(spec()).train()
        mixed = torch.cat((row(), row(n=(8, 7), x=(9,), t=(3,))))
        _mask, nd, xd, nm, xm, steps = model.prepare(mixed)
        self.assertEqual(nd[0, :4].tolist(), [4, 3, 2, 1])
        self.assertEqual(xd[0, :4].tolist(), [6, 5, 0, 0])
        self.assertEqual((nm.sum(1).tolist(), xm.sum(1).tolist(), steps.tolist()),
                         ([4, 2], [2, 1], [1, 3]))
        calls = []
        hook = model.cell.register_forward_hook(lambda *args: calls.append(id(model.cell)))
        logits, aux = model(mixed)
        hook.remove()
        self.assertEqual(aux["grid_shape"], (2, 4, 4, 48))
        self.assertEqual(len(calls), 12)
        self.assertEqual(set(calls), {id(model.cell)})
        self.assertEqual((aux["macrosteps"], aux["microticks_per_macrostep"],
                          aux["cell_calls"]), (1, 12, 12))
        logits[0, :, candidate.DIGIT:].sum().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.parameters()))
        model.zero_grad(set_to_none=True)
        model(row(t=(3,)))[0].sum().backward()
        self.assertTrue(all(torch.count_nonzero(p.grad) == 0 for p in model.parameters()))

    def test_eval_max_t_deterministic_pure_and_exact_calls(self):
        torch.manual_seed(8)
        model = candidate.build_model(spec()).eval()
        before = {key: value.clone() for key, value in model.state_dict().items()}
        ids = row(n=(1, 2, 3, 4), x=(5,), t=(6, 4))
        calls = []
        hook = model.cell.register_forward_hook(lambda *args: calls.append(1))
        with torch.no_grad():
            first, aux = model(ids)
        hook.remove()
        with torch.no_grad():
            second, _ = model(ids)
        self.assertEqual((aux["parsed_steps"].item(), aux["macrosteps"], aux["cell_calls"],
                          aux["scratch_resets"]), (64, 64, 768, 64))
        self.assertEqual(len(calls), 768)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.isfinite(first).all())
        self.assertTrue(all(torch.equal(value, before[key])
                            for key, value in model.state_dict().items()))

    def test_source_hash_is_reportable(self):
        self.assertEqual(len(hashlib.sha256(PATH.read_bytes()).hexdigest()), 64)


if __name__ == "__main__":
    unittest.main()
