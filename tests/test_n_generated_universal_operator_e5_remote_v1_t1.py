import importlib.util
from pathlib import Path

import torch

from benchmark import ModelSpec, OptimizerSpec, count_model_state_elements

PATH = Path(__file__).parents[1] / "submissions/n_generated_universal_operator_e5_remote_v1_t1/submission.py"
spec = importlib.util.spec_from_file_location("generated_operator", PATH)
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)


def prompt(n=(1, 2), x=(3, 4), t=1):
    return torch.tensor([[2, *[7 + d for d in n], 3, *[7 + d for d in x],
                          4, *[7 + int(d) for d in str(t)], 5, 5]])


def test_state_contract_optimizer_and_dynamics():
    model = candidate.build_model(ModelSpec(17, 32, 250_000))
    assert count_model_state_elements(model) < 250_000
    a, ia = model(prompt())
    b, ib = model(prompt(n=(2, 1)))
    assert not torch.equal(ia["generated_film"], ib["generated_film"])
    assert not torch.equal(a, b)
    assert ia["universal_block_calls"] == 6 and ia["scratch_resets"] == 1
    bundle = candidate.build_optimizer(model, OptimizerSpec(1, "cpu"))
    assert bundle.optimizer.defaults["betas"] == (0.9, 0.95)
    assert candidate.SUBMISSION.batch_size == 256 and candidate.SUBMISSION.eval_batch_size == 512


def test_gate_gradients_parser_and_eval_t64():
    model = candidate.build_model(ModelSpec(17, 32, 250_000))
    model.train()
    logits, info = model(prompt(t=1))
    logits[:, -2:, 7:17].sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    model.zero_grad(set_to_none=True)
    model(prompt(t=2))[0][:, -2:, 7:17].sum().backward()
    assert all(torch.count_nonzero(p.grad) == 0 for p in model.parameters())
    model.eval()
    with torch.no_grad():
        out, info = model(prompt(t=64))
    assert torch.isfinite(out).all() and info["parsed_steps"].item() == 64
    assert info["universal_block_calls"] == 64 * 6 and info["scratch_resets"] == 64
