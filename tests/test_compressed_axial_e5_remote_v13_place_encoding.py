import importlib.util
from pathlib import Path
import torch
from benchmark import ModelSpec, count_model_state_elements
from submission_validation import validate_submission_source

PATH = Path(__file__).resolve().parents[1] / "submissions/compressed_axial_e5_remote_v13_place_encoding/submission.py"
spec = importlib.util.spec_from_file_location("v13_place", PATH)
candidate = importlib.util.module_from_spec(spec); spec.loader.exec_module(candidate)

def inputs(step=1):
    t = [7 + int(x) for x in str(step)]
    return torch.tensor([[2, 8, 9, 10, 3, 11, 12, 4, *t, 5, 5, 5]])

def test_contract_validation_state_and_no_routing():
    source = PATH.read_text()
    validate_submission_source(PATH.name, source, 256 * 1024, required_filename=None)
    assert "einsum" not in source and "diagonal" not in source.lower()
    model = candidate.build_model(ModelSpec(17, 20, 1_000_000))
    assert count_model_state_elements(model) == candidate.STATE_ELEMENTS == 49_018
    assert model.initializer[0].in_features == 102
    assert list(n for n, _ in model.named_parameters()).count("place_embedding.weight") == 1
    assert model.place_embedding.embedding_dim == 16
    assert (candidate.MAX_STEPS, candidate.SUBMISSION.batch_size, candidate.SUBMISSION.eval_batch_size,
            candidate.SUBMISSION.max_steps) == (64, 512, 512, None)

def test_places_distinguishable_shared_and_boundary_canonical():
    model = candidate.build_model(ModelSpec(17, 20, 1_000_000))
    with torch.no_grad():
        model.place_embedding.weight.copy_(torch.arange(21)[:, None].expand(-1, 16))
    probs = torch.nn.functional.one_hot(torch.tensor([[1, 1, 1], [1, 1, 0]]), 10).float()
    nd = torch.tensor([[1, 1, 1], [1, 1, 0]])
    captured = {}
    model.initializer[0].register_forward_pre_hook(lambda m, a: captured.setdefault("x", a[0]))
    _, valid = model._workspace(probs, nd, torch.tensor([3, 2]))
    x = captured["x"]
    assert not torch.equal(x[0, 1, 1, 64:80], x[0, 2, 1, 64:80])
    torch.testing.assert_close(x[0, 1, 1, 64:80], x[0, 1, 1, 80:96])
    torch.testing.assert_close(x[0, 0, 3, 80:96], x[1, 0, 2, 80:96])
    assert x[0, 0, 3, 80].item() == 20 and valid.sum().item() == 18

def test_purity_max_t_and_gradient_gate():
    model = candidate.build_model(ModelSpec(17, 20, 1_000_000))
    before = {n: p.clone() for n, p in model.state_dict().items()}
    out, info = model(inputs(64))
    assert torch.isfinite(out).all() and info["vectorized_cell_calls"] == 256
    for n, p in model.state_dict().items(): torch.testing.assert_close(p, before[n])
    for step in (1, 2, 3):
        model.zero_grad(set_to_none=True)
        logits, _ = model(inputs(step)); logits[:, -3:, 7:17].sum().backward()
        for p in model.parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all()
            if step != 1: assert torch.count_nonzero(p.grad) == 0
