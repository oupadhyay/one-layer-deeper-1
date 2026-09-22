import importlib.util
from pathlib import Path

import pytest
import torch

from benchmark import ModelSpec, OptimizerSpec, count_model_state_elements
from submission_validation import validate_submission_source

ROOT = Path(__file__).resolve().parents[1]
RMS = {
    "sgu_rms_functional_easy_v1": 79_490,
    "sgu_rms_native_easy_v1": 79_490,
    "sgu_rms_input_ln_easy_v1": 79_602,
    "sgu_rms_gate_ln_easy_v1": 79_714,
    "sgu_rms_readout_ln_easy_v1": 79_602,
    "sgu_rms_eps_easy_v1": 79_490,
}
AXIAL = {
    "sgu_axial_einsum_easy_v1": 79_894,
    "sgu_axial_role_place_easy_v1": 79_910,
    "sgu_axial_place_role_easy_v1": 79_906,
    "sgu_axial_serial_easy_v1": 79_894,
    "sgu_axial_second_order_easy_v1": 79_895,
    "sgu_axial_rank1_easy_v1": 79_910,
}
CASES = RMS | AXIAL


def load(name):
    path = ROOT / "submissions" / name / "submission.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return path, module


@pytest.mark.parametrize("name,count", CASES.items())
def test_contract_state_source_and_behavior(name, count):
    path, module = load(name)
    torch.manual_seed(112)
    model = module.build_model(ModelSpec(17, 24, 250_000))
    assert count_model_state_elements(model) == count < 250_000
    assert (module.SUBMISSION.batch_size, module.SUBMISSION.eval_batch_size,
            module.SUBMISSION.max_steps) == (256, 512, None)
    bundle = module.build_optimizer(model, OptimizerSpec(1, "cpu"))
    assert bundle.optimizer.defaults["lr"] == 6e-4
    assert bundle.optimizer.defaults["betas"] == (.9, .95)
    assert bundle.scheduler.get_last_lr()[0] == 6e-4 / 32
    source = path.read_text()
    assert validate_submission_source(path.name, source, 256_000) == "submission.py"
    for forbidden in ("cross_entropy", "argmax", "remainder", "router", "cache",
                      "scaled_dot_product", "conv1d", "covariance", "scan"):
        assert forbidden not in source.lower()

    parsed = torch.tensor([[2, 8, 9, 10, 11, 3, 12, 7, 4, 13, 11, 5, 0, 0]])
    _, place, steps = model.parse(parsed, parsed.ne(0))
    assert (steps.item(), place[0, 1:5].tolist()) == (64, [3, 2, 1, 0])
    model.eval(); output, info = model(parsed)
    assert output.shape == (1, 14, 17) and info["macrosteps"] == 64
    assert (output[0, 8:10, 7:17] > -10_000).all()

    ids = torch.tensor([[2,8,9,10,3,10,4,10,5,0], [2,11,3,12,13,4,10,5,0,0]])
    model.train(); first, info = model(ids); permuted, _ = model(ids.flip(0))
    assert info["macrosteps"] == 3
    torch.testing.assert_close(first, permuted.flip(0))
    model.eval(); evaluated, info = model(ids)
    assert info["macrosteps"] == 3
    torch.testing.assert_close(first, evaluated)
    model.train(); first[:, :, 7:17].sum().backward()
    for parameter_name, parameter in model.named_parameters():
        assert parameter.grad is not None, parameter_name
        assert torch.isfinite(parameter.grad).all(), parameter_name


def test_rms_exact_deltas():
    modules = {name: load(name)[1] for name in RMS}
    for name, module in modules.items():
        model = module.build_model(ModelSpec(17, 24, 250_000)); block = model.transition.block
        layer_norms = [block.norm, block.gate_norm, model.transition.readout_norm]
        expected = {"sgu_rms_input_ln_easy_v1": 0, "sgu_rms_gate_ln_easy_v1": 1,
                    "sgu_rms_readout_ln_easy_v1": 2}.get(name)
        assert [isinstance(x, torch.nn.LayerNorm) for x in layer_norms] == [i == expected for i in range(3)]
        if name == "sgu_rms_eps_easy_v1":
            assert all(x.eps == 1e-5 for x in layer_norms)
        if name == "sgu_rms_functional_easy_v1":
            assert "F.rms_norm" in (ROOT / "submissions" / name / "submission.py").read_text()
        if name == "sgu_rms_native_easy_v1":
            assert "value.float().square()" not in (ROOT / "submissions" / name / "submission.py").read_text()


def test_axial_shapes_initialization_and_topologies():
    blocks = {name: load(name)[1].build_model(ModelSpec(17,24,250_000)).transition.block for name in AXIAL}
    assert blocks["sgu_axial_role_place_easy_v1"].place_operator.shape == (2,4,4)
    assert blocks["sgu_axial_role_place_easy_v1"].role_operator.shape == (2,2)
    assert blocks["sgu_axial_place_role_easy_v1"].place_operator.shape == (4,4)
    assert blocks["sgu_axial_place_role_easy_v1"].role_operator.shape == (4,2,2)
    second = blocks["sgu_axial_second_order_easy_v1"]
    assert second.alpha.shape == () and second.alpha.item() == 0
    rank = blocks["sgu_axial_rank1_easy_v1"]
    assert rank.rank_left.shape == rank.rank_right.shape == (8,)
    assert rank.rank_left.abs().max() <= 1.25e-4 and rank.rank_right.abs().max() <= 1.25e-4
    for block in blocks.values():
        assert block.spatial_bias.shape == (4,2)
        torch.testing.assert_close(block.spatial_bias, torch.ones(4,2))


def test_einsum_is_numerically_identical_to_frozen_axial():
    _, parent_module = load("sgu_axial_easy_v1")
    _, child_module = load("sgu_axial_einsum_easy_v1")
    torch.manual_seed(9); parent = parent_module.SpatialGatingBlock()
    torch.manual_seed(10); child = child_module.SpatialGatingBlock()
    child.norm.load_state_dict(parent.norm.state_dict())
    child.expand.load_state_dict(parent.expand.state_dict())
    child.gate_norm.load_state_dict(parent.gate_norm.state_dict())
    child.project.load_state_dict(parent.project.state_dict())
    child.place_operator.data.copy_(parent.place_operator.weight)
    child.role_operator.data.copy_(parent.role_operator.weight)
    child.spatial_bias.data.copy_(parent.spatial_bias)
    value = torch.randn(3, 8, 112)
    torch.testing.assert_close(child(value), parent(value), rtol=1e-6, atol=1e-7)


def test_zero_initialized_second_order_alpha_still_has_gradient():
    _, module = load("sgu_axial_second_order_easy_v1")
    block = module.SpatialGatingBlock()
    block(torch.randn(2,8,112)).square().sum().backward()
    assert block.alpha.grad is not None and torch.isfinite(block.alpha.grad) and block.alpha.grad != 0


def test_rank_one_vectors_receive_gradients():
    _, module = load("sgu_axial_rank1_easy_v1")
    block = module.SpatialGatingBlock()
    block(torch.randn(2,8,112)).square().sum().backward()
    assert torch.count_nonzero(block.rank_left.grad) and torch.count_nonzero(block.rank_right.grad)
