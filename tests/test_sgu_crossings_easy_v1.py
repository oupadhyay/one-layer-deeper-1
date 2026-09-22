"""Round-2 RMS by axial crossing package regression tests."""
import importlib.util
from pathlib import Path
import unittest

import torch
from benchmark import ModelSpec, OptimizerSpec, count_model_state_elements
from submission_validation import validate_submission_source

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "c11": (79_458, "functional", "place_role"),
    "c12": (79_447, "functional", "second_order"),
    "c21": (79_682, "gate_ln", "place_role"),
    "c22": (79_671, "gate_ln", "second_order"),
    "c33": (79_558, "input_ln", "einsum"),
    "c34": (79_574, "input_ln", "role_place"),
    "c43": (79_446, "eps", "einsum"),
    "c44": (79_462, "eps", "role_place"),
}
AXIAL_ROOT = {"place_role": "sgu_axial_place_role_easy_v1", "second_order": "sgu_axial_second_order_easy_v1",
              "einsum": "sgu_axial_einsum_easy_v1", "role_place": "sgu_axial_role_place_easy_v1"}

def load(name):
    path = ROOT / "submissions" / name / "submission.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return path, module

def check_exact_contract_state_norms_topology_and_initialization(code, case):
    count, rms, axial = case; path, module = load(f"sgu_cross_{code}_easy_v1")
    torch.manual_seed(112); model = module.build_model(ModelSpec(17, 24, 250_000)); block = model.transition.block
    assert module.STATE_ELEMENTS == count == count_model_state_elements(model) < 250_000
    assert (module.SUBMISSION.batch_size, module.SUBMISSION.eval_batch_size, module.SUBMISSION.max_steps) == (256,512,None)
    bundle = module.build_optimizer(model, OptimizerSpec(1, "cpu"))
    assert bundle.optimizer.defaults["lr"] == 6e-4 and bundle.optimizer.defaults["betas"] == (.9,.95)
    assert bundle.scheduler.get_last_lr()[0] == 6e-4 / 32
    norms = [block.norm, block.gate_norm, model.transition.readout_norm]
    expected_ln = {"gate_ln": [False,True,False], "input_ln": [True,False,False]}.get(rms, [False]*3)
    assert [isinstance(n, torch.nn.LayerNorm) for n in norms] == expected_ln
    for n, is_ln in zip(norms, expected_ln):
        if not is_ln: assert isinstance(n, module.RMSNorm) and n.eps == (1e-5 if rms == "eps" else 1e-6)
    source = path.read_text()
    assert ("F.rms_norm" in source) == (rms == "functional")
    assert validate_submission_source(path.name, source, 256_000) == "submission.py"
    assert "self.spatial = nn.Linear" not in source and "cross_entropy" not in source.lower()
    assert tuple(block.spatial_bias.shape) == (4,2); torch.testing.assert_close(block.spatial_bias, torch.ones(4,2))
    if axial == "place_role": assert block.place_operator.shape == (4,4) and block.role_operator.shape == (4,2,2)
    elif axial == "second_order":
        assert block.place_operator.weight.shape == (4,4) and block.role_operator.weight.shape == (2,2)
        assert block.alpha.shape == () and block.alpha.item() == 0
    elif axial == "einsum": assert block.place_operator.shape == (4,4) and block.role_operator.shape == (2,2)
    else: assert block.place_operator.shape == (2,4,4) and block.role_operator.shape == (2,2)
    for name, p in block.named_parameters():
        if "operator" in name: assert p.abs().max() <= 1.25e-4

def check_parser_alignment_t64_parity_determinism_permutation_and_gradients(code):
    _, module = load(f"sgu_cross_{code}_easy_v1"); torch.manual_seed(7)
    model = module.build_model(ModelSpec(17,24,250_000))
    p = torch.tensor([[2,8,9,10,11,3,12,7,4,13,11,5,0,0]])
    _, place, steps = model.parse(p,p.ne(0)); assert (steps.item(),place[0,1:5].tolist()) == (64,[3,2,1,0])
    model.eval(); out, info = model(p); assert out.shape == (1,14,17) and info["macrosteps"] == 64
    assert (out[0,8:10,7:17] > -10_000).all()
    ids=torch.tensor([[2,8,9,10,3,10,4,10,5,0],[2,11,3,12,13,4,10,5,0,0]])
    model.train(); a,info=model(ids); b,_=model(ids); q,_=model(ids.flip(0))
    assert info["macrosteps"] == 3; torch.testing.assert_close(a,b); torch.testing.assert_close(a,q.flip(0))
    model.eval(); e,info=model(ids); assert info["macrosteps"] == 3; torch.testing.assert_close(a,e)
    model.train(); a[:,:,7:17].sum().backward()
    for name,param in model.named_parameters(): assert param.grad is not None and torch.isfinite(param.grad).all(), name

def check_axial_operator_numerical_parity_with_frozen_roots():
    for code, (_, _, axial) in CASES.items():
        _, cross = load(f"sgu_cross_{code}_easy_v1"); _, root = load(AXIAL_ROOT[axial])
        cb, rb = cross.SpatialGatingBlock(), root.SpatialGatingBlock()
        # The transplanted operator receives the same already-normalized v; compare its exact equations.
        for name in ("place_operator", "role_operator", "spatial_bias", "alpha"):
            if hasattr(cb,name): getattr(cb,name).load_state_dict(getattr(rb,name).state_dict()) if isinstance(getattr(cb,name),torch.nn.Module) else getattr(cb,name).data.copy_(getattr(rb,name).data)
        v=torch.randn(2,4,2,224)
        def op(b):
            if axial=="place_role": return torch.einsum("op,bprh->borh",b.place_operator,v)+torch.einsum("por,bprh->bpoh",b.role_operator,v)+b.spatial_bias[None,:,:,None]
            if axial=="second_order":
                pg=b.place_operator(v.permute(0,2,3,1)).permute(0,3,1,2); rg=b.role_operator(v.permute(0,1,3,2)).permute(0,1,3,2)
                return pg+rg+b.alpha*b.place_operator(rg.permute(0,2,3,1)).permute(0,3,1,2)+b.spatial_bias[None,:,:,None]
            if axial=="einsum": return torch.einsum("op,bprh->borh",b.place_operator,v)+torch.einsum("or,bprh->bpoh",b.role_operator,v)+b.spatial_bias[None,:,:,None]
            return torch.einsum("rop,bprh->borh",b.place_operator,v)+torch.einsum("or,bprh->bpoh",b.role_operator,v)+b.spatial_bias[None,:,:,None]
        torch.testing.assert_close(op(cb),op(rb))


class SGUCrossingTests(unittest.TestCase):
    def test_exact_contract_state_norms_topology_and_initialization(self):
        for code, case in CASES.items():
            with self.subTest(code=code):
                check_exact_contract_state_norms_topology_and_initialization(code, case)

    def test_parser_alignment_t64_parity_determinism_permutation_and_gradients(self):
        for code in CASES:
            with self.subTest(code=code):
                check_parser_alignment_t64_parity_determinism_permutation_and_gradients(code)

    def test_axial_operator_numerical_parity_with_frozen_roots(self):
        check_axial_operator_numerical_parity_with_frozen_roots()


if __name__ == "__main__":
    unittest.main()
