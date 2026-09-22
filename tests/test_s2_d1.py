import importlib, inspect, tempfile, unittest
from pathlib import Path
import torch

import benchmark.s2_d1 as d1
from benchmark.s2_c1 import S2Transformer

class DenseRunnerTests(unittest.TestCase):
    def test_exact_import_init_and_parameter_count(self):
        self.assertIs(d1.S2Transformer,S2Transformer); d1.verify_architecture_source()
        model=d1.initialized_model(); self.assertEqual(sum(p.numel() for p in model.parameters()),796426)
        torch.manual_seed(78); sparse=S2Transformer()
        self.assertTrue(all(torch.equal(model.state_dict()[n],sparse.state_dict()[n]) for n in model.state_dict()))
    def test_projection_access_and_api(self):
        access=[]; ds=d1.project_train(Path("data/generated/s2_doublemod_seen_dense_v1"),access)
        self.assertEqual(len(ds),34316); self.assertEqual(access,["train.jsonl"])
        with self.assertRaises(ValueError): d1.load_split(Path("."),"depth_t_2",access)
        self.assertEqual(tuple(inspect.signature(S2Transformer().step.forward).parameters),("state","context"))
    def test_exact_schedule(self):
        self.assertEqual(d1.schedule([256]*134+[12]),(5040,1_281_212))
        with self.assertRaises(RuntimeError): d1.schedule([256]*135)
    def test_gate_names_keep_test_and_train_thresholds_separate(self):
        source=Path(d1.__file__).read_text(encoding="utf-8")
        self.assertIn('"test_threshold_abs1_16_exact"',source)
        self.assertIn('"train_threshold_abs3_16_exact"',source)
        self.assertNotIn('"threshold_abs1_and_abs3_exact"',source)
    def test_purity_and_strict_weights_only_checkpoint_semantics(self):
        model=d1.initialized_model().eval(); ids=torch.tensor([[0,3,2,2,2,1,11,2,2,10]])
        before={n:t.clone() for n,t in model.state_dict().items()}; self.assertTrue(torch.equal(model(ids),d1.Harness(model).run(ids)))
        self.assertTrue(torch.equal(model(ids),model(ids))); self.assertTrue(all(torch.equal(t,model.state_dict()[n]) for n,t in before.items()))
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"x.pt"; torch.save({"state_dict":model.state_dict()},p)
            restored=S2Transformer(); restored.load_state_dict(torch.load(p,weights_only=True)["state_dict"],strict=True)
            self.assertTrue(all(torch.equal(t,restored.state_dict()[n]) for n,t in model.state_dict().items()))
    def test_import_has_no_training_side_effect(self):
        checkpoint=Path("artifacts/s2_d1_dense_density_control_seed78.pt")
        before=(checkpoint.exists(),checkpoint.stat().st_mtime_ns if checkpoint.exists() else None)
        importlib.reload(d1)
        after=(checkpoint.exists(),checkpoint.stat().st_mtime_ns if checkpoint.exists() else None)
        self.assertEqual(after,before)

if __name__=="__main__": unittest.main()
