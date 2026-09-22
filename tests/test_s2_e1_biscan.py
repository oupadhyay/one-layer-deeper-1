import importlib, inspect, json, subprocess, sys, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import torch
from torch import nn
import benchmark.s2_e1_biscan as e1

class E1Tests(unittest.TestCase):
    def test_structure_count_sharing_and_forbidden(self):
        m=e1.initialized_model(); self.assertEqual(sum(p.numel() for p in m.parameters()),63178)
        self.assertIs(m.embedding, m.embedding); self.assertEqual(m.embedding.weight.shape,(10,32)); self.assertEqual(m.readout.weight.shape,(10,64))
        self.assertEqual((m.step.down_cell.input_size,m.step.up_cell.input_size),(64,128))
        self.assertFalse(any(isinstance(x,(nn.Dropout,nn.MultiheadAttention)) for x in m.modules()))
        self.assertEqual(set(dict(m.named_modules())),{"","embedding","step","step.down_cell","step.up_cell","readout"})
    def test_sweep_order_and_signature(self):
        m=e1.initialized_model(); calls=[]
        def hook(tag): return lambda mod,args,out:calls.append((tag,args[0].clone()))
        a=m.step.down_cell.register_forward_hook(hook("d")); b=m.step.up_cell.register_forward_hook(hook("u"))
        state=torch.arange(4.).view(1,4,1).expand(-1,-1,32); context=torch.zeros_like(state); out=m.step(state,context); a.remove(); b.remove()
        self.assertEqual(tuple(inspect.signature(m.step.forward).parameters),("state","context")); self.assertEqual(out.shape,(1,4,64))
        self.assertEqual([int(x[1][0,0]) for x in calls[:4]],[3,2,1,0]); self.assertEqual([int(x[1][0,0]) for x in calls[4:]],[0,1,2,3])
    def test_one_step_shape_canonical_and_purity(self):
        m=e1.initialized_model().eval(); ids=torch.tensor([[0,2,3,4,5,1,6,7,8,9]]); count=0
        def hook(*_):
            nonlocal count; count+=1
        h=m.step.register_forward_hook(hook); before={n:t.clone() for n,t in m.state_dict().items()}; a=m(ids); h.remove()
        self.assertEqual(count,1); self.assertEqual(a.shape,(1,4,10)); self.assertTrue(torch.equal(a,e1.Harness(m).run(ids))); self.assertTrue(torch.equal(a,m(ids))); self.assertTrue(all(torch.equal(v,m.state_dict()[n]) for n,v in before.items()))
        for bad in (torch.zeros(1,9,dtype=torch.long),torch.tensor([[1,2,3,4,5,1,6,7,8,9]]),torch.tensor([[0,12,3,4,5,1,6,7,8,9]])):
            with self.assertRaises(ValueError): m(bad)
    def test_init_hash_reproducible_predeclared(self):
        a=e1.canonical_state_hash(e1.initialized_model()); b=e1.canonical_state_hash(e1.initialized_model()); self.assertEqual(a,b)
        pre=json.loads(Path("artifacts/s2_e1_biscan_seed79_predeclaration.json").read_text())
        code="import torch; from benchmark.s2_e1_biscan import initialized_model,canonical_state_hash; torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True); print(canonical_state_hash(initialized_model()))"
        run_hash=subprocess.check_output([sys.executable,"-c",code],text=True).strip()
        self.assertEqual(run_hash,pre["initial_state_sha256"])
    def test_schedule_projection_access(self):
        self.assertEqual(e1.schedule([256]*134+[12]),(5040,1281212)); access=[]; ds=e1.project_train(Path("data/generated/s2_doublemod_seen_dense_v1"),access); self.assertEqual(len(ds),34316); self.assertEqual(access,["train.jsonl"]); self.assertEqual(set(ds[0]),{"input_ids","labels"})
        with self.assertRaises(ValueError): e1.load_rows(Path("."),"depth_t_2",access)
    def test_weights_only_strict_restore(self):
        m=e1.initialized_model()
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"x.pt"; torch.save({"state_dict":m.state_dict()},p); restored=e1.S2BiScan(); restored.load_state_dict(torch.load(p,weights_only=True)["state_dict"],strict=True)
            self.assertEqual(e1.canonical_state_hash(m),e1.canonical_state_hash(restored))
    def test_import_inert(self):
        targets=[Path("artifacts/s2_e1_biscan_seed79.pt"),Path("metrics_s2_e1_biscan_seed79.jsonl"),Path("artifacts/s2_e1_biscan_seed79_provenance.json")]; before=[p.exists() for p in targets]; importlib.reload(e1); self.assertEqual(before,[p.exists() for p in targets])
    def test_scientific_failure_is_recorded_not_raised(self):
        source=Path(e1.__file__).read_text(encoding="utf-8")
        self.assertIn('e1_pass=all(gates.values())',source)
        self.assertNotIn('raise RuntimeError(f"E1 gates failed:',source)

if __name__ == "__main__": unittest.main()
