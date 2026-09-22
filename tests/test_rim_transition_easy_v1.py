import importlib.util
import unittest
from pathlib import Path
import torch
from benchmark import ModelSpec, count_model_state_elements

PATH = Path(__file__).resolve().parents[1] / "submissions/rim_transition_easy_v1/submission.py"
spec = importlib.util.spec_from_file_location("rim_easy", PATH)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

class RIMTests(unittest.TestCase):
    def setUp(self): self.model = m.build_model(ModelSpec(17, 20, 150_000))
    def sample(self, t=3):
        ds = [int(c)+7 for c in str(t)]
        return torch.tensor([[2,8,9,10,11,3,12,13,4,*ds,5] + [0]*(6-len(ds))])
    def test_state_tying_count_and_router(self):
        self.assertEqual(count_model_state_elements(self.model), m.STATE_ELEMENTS)
        self.assertLess(m.STATE_ELEMENTS, 150000)
        self.assertEqual(len([x for x in self.model.modules() if isinstance(x, torch.nn.GRUCell)]), 1)
        ids=self.sample(); self.model.train(); out,d=self.model(ids,ids.ne(0))
        self.assertEqual(d["macrosteps"],3); masks=d["transition"]["router_masks"]
        self.assertEqual(masks.shape[1],8); self.assertTrue((masks.ne(0).sum(-1)==2).all())
        self.assertTrue(self.model.transition.router.weight.requires_grad)
        out[:,7:11,7:17].sum().backward()
        for n,p in self.model.named_parameters(): self.assertIsNotNone(p.grad,n)
    def test_parser_alignment_and_t64(self):
        ids=self.sample(64); self.model.eval()
        with torch.no_grad(): out,d=self.model(ids,ids.ne(0))
        self.assertEqual(d["steps"].item(),64); self.assertEqual(d["macrosteps"],64)
        self.assertTrue(torch.isfinite(out[:,7:11,7:17]).all())
    def test_purity_permutation_determinism(self):
        self.model.eval(); a=self.sample(1); b=self.sample(2); batch=torch.cat((a,b))
        with torch.no_grad():
            x,_=self.model(batch,batch.ne(0)); y,_=self.model(batch.flip(0),batch.flip(0).ne(0)); z,_=self.model(batch,batch.ne(0))
        torch.testing.assert_close(x,y.flip(0)); torch.testing.assert_close(x,z)
    def test_rejections_and_source(self):
        bad=self.sample(65)
        with self.assertRaises(ValueError): self.model(bad,bad.ne(0))
        source=PATH.read_text().lower()
        for forbidden in ("lookup table", "cross-batch", "synthetic data"):
            self.assertNotIn(forbidden,source)

if __name__ == "__main__": unittest.main()
