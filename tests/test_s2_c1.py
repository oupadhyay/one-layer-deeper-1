import inspect, unittest
import torch
from benchmark.s2_c1 import Harness,S2Transformer

class S2C1Tests(unittest.TestCase):
    def setUp(self): torch.manual_seed(78); self.m=S2Transformer()
    def test_exact_structure(self):
        self.assertEqual(self.m.embedding.weight.shape,(10,128)); self.assertEqual(self.m.role_embedding.weight.shape,(2,128)); self.assertEqual(self.m.place_embedding.weight.shape,(4,128))
        self.assertEqual(len(self.m.step.blocks),4); self.assertEqual(tuple(inspect.signature(self.m.step.forward).parameters),("state","context"))
        self.assertEqual(len({id(b.attn.in_proj_weight) for b in self.m.step.blocks}),4); self.assertIs(self.m.readout,self.m.readout)
    def test_encoders_layout_and_one_step_equality(self):
        ids=torch.tensor([[0,3,2,2,2,1,11,2,2,10]])
        x,n=self.m.canonical(ids); self.assertEqual(self.m.encode_state(x).shape,(1,4,128)); self.assertFalse(torch.equal(self.m.encode_state(x),self.m.encode_context(x)))
        self.assertTrue(torch.equal(self.m(ids),Harness(self.m).run(ids)))
        with self.assertRaises(ValueError): self.m(torch.zeros(1,10,dtype=torch.long))
    def test_cross_slot_attention_path(self):
        block=self.m.step.blocks[0]; captured=[]
        hook=block.attn.register_forward_hook(lambda m,a,o: captured.append(o[0].detach()))
        s=torch.randn(1,4,128); n=torch.randn(1,4,128); self.m.step(s,n); first=captured[-1][:,0].clone(); s[:,3]+=3; self.m.step(s,n)
        self.assertFalse(torch.allclose(first,captured[-1][:,0])); first=captured[-1][:,0].clone(); n[:,3]+=3; self.m.step(s,n); self.assertFalse(torch.allclose(first,captured[-1][:,0])); hook.remove()
    def test_purity(self):
        ids=torch.tensor([[0,3,2,2,2,1,11,2,2,10]]); before={n:v.clone() for n,v in self.m.state_dict().items()}; a=self.m(ids); b=self.m(ids)
        self.assertTrue(torch.equal(a,b)); self.assertTrue(all(torch.equal(v,self.m.state_dict()[n]) for n,v in before.items()))
if __name__=="__main__": unittest.main()
