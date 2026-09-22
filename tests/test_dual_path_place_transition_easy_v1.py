import importlib.util, py_compile, unittest
from pathlib import Path
import torch
from benchmark import ModelSpec, OptimizerSpec, count_model_state_elements
from submission_validation import validate_submission_source
ROOT=Path(__file__).resolve().parents[1]; PATH=ROOT/'submissions/dual_path_place_transition_easy_v1/submission.py'
spec=importlib.util.spec_from_file_location('dual',PATH); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
def row(n,x,t,pad=0): return [m.N,*[m.DIGIT+int(v) for v in str(n)],m.X,*[m.DIGIT+int(v) for v in str(x)],m.T,*[m.DIGIT+int(v) for v in str(t)],m.ANS,m.ANS,*([0]*pad)]
class Tests(unittest.TestCase):
 def setUp(self): self.s=ModelSpec(17,24,100000); torch.manual_seed(2); self.model=m.build_model(self.s)
 def test_source_state_config_optimizer(self):
  source=PATH.read_text(); validate_submission_source(PATH.name,source,256*1024); py_compile.compile(str(PATH),doraise=True); self.assertNotIn('einsum',source); self.assertEqual(count_model_state_elements(self.model),84276); self.assertEqual((self.model.config.vocab_size,self.model.config.max_seq_len),(17,24)); b=m.build_optimizer(self.model,OptimizerSpec(1,'cpu')); self.assertEqual([g['lr'] for g in b.optimizer.param_groups],[1e-3,1e-4]); ps=[id(p) for g in b.optimizer.param_groups for p in g['params']]; self.assertEqual(len(ps),len(set(ps))); self.assertEqual(set(ps),{id(p) for p in self.model.parameters()}); self.assertIsNone(b.scheduler)
 def test_mixed_alignment_purity_and_parity(self):
  a=row(7,4,1,4); b=row(123,45,2,1); ids=torch.tensor([a,b]); before={k:v.clone() for k,v in self.model.state_dict().items()}; self.model.eval(); x,aux=self.model(ids); y,_=self.model(ids); self.assertTrue(torch.equal(x,y)); self.assertEqual(aux['widths'].tolist(),[1,3]); self.assertEqual(aux['parsed_steps'].tolist(),[1,2]); self.assertTrue(torch.isfinite(x).all()); self.assertTrue(all(torch.equal(v,before[k]) for k,v in self.model.state_dict().items()))
  with torch.no_grad(): alone,_=self.model(ids[:1])
  torch.testing.assert_close(x[0],alone[0],atol=2e-5,rtol=0)
 def test_calls_loss_gradients_guards_js(self):
  ids=torch.tensor([row(12,34,2,2)]); logits,aux=self.model(ids); selected=logits[:,-2:].reshape(-1,17); labels=torch.tensor([7,8]); loss=m.token_training_loss(selected,labels,aux); loss.backward(); self.assertEqual((aux['axial_calls'],aux['pair_calls']),(2,2)); self.assertTrue(torch.isfinite(aux['branch_js'])); self.assertGreater(sum((p.grad.abs().sum().item() if p.grad is not None else 0) for p in self.model.axial.parameters()),0); self.assertGreater(sum((p.grad.abs().sum().item() if p.grad is not None else 0) for p in self.model.pair.parameters()),0)
  for badx,bady in ((selected[None],labels),(selected,labels[None]),(selected,labels[:1])):
   with self.assertRaises(ValueError): m.token_training_loss(badx,bady,aux)
 def test_t64_tiny(self):
  self.model.eval()
  with torch.no_grad(): out,aux=self.model(torch.tensor([row(3,6,64)]))
  self.assertEqual(aux['macrosteps'],64); self.assertEqual(aux['axial_calls'],64); self.assertTrue(torch.isfinite(out).all())
if __name__=='__main__': unittest.main()
