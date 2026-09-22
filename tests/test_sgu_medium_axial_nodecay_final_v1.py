import difflib, importlib.util, unittest
from pathlib import Path
import torch
from benchmark import ModelSpec, OptimizerSpec, count_model_state_elements
from submission_validation import validate_submission_source

ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT/"submissions/sgu_medium_axial_ln_v1/submission.py"
CANDIDATE=ROOT/"submissions/sgu_medium_axial_nodecay_final_v1/submission.py"
def load(path,name):
 spec=importlib.util.spec_from_file_location(name,path); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module
p=load(PARENT,"axial_parent"); m=load(CANDIDATE,"axial_nodecay")

class AxialNoDecayFinalTests(unittest.TestCase):
 def setUp(self):
  torch.manual_seed(80033); self.parent=p.build_model(ModelSpec(17,64,250000))
  torch.manual_seed(80033); self.model=m.build_model(ModelSpec(17,64,250000))
 def test_exact_initial_parity_topology_and_contract(self):
  self.assertEqual(count_model_state_elements(self.model),80033)
  self.assertEqual(self.model.state_dict().keys(),self.parent.state_dict().keys())
  for key,value in self.model.state_dict().items(): torch.testing.assert_close(value,self.parent.state_dict()[key])
  b=self.model.transition.block
  self.assertTrue(all(isinstance(x,torch.nn.LayerNorm) for x in (b.norm,b.gate_norm,self.model.transition.readout_norm)))
  self.assertEqual((tuple(b.place_operator.shape),tuple(b.role_operator.shape),tuple(b.spatial_bias.shape)),((5,5),(5,2,2),(5,2)))
  self.assertEqual((m.SUBMISSION.batch_size,m.SUBMISSION.eval_batch_size,m.SUBMISSION.max_steps),(256,512,None))
 def test_sole_source_and_optimizer_delta(self):
  a=PARENT.read_text().splitlines(); b=CANDIDATE.read_text().splitlines()
  changes=[x for x in difflib.ndiff(a,b) if x[:2] in ("- ","+ ")]
  self.assertEqual(len(changes),2); self.assertIn('"weight_decay":.01',changes[0]); self.assertIn('"weight_decay":0.0',changes[1])
  bundle=m.build_optimizer(self.model,OptimizerSpec(1,"cpu")); opt=bundle.optimizer
  self.assertTrue(all(g["weight_decay"]==0 for g in opt.param_groups)); self.assertEqual(opt.defaults["lr"],6e-4)
  self.assertEqual(opt.defaults["betas"],(.9,.95)); self.assertEqual(opt.defaults["eps"],1e-8); self.assertEqual(bundle.scheduler.get_last_lr()[0],6e-4/32)
 def test_parser_alignment_width_t64_fixed_t2_and_parity(self):
  for width in range(1,6):
   row=[2]+list(range(8,8+width))+[3]+list(range(8,8+width))+[4,9,5]
   a,ia=self.model(torch.tensor([row])); b,ib=self.parent(torch.tensor([row])); torch.testing.assert_close(a,b); self.assertEqual((ia["widths"].item(),ia["x_widths"].item()),(width,width))
  ids=torch.tensor([[2,8,3,9,4,9,5,0],[2,9,3,10,4,9,5,0]])
  out,info=self.model(ids); self.assertEqual(info["macrosteps"],2); self.assertEqual((out[:,:,7:17]>m.NEG).sum().item(),20)
  _,info=self.model(torch.tensor([[2,8,3,9,4,13,11,5]])); self.assertEqual(info["macrosteps"],64)
 def test_determinism_permutation_gradients_validation_and_exclusions(self):
  ids=torch.tensor([[2,8,3,9,4,9,5,0],[2,9,3,10,4,11,5,0]])
  a,_=self.model(ids); b,_=self.model(ids); torch.testing.assert_close(a,b)
  c,_=self.model(ids.flip(0)); torch.testing.assert_close(a,c.flip(0))
  a[:,:,7:17].sum().backward()
  for name,param in self.model.named_parameters(): self.assertIsNotNone(param.grad,name); self.assertTrue(torch.isfinite(param.grad).all(),name)
  source=CANDIDATE.read_text(); self.assertEqual(validate_submission_source(CANDIDATE.name,source,256000),"submission.py")
  for forbidden in ("from submissions","cross_entropy","argmax","remainder","carry","lookup","cache","conv1d","conv2d"): self.assertNotIn(forbidden,source.lower())

if __name__=="__main__": unittest.main()
