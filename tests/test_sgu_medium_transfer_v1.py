import importlib.util, unittest
from pathlib import Path
import torch
from benchmark import ModelSpec, OptimizerSpec, count_model_state_elements
from submission_validation import validate_submission_source
ROOT=Path(__file__).resolve().parents[1]
NAMES=("sgu_medium_rms_dense_v1","sgu_medium_axial_ln_v1","sgu_medium_c21_v1","sgu_medium_c21_h280_v1")
STATES=(79864,80033,79809,98849)
def load(name):
 p=ROOT/"submissions"/name/"submission.py"; s=importlib.util.spec_from_file_location(name,p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return p,m
class MediumTransferTests(unittest.TestCase):
 def test_states_topologies_norms_and_contracts(self):
  mods=[]
  for name,state in zip(NAMES,STATES):
   p,m=load(name); model=m.build_model(ModelSpec(17,32,250000)); mods.append((m,model)); b=model.transition.block
   self.assertEqual(count_model_state_elements(model),state); self.assertLess(state,250000); self.assertEqual((m.WIDTH,m.D_MODEL,m.MAX_T),(5,112,64)); self.assertEqual(tuple(model.transition.place_embedding.weight.shape),(5,112)); self.assertIsInstance(b.gate_norm,torch.nn.LayerNorm)
   self.assertEqual((m.SUBMISSION.batch_size,m.SUBMISSION.eval_batch_size,m.SUBMISSION.max_steps),(256,512,None)); bundle=m.build_optimizer(model,OptimizerSpec(1,"cpu")); self.assertEqual(bundle.optimizer.defaults["lr"],6e-4); self.assertEqual(bundle.scheduler.get_last_lr()[0],6e-4/32)
   source=p.read_text(); self.assertEqual(validate_submission_source(p.name,source,256000),"submission.py")
   for x in ("from submissions","cross_entropy","argmax","remainder","carry","lookup","cache","conv1d","conv2d"): self.assertNotIn(x,source.lower())
  m1,x1=mods[0]; self.assertIsInstance(x1.transition.block.norm,m1.RMSNorm); self.assertIsInstance(x1.transition.readout_norm,m1.RMSNorm); self.assertEqual(x1.transition.block.norm.eps,1e-6); self.assertEqual(tuple(x1.transition.block.spatial.weight.shape),(10,10))
  m2,x2=mods[1]; self.assertIsInstance(x2.transition.block.norm,torch.nn.LayerNorm); self.assertIsInstance(x2.transition.readout_norm,torch.nn.LayerNorm)
  for m,x in mods[1:]:
   b=x.transition.block; self.assertEqual(tuple(b.place_operator.shape),(5,5)); self.assertEqual(tuple(b.role_operator.shape),(5,2,2)); self.assertEqual(tuple(b.spatial_bias.shape),(5,2)); torch.testing.assert_close(b.spatial_bias,torch.ones(5,2))
  m3,x3=mods[2]; self.assertIsInstance(x3.transition.block.norm,m3.RMSNorm); self.assertIsInstance(x3.transition.readout_norm,m3.RMSNorm); self.assertEqual(m3.HIDDEN,224)
  m4,x4=mods[3]; self.assertEqual(m4.HIDDEN,280); self.assertEqual(tuple(x3.transition.block.expand.weight.shape),(448,112)); self.assertEqual(tuple(x4.transition.block.expand.weight.shape),(560,112)); self.assertEqual(set(x3.state_dict()),set(x4.state_dict()))
 def test_parser_execution_parity_permutation_gradients_and_rejection(self):
  for name in NAMES:
   _,m=load(name); torch.manual_seed(17); model=m.build_model(ModelSpec(17,32,250000))
   for w in range(1,6):
    row=[2]+list(range(8,8+w))+[3]+list(range(8,8+w))+[4,9,5]; out,info=model(torch.tensor([row])); self.assertEqual((info["widths"].item(),info["x_widths"].item()),(w,w)); self.assertEqual(out.shape,(1,len(row),17))
   ids=torch.tensor([[2,8,3,9,4,9,5,0],[2,9,3,10,4,9,5,0]]); model.train(); a,i=model(ids); self.assertEqual(i["macrosteps"],2); b,_=model(ids.flip(0)); torch.testing.assert_close(a,b.flip(0)); model.eval(); c,j=model(ids); self.assertEqual(j["macrosteps"],2); torch.testing.assert_close(a,c); d,_=model(ids); torch.testing.assert_close(c,d)
   _,info=model(torch.tensor([[2,8,3,9,4,13,11,5]])); self.assertEqual(info["macrosteps"],64)
   with self.assertRaisesRegex(ValueError,"one and five"): model(torch.tensor([[2,8,9,10,11,12,13,3,8,4,9,5]]))
   with self.assertRaisesRegex(ValueError,"T exceeds 64"): model(torch.tensor([[2,8,3,9,4,13,12,5]]))
   model.zero_grad(); logits,_=model(ids); logits[:,:,7:17].sum().backward()
   for key,p in model.named_parameters(): self.assertIsNotNone(p.grad,key); self.assertTrue(torch.isfinite(p.grad).all(),key)
if __name__=="__main__": unittest.main()
