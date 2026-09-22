import importlib.util
from pathlib import Path
import unittest
import torch
from benchmark import ModelSpec,count_model_state_elements
from submission_validation import validate_submission_source
ROOT=Path(__file__).resolve().parents[1];PATH=ROOT/'submissions/agreement_capsule_easy_v1/submission.py';spec=importlib.util.spec_from_file_location('caps',PATH);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def row(t=(3,),length=20):
 v=[m.N,8,9,10,11,m.X,12,13,m.T,*(x+m.DIGIT for x in t)];return torch.tensor([v+[0]*(length-len(v))])
class CapsuleTests(unittest.TestCase):
 def test_contract_routing_gradients_alignment_purity_and_t64(self):
  self.assertEqual(validate_submission_source(PATH.name,PATH.read_text(),256000),'submission.py');model=m.build_model(ModelSpec(17,20,100000));self.assertEqual(count_model_state_elements(model),m.STATE)
  model.train();out,info=model(row());self.assertEqual((info['steps'].item(),info['macrosteps']),(3,3));a=info['routing']['assignments'];self.assertEqual(a.shape,(1,8,4));torch.testing.assert_close(a.sum(2),torch.tensor([[1.,1.,1.,1.,1.,1.,0.,0.]]))
  out[:,:,7:].sum().backward();self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()));self.assertEqual(m.ROUNDS,3)
  model.eval();before={k:v.clone() for k,v in model.state_dict().items()}
  with torch.no_grad():x,d=model(row((6,4)));y,_=model(row((6,4)))
  self.assertEqual((d['steps'].item(),d['macrosteps']),(64,64));torch.testing.assert_close(x,y,atol=0,rtol=0);self.assertTrue(all(torch.equal(before[k],v) for k,v in model.state_dict().items()));self.assertEqual(m.SUBMISSION.batch_size,256);self.assertIsNone(m.SUBMISSION.max_steps)
if __name__=='__main__':unittest.main()
