import importlib.util
from pathlib import Path
import unittest
import torch
from benchmark import ModelSpec,count_model_state_elements
from submission_validation import validate_submission_source

ROOT=Path(__file__).resolve().parents[1]
PATH=ROOT/'submissions/tpr_bilinear_easy_v2_two_ticks/submission.py'
spec=importlib.util.spec_from_file_location('tpr_v2',PATH);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def row(t=(3,),length=20):
 v=[m.N,8,9,10,11,m.X,12,13,m.T,*(x+m.DIGIT for x in t)];return torch.tensor([v+[0]*(length-len(v))])

class TPRV2Tests(unittest.TestCase):
 def test_contract_binding_gradients_purity_and_max_t(self):
  self.assertEqual(validate_submission_source(PATH.name,PATH.read_text(),256_000),'submission.py')
  model=m.build_model(ModelSpec(17,20,250_000));self.assertEqual(count_model_state_elements(model),14186)
  z=torch.randn(2,32,16,requires_grad=True);expected=torch.stack([model.cell.left[k,0]@z@model.cell.right[k,0] for k in range(4)],1);torch.testing.assert_close(model.cell.kron(z,0),expected)
  model.train();out,info=model(row());self.assertEqual((info['steps'].item(),info['macrosteps'],info['microticks']),(3,3,6));out[:,:,7:].sum().backward();self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
  model.eval()
  with torch.no_grad():a,detail=model(row((6,4)));b,_=model(row((6,4)))
  self.assertEqual((detail['macrosteps'],detail['microticks']),(64,128));torch.testing.assert_close(a,b,atol=0,rtol=0)
  self.assertEqual(m.SUBMISSION.batch_size,256);self.assertIsNone(m.SUBMISSION.max_steps)

if __name__=='__main__':unittest.main()
