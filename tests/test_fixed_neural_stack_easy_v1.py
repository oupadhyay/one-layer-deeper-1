import importlib.util
import unittest
from pathlib import Path
import torch
import torch.nn.functional as F
from benchmark import ModelSpec, OptimizerSpec, count_model_state_elements

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "submissions/fixed_neural_stack_easy_v1/submission.py"
loader = importlib.util.spec_from_file_location("fnst8", PATH)
m = importlib.util.module_from_spec(loader); loader.loader.exec_module(m)

def spec(): return ModelSpec(17, 20, 500_000_000)
def sample(t=1):
    t_tokens = [7 + int(digit) for digit in str(t)]
    tokens = [2, 8, 9, 10, 11, 3, 12, 13, 14, 15, 4, *t_tokens, 5]
    ids = torch.tensor([tokens + [0] * (20 - len(tokens))])
    return ids, torch.tensor([[1] * len(tokens) + [0] * (20 - len(tokens))])

class FNST8Tests(unittest.TestCase):
    def test_state_optimizer(self):
        model=m.build_model(spec()); self.assertEqual(count_model_state_elements(model),m.STATE_ELEMENTS)
        b=m.build_optimizer(model,OptimizerSpec(1,"cpu")); self.assertEqual(b.optimizer.defaults["lr"],6e-4)
        self.assertEqual((m.SUBMISSION.batch_size,m.SUBMISSION.eval_batch_size),(256,512))

    def test_stack(self):
        v=torch.zeros(1,8,64); s=torch.zeros(1,8); a=torch.tensor([[1.,0.,0.]])
        v,s,r=m.continuous_stack_step(v,s,torch.ones(1,64),a)
        v,s,r=m.continuous_stack_step(v,s,2*torch.ones(1,64),a); self.assertEqual(r[0,0],2)
        v,s,r=m.continuous_stack_step(v,s,torch.zeros(1,64),torch.tensor([[0.,1.,0.]])); self.assertEqual(r[0,0],1)
        before=(v.clone(),s.clone()); v,s,_=m.continuous_stack_step(v,s,torch.randn(1,64),torch.tensor([[0.,0.,1.]]))
        torch.testing.assert_close(v[:,:-1],before[0][:,1:]); self.assertEqual(s[:,-1],0)
        x=torch.tensor([[.4,.3,.3]],requires_grad=True)
        _,_,r=m.continuous_stack_step(torch.zeros_like(v),torch.zeros_like(s),torch.ones(1,64),x.softmax(-1)); r.sum().backward()
        self.assertGreater(x.grad.abs().sum(),0)

    def test_ticks_alignment_gradient_gate_and_determinism(self):
        model=m.build_model(spec()); ids,mask=sample(1)
        calls=[]; hook=model.transition.controller.register_forward_hook(lambda *args:calls.append(1))
        model.train(); out,info=model(ids,mask); hook.remove()
        self.assertEqual(len(calls),8); self.assertEqual((info["widths"].item(),info["steps"].item()),(4,1))
        F.cross_entropy(out[:,12],torch.tensor([8])).backward()
        for n,p in model.named_parameters():
            self.assertIsNotNone(p.grad,n); self.assertTrue(torch.isfinite(p.grad).all(),n); self.assertGreater(p.grad.abs().sum(),0,n)
        train=out.detach(); model.eval()
        with torch.no_grad(): eval1,_=model(ids,mask)
        torch.testing.assert_close(train,eval1)
        model.zero_grad(); model.train(); ids2,mask2=sample(2); out,_=model(ids2,mask2)
        out[:,12].sum().backward()
        for p in model.parameters():
            if p.grad is not None: self.assertEqual(p.grad.abs().sum(),0)
        model.eval(); ids64,mask64=sample(64)
        state={k:v.clone() for k,v in model.state_dict().items()}
        with torch.no_grad(): a,info64=model(ids64,mask64); b,_=model(ids64,mask64)
        self.assertEqual((info64["steps"].item(), info64["macrosteps"]), (64, 64))
        torch.testing.assert_close(a,b)
        for k,v in model.state_dict().items(): torch.testing.assert_close(v,state[k])

        narrow,narrow_mask=sample(1)
        wide,wide_mask=sample(3)
        with torch.no_grad():
            alone,_=model(narrow,narrow_mask)
            mixed,_=model(torch.cat((narrow,wide)),torch.cat((narrow_mask,wide_mask)))
        torch.testing.assert_close(alone[0],mixed[0],atol=2e-5,rtol=2e-5)

    def test_reject_width(self):
        model=m.build_model(spec()); ids=torch.tensor([[2,7,8,9,10,11,3,7,4,8,5]+[0]*9]); mask=ids.ne(0)
        with self.assertRaises(ValueError): model(ids,mask)

if __name__ == "__main__": unittest.main()
