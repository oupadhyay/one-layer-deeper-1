import unittest
from pathlib import Path
import torch
import torch.nn.functional as F
from benchmark import ModelSpec, OptimizerSpec, count_model_state_elements
from submission_validation import validate_submission_source
import competition_loop11_factorized_pair_reducer as candidate


def spec(length=13): return ModelSpec(17, length, 250_000)
def row(n=(3,2,3), x=(1,2,3), t=(1,), padding=0):
    return torch.tensor([[candidate.N, *(d+candidate.DIGIT for d in n), candidate.X,
                          *(d+candidate.DIGIT for d in x), candidate.T,
                          *(d+candidate.DIGIT for d in t)] + [0]*padding])


class FactorizedPairReducerTests(unittest.TestCase):
    def test_state_seed_optimizer_and_source(self):
        torch.manual_seed(74); a=candidate.build_model(spec())
        torch.manual_seed(74); b=candidate.build_model(spec(32))
        self.assertEqual(count_model_state_elements(a), 64778)
        self.assertLess(count_model_state_elements(a), 250000)
        for key, value in a.state_dict().items():
            self.assertEqual(value.numpy().tobytes(), b.state_dict()[key].numpy().tobytes())
        bundle=candidate.build_optimizer(a, OptimizerSpec(1600,"cpu"))
        self.assertEqual(bundle.optimizer.defaults["lr"],8e-4); self.assertEqual(bundle.optimizer.defaults["betas"],(.9,.98))
        self.assertEqual([g["weight_decay"] for g in bundle.optimizer.param_groups],[.02,0.])
        self.assertAlmostEqual(bundle.scheduler.get_last_lr()[0],4e-5)
        source=Path(candidate.__file__).read_text()
        validate_submission_source(Path(candidate.__file__).name,source,256*1024,required_filename=None)
        for text in ("custom_training_loss","competition_loop10","100-entry","i+j","remainder","carry"):
            self.assertNotIn(text,source)

    def test_parsing_context_pairs_counts_and_feedback(self):
        model=candidate.build_model(spec(16))
        ids=torch.cat((row(padding=2),row(n=(9,0,0,1),x=(4,3,2,1),t=(3,))))
        _,nd,xd,widths,steps,pm=model.prepare(ids)
        self.assertEqual(nd.tolist(),[[3,2,3,0],[1,0,0,9]])
        self.assertEqual(xd.tolist(),[[3,2,1,0],[1,2,3,4]])
        self.assertEqual(widths.tolist(),[3,4]); self.assertEqual(steps.tolist(),[1,3])
        probs=F.one_hot(xd,10).float(); state,context,cm=model._contexts(probs,nd,pm)
        self.assertEqual(context.shape[1],20); self.assertEqual(cm.sum(1).tolist(),[12,20])
        debug=model.debug_execution(ids); self.assertEqual(debug["active_macrosteps"],4)
        self.assertEqual(debug["pair_token_count"],57); self.assertEqual(debug["reducer_calls"],6); self.assertEqual(debug["scratch_resets"],3)
        model.train(); out,aux=model(ids)
        self.assertTrue(all(torch.equal(x,x.round()) for x in aux["feedback_states"]))
        self.assertTrue(torch.equal(aux["digit_probabilities"][0,3:], F.one_hot(torch.zeros(1,dtype=torch.long),10).float()))
        model.eval(); _,aux=model(ids); self.assertTrue(torch.equal(aux["digit_probabilities"],aux["digit_probabilities"].round()))

    def test_sharing_invariance_gradients_and_limits(self):
        torch.manual_seed(4); model=candidate.build_model(spec(16)).eval()
        narrow=row(padding=2); wide=row(n=(9,0,0,1),x=(4,3,2,1),t=(3,))
        before={k:v.clone() for k,v in model.state_dict().items()}
        with torch.no_grad():
            together=model(torch.cat((narrow,wide)))[0]; separate=torch.cat((model(narrow)[0],model(wide)[0])); again=model(torch.cat((narrow,wide)))[0]
        torch.testing.assert_close(together,separate,atol=2e-5,rtol=2e-5); self.assertTrue(torch.equal(together,again))
        self.assertTrue(all(torch.equal(v,before[k]) for k,v in model.state_dict().items()))
        model=candidate.build_model(spec()).to(torch.bfloat16).train(); logits,aux=model(row(t=(6,4)))
        loss=F.cross_entropy(logits[0,-3:,candidate.DIGIT:],torch.tensor([1,2,3])); loss.backward()
        self.assertTrue(torch.isfinite(loss)); self.assertEqual(aux["parsed_steps"].item(),64)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in model.parameters()))
        exact=row(n=(9,0,0,1),x=(1,2,3,4),t=(3,)); self.assertEqual(candidate.build_model(spec(12))(exact)[0].shape,(1,12,17))
        with self.assertRaises(ValueError): candidate.build_model(spec(11))(exact)

    def test_capacity_contract(self):
        x=torch.tensor([[2,10,9,10,3,8,12,4,8],[2,10,9,10,3,15,16,4,8],[2,10,9,10,3,11,9,4,8]])
        labels=torch.tensor([[9,9,12],[8,13,16],[8,11,16]]); pos=torch.tensor([[6,7,8]]*3); bi=torch.arange(3)[:,None]
        torch.manual_seed(74); model=candidate.build_model(spec()).train(); opt=torch.optim.AdamW(model.parameters(),lr=3e-3)
        for update in range(1,101):
            opt.zero_grad(set_to_none=True); selected=model(x)[0][bi,pos]
            F.cross_entropy(selected.flatten(0,1),labels.flatten()).backward(); opt.step()
            if selected.argmax(-1).eq(labels).all(): break
        self.assertTrue(selected.argmax(-1).eq(labels).all(),update)
        model.eval()
        with torch.no_grad(): selected=model(x)[0][bi,pos]
        self.assertTrue(selected.argmax(-1).eq(labels).all()); self.assertLessEqual(update,100)

if __name__=="__main__": unittest.main()
