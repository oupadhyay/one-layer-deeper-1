import unittest
from pathlib import Path
import torch
import torch.nn.functional as F
from benchmark import ModelSpec, OptimizerSpec, count_model_state_elements
from submission_validation import validate_submission_source
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "submissions/factorized_pair_reducer_e5_remote_v4_d48k1/submission.py"
V2_PATH = ROOT / "submissions/factorized_pair_reducer_e5_remote_v2_k1/submission.py"
module_spec = importlib.util.spec_from_file_location("factorized_pair_reducer_e5_remote_v4_d48k1", PATH)
candidate = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(candidate)


def spec(length=13): return ModelSpec(17, length, 250_000)
def row(n=(3,2,3), x=(1,2,3), t=(1,), padding=0):
    return torch.tensor([[candidate.N, *(d+candidate.DIGIT for d in n), candidate.X,
                          *(d+candidate.DIGIT for d in x), candidate.T,
                          *(d+candidate.DIGIT for d in t)] + [0]*padding])


class FactorizedPairReducerTests(unittest.TestCase):
    def test_state_seed_optimizer_and_source(self):
        torch.manual_seed(74); a=candidate.build_model(spec())
        torch.manual_seed(74); b=candidate.build_model(spec(32))
        self.assertEqual(count_model_state_elements(a), 37018)
        self.assertLess(count_model_state_elements(a), 250000)
        for key, value in a.state_dict().items():
            self.assertEqual(value.numpy().tobytes(), b.state_dict()[key].numpy().tobytes())
        bundle=candidate.build_optimizer(a, OptimizerSpec(1600,"cpu"))
        self.assertEqual(bundle.optimizer.defaults["lr"],1e-4); self.assertEqual(bundle.optimizer.defaults["betas"],(.9,.95))
        self.assertEqual([g["weight_decay"] for g in bundle.optimizer.param_groups],[.1])
        self.assertIsNone(bundle.scheduler)
        source=Path(candidate.__file__).read_text()
        validate_submission_source(Path(candidate.__file__).name,source,256*1024,required_filename=None)
        for text in ("custom_training_loss","competition_loop10","100-entry","i+j","remainder","carry"):
            self.assertNotIn(text,source)
        v2 = V2_PATH.read_text()
        expected = v2.replace("D, PLACES, MICROPHASES, MAX_STEPS = 64, 4, 1, 64", "D, PLACES, MICROPHASES, MAX_STEPS = 48, 4, 1, 64")
        expected = expected.replace("STATE_ELEMENTS = 64_714", "STATE_ELEMENTS = 37_018")
        self.assertEqual(source, expected)

        self.assertEqual(candidate.D, 48)
        self.assertEqual(a.digit_embedding.weight.shape, (10, 48))
        self.assertEqual(a.pair_encoder.mlp[0].weight.shape, (48, 96))
        self.assertEqual(a.pair_encoder.mlp[2].weight.shape, (48, 48))
        self.assertEqual(a.reducer.qkv.weight.shape, (144, 48))

    def test_parsing_context_pairs_counts_and_feedback(self):
        model=candidate.build_model(spec(16))
        ids=torch.cat((row(padding=2),row(n=(9,0,0,1),x=(4,3,2,1),t=(3,))))
        _,nd,xd,widths,steps,pm=model.prepare(ids)
        self.assertEqual(nd.tolist(),[[3,2,3,0],[1,0,0,9]])
        self.assertEqual(xd.tolist(),[[3,2,1,0],[1,2,3,4]])
        self.assertEqual(widths.tolist(),[3,4]); self.assertEqual(steps.tolist(),[1,3])
        probs=F.one_hot(xd,10).float(); state,context,cm=model._contexts(probs,nd,pm)
        self.assertEqual(context.shape[1],20); self.assertEqual(cm.sum(1).tolist(),[12,20])
        model.train(); out,aux=model(ids)
        self.assertNotIn("feedback_states", aux)
        self.assertEqual(aux["active_macrosteps"],4); self.assertEqual(aux["pair_token_count"],57)
        self.assertEqual(aux["reducer_calls"],3); self.assertEqual(aux["scratch_resets"],3)
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
        model=candidate.build_model(spec()).to(torch.bfloat16).eval(); logits,aux=model(row(t=(6,4)))
        self.assertTrue(torch.isfinite(logits).all()); self.assertEqual(aux["parsed_steps"].item(),64)
        self.assertEqual(aux["reducer_calls"],64)
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

    def test_t1_only_gradient_gate_and_remote_contract(self):
        model=candidate.build_model(spec()).train()
        for step in (1,2,3):
            model.zero_grad(set_to_none=True); logits,_=model(row(t=(step,)))
            logits[:,:,candidate.DIGIT:candidate.DIGIT+10].sum().backward()
            for parameter in model.parameters():
                self.assertIsNotNone(parameter.grad)
                if step > 1: self.assertEqual(torch.count_nonzero(parameter.grad).item(),0)
        cuda=candidate.build_optimizer(model,OptimizerSpec(1,"cuda"))
        self.assertTrue(cuda.optimizer.param_groups[0]["capturable"])
        self.assertEqual((candidate.SUBMISSION.batch_size,candidate.SUBMISSION.eval_batch_size,candidate.SUBMISSION.max_steps),(512,512,None))
        source=PATH.read_text()
        self.assertNotIn("einsum",source); self.assertNotIn("debug_execution",source); self.assertNotIn("all_feedback",source)

if __name__=="__main__": unittest.main()
