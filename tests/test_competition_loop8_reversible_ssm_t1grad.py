import unittest
from pathlib import Path
import torch
import torch.nn.functional as F
from benchmark import ModelSpec, OptimizerSpec, count_model_state_elements
from submission_validation import validate_submission_source
import competition_loop8_reversible_ssm_t1grad as c
import competition_submission_c1a as c1a


def spec(length=13): return ModelSpec(17, length, 20_000_000)
def row(t=1, x=(8,12), n=(10,9,10), padding=0):
    td=[c.DIGIT+t] if t<10 else [c.DIGIT+t//10,c.DIGIT+t%10]
    return torch.tensor([[c.N,*n,c.X,*x,c.T,*td]+[c.PAD]*padding])


class ReversibleSSMTests(unittest.TestCase):
    def test_inventory_seed_width_optimizer_and_source(self):
        torch.manual_seed(74); a=c.build_model(spec())
        torch.manual_seed(74); b=c.Model(spec()); torch.manual_seed(74); wide=c.Model(spec(24))
        self.assertEqual(count_model_state_elements(a),3852)
        self.assertEqual(set(a.state_dict()),set(wide.state_dict()))
        self.assertEqual([type(x).__name__ for x in a.modules()].count("Linear"),5)
        self.assertIs(c.build_optimizer,c1a.build_optimizer)
        for k,v in a.state_dict().items():
            self.assertEqual(v.numpy().tobytes(),b.state_dict()[k].numpy().tobytes())
            self.assertEqual(v.shape,wide.state_dict()[k].shape)
        oa=c.build_optimizer(a,OptimizerSpec(60,"cpu")); ob=c.build_optimizer(b,OptimizerSpec(60,"cpu"))
        self.assertEqual(oa.optimizer.state_dict(),ob.optimizer.state_dict())
        self.assertEqual(oa.scheduler.state_dict(),ob.scheduler.state_dict())
        source=Path(c.__file__).read_text(encoding="utf-8")
        validate_submission_source(Path(c.__file__).name,source,256*1024,required_filename=None)

    def test_parse_features_context_initial_and_kernel(self):
        m=c.Model(spec()).eval(); _,a=m(row(3))
        self.assertEqual(a["context_digits"][0,:4].tolist(),[3,2,3,0])
        self.assertEqual(a["initial_state_digits"][0,:4].tolist(),[5,1,0,0])
        self.assertEqual(a["parsed_steps"].item(),3)
        torch.testing.assert_close(a["context"],m.embedding(a["context_digits"])+a["place_features"][None])
        torch.testing.assert_close(a["initial_state"],m.embedding(a["initial_state_digits"])+a["place_features"][None])
        torch.testing.assert_close(c.fixed_place_features(5,"cpu",torch.float32),c.fixed_place_features(9,"cpu",torch.float32)[:5],atol=0,rtol=0)
        self.assertFalse(any("place" in key or "position" in key for key in m.state_dict()))
        active=torch.tensor([[True,True,True,False]])
        k=m.transition.F.kernel(4,active,torch.float32)
        self.assertTrue((k[:,:3,:3]>0).all()); torch.testing.assert_close(k[:,:3].sum(2),torch.ones(1,3,16))
        self.assertTrue((k[:,:,3]==0).all())
        positions=torch.arange(3); distance=(positions[:,None]-positions[None,:]).abs()
        raw=torch.sigmoid(m.transition.F.decay_logit)
        unnormalized=raw[None,None,:].pow(distance[:,:,None])
        self.assertTrue(torch.equal(unnormalized,unnormalized.transpose(0,1)))

    def test_exact_coupling_calls_freeze_readout_and_gate(self):
        m=c.Model(spec()).train(); state=torch.randn(2,5,32); context=torch.randn(2,5,32); mask=torch.ones(2,5,dtype=torch.bool)
        xa,xb=state.chunk(2,-1); ca,cb=context.chunk(2,-1)
        f=m.transition.F(xb,cb,mask); ya=xa+torch.tanh(m.transition.raw_scale_f)*f
        expected=torch.cat((ya,xb+torch.tanh(m.transition.raw_scale_g)*m.transition.G(ya,ca,mask)),-1)
        torch.testing.assert_close(m.transition(state,context,mask),expected)
        self.assertAlmostEqual(torch.tanh(m.transition.raw_scale_f).item(),.1,places=6)
        counts={"transition":0,"F":0,"G":0,"readout":0}; transition_outputs=[]
        hooks=[getattr(m,n).register_forward_hook(lambda _,__,___,n=n:counts.__setitem__(n,counts[n]+1)) for n in ("readout",)]
        def capture_transition(_,__,output):
            counts["transition"]+=1; output.retain_grad(); transition_outputs.append(output)
        hooks += [m.transition.register_forward_hook(capture_transition),m.transition.F.register_forward_hook(lambda *_:counts.__setitem__("F",counts["F"]+1)),m.transition.G.register_forward_hook(lambda *_:counts.__setitem__("G",counts["G"]+1))]
        mixed=torch.cat((row(1),row(3))); out,aux=m(mixed); [h.remove() for h in hooks]
        self.assertEqual(counts,{"transition":3,"F":3,"G":3,"readout":1})
        isolated=m(row(1))[1]["state"]; torch.testing.assert_close(aux["state"][:1],isolated,atol=2e-6,rtol=2e-6)
        F.cross_entropy(out[1,-2:,c.DIGIT:],torch.tensor([1,2])).backward()
        self.assertTrue(all(value.grad is not None and value.grad.abs().sum()>0 for value in transition_outputs))
        for horizon,scale in ((1,1.),(2,.01),(3,.01)):
            torch.manual_seed(9); model=c.Model(spec()).train(); logits,a=model(row(horizon)); target=torch.tensor([1,2])
            gated=F.cross_entropy(logits[0,-2:,c.DIGIT:],target); plain=F.cross_entropy(a["ungated_logits"][0,-2:,c.DIGIT:],target)
            gg=torch.autograd.grad(gated,a["ungated_logits"],retain_graph=True)[0]; pg=torch.autograd.grad(plain,a["ungated_logits"])[0]
            torch.testing.assert_close(gg,pg*scale,atol=2e-8,rtol=2e-6)
            torch.manual_seed(9); gated_model=c.Model(spec()).train()
            torch.manual_seed(9); plain_model=c.Model(spec()).train()
            gated_output=gated_model(row(horizon))[0]; plain_aux=plain_model(row(horizon))[1]
            F.cross_entropy(gated_output[0,-2:,c.DIGIT:],target).backward()
            F.cross_entropy(plain_aux["ungated_logits"][0,-2:,c.DIGIT:],target).backward()
            for ga,pa in zip(gated_model.parameters(),plain_model.parameters()):
                torch.testing.assert_close(ga.grad,pa.grad*scale,atol=2e-8,rtol=2e-5)

    def test_invariance_bf16_t64_and_capacity(self):
        torch.manual_seed(17); m=c.Model(spec()).eval(); mixed=torch.cat((row(1),row(2),row(3)))
        before={key:value.clone() for key,value in m.state_dict().items()}
        with torch.no_grad():
            together=m(mixed)[0]; separate=torch.cat([m(row(t))[0] for t in (1,2,3)]); again=m(mixed)[0]
        torch.testing.assert_close(together,separate,atol=3e-6,rtol=3e-6)
        self.assertTrue(torch.equal(together,again))
        for key,value in m.state_dict().items(): self.assertTrue(torch.equal(value,before[key]))
        padded=row(2,padding=3); other=row(3,padding=3)
        with torch.no_grad(): single=m(row(2))[0]; batched=m(torch.cat((padded,other)))[0]
        torch.testing.assert_close(single,batched[:1,:single.shape[1]],atol=3e-6,rtol=3e-6)
        exact=torch.tensor([[c.N,8,10,9,10,c.X,8,9,10,11,c.T,13,11]])
        self.assertEqual(m(exact)[0].shape,(1,13,17))
        with self.assertRaises(ValueError): m(torch.cat((exact,torch.zeros(1,1,dtype=torch.long)),1))
        self.assertEqual(m(row(64))[1]["active_updates"].item(),64)
        m.train(); m.zero_grad()
        with torch.autocast("cpu",dtype=torch.bfloat16): output,_=m(row(3)); loss=F.cross_entropy(output[0,-2:,c.DIGIT:],torch.tensor([1,2]))
        loss.backward(); self.assertTrue(torch.isfinite(loss)); self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters()))
        x=torch.tensor([[2,10,9,10,3,8,12,4,8],[2,10,9,10,3,15,16,4,8],[2,10,9,10,3,11,9,4,8]])
        y=torch.tensor([[9,9,12],[8,13,16],[8,11,16]]); pos=torch.tensor([[6,7,8]]*3); bi=torch.arange(3)[:,None]
        torch.manual_seed(74); m=c.Model(spec()).train(); opt=torch.optim.AdamW(m.parameters(),lr=3e-3)
        for _ in range(100):
            opt.zero_grad(); selected=m(x)[0][bi,pos]; F.cross_entropy(selected.flatten(0,1),y.flatten()).backward(); opt.step()
            if selected.argmax(-1).eq(y).all(): break
        self.assertTrue(selected.argmax(-1).eq(y).all()); m.eval()
        with torch.no_grad(): self.assertTrue(m(x)[0][bi,pos].argmax(-1).eq(y).all())


if __name__ == "__main__": unittest.main()
