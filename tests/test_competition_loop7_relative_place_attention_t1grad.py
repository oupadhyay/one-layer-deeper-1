import unittest
from pathlib import Path
import torch
import torch.nn.functional as F
from benchmark import ModelSpec, OptimizerSpec, count_model_state_elements
from submission_validation import validate_submission_source
import competition_loop7_relative_place_attention_t1grad as c
import competition_submission_c1a as c1a


def spec(length=13): return ModelSpec(17, length, 20_000_000)
def row(t=1, x=(8,12), n=(10,9,10), padding=0):
    td=[c.DIGIT+t] if t<10 else [c.DIGIT+t//10,c.DIGIT+t%10]
    return torch.tensor([[c.N,*n,c.X,*x,c.T,*td]+[c.PAD]*padding])


class RelativePlaceAttentionTests(unittest.TestCase):
    def test_state_seed_width_optimizer_source(self):
        torch.manual_seed(74); a=c.build_model(spec())
        torch.manual_seed(74); b=c.Model(spec())
        torch.manual_seed(74); wide=c.Model(spec(24))
        self.assertEqual(count_model_state_elements(a),5362)
        self.assertEqual(set(a.state_dict()),set(wide.state_dict()))
        for k,v in a.state_dict().items():
            self.assertEqual(v.numpy().tobytes(),b.state_dict()[k].numpy().tobytes())
            self.assertEqual(v.numpy().tobytes(),wide.state_dict()[k].numpy().tobytes())
        self.assertIs(c.build_optimizer,c1a.build_optimizer)
        oa=c.build_optimizer(a,OptimizerSpec(60,"cpu")); ob=c.build_optimizer(b,OptimizerSpec(60,"cpu"))
        self.assertEqual(oa.optimizer.state_dict(),ob.optimizer.state_dict())
        self.assertEqual(oa.scheduler.state_dict(),ob.scheduler.state_dict())
        source=Path(c.__file__).read_text(encoding="utf-8")
        validate_submission_source(Path(c.__file__).name,source,256*1024,required_filename=None)

    def test_parse_places_width_and_calls(self):
        m=c.Model(spec()).eval(); calls={"block":0,"attention":0,"readout":0}
        hooks=[m.block.register_forward_hook(lambda *_:calls.__setitem__("block",calls["block"]+1)),
               m.block.attention.register_forward_hook(lambda *_:calls.__setitem__("attention",calls["attention"]+1)),
               m.readout.register_forward_hook(lambda *_:calls.__setitem__("readout",calls["readout"]+1))]
        out,aux=m(row(3)); [h.remove() for h in hooks]
        self.assertEqual(calls,{"block":3,"attention":3,"readout":3})
        self.assertEqual(aux["context_digits"][0,:4].tolist(),[3,2,3,0])
        self.assertEqual(aux["initial_state_digits"][0,:4].tolist(),[5,1,0,0])
        self.assertEqual(aux["width_mask"][0,:5].tolist(),[True,True,True,False,False])
        torch.testing.assert_close(c.fixed_place_features(5,"cpu",torch.float32),
                                   c.fixed_place_features(9,"cpu",torch.float32)[:5],atol=0,rtol=0)
        self.assertFalse(any("position" in k or "place" in k for k in m.state_dict()))
        self.assertEqual(out.shape,(1,row(3).shape[1],17))

    def test_feedback_freeze_gate_gradients_and_t64(self):
        for training in (True,False):
            torch.manual_seed(7); m=c.Model(spec()).train(training); inputs=[]; outputs=[]
            original=m.block.forward
            def capture(tokens,mask): inputs.append(tokens.clone()); return original(tokens,mask)
            m.block.forward=capture
            def capture_readout(_,__,output):
                if output.requires_grad: output.retain_grad()
                outputs.append(output)
            hook=m.readout.register_forward_hook(capture_readout)
            logits,aux=m(row(3)); hook.remove()
            expected=(torch.softmax(outputs[0],-1) if training else F.one_hot(outputs[0].argmax(-1),10).to(outputs[0].dtype))
            expected=torch.where(aux["width_mask"][:,:,None],expected,F.one_hot(torch.zeros_like(aux["initial_state_digits"]),10).to(expected.dtype))
            places=aux["place_features"][None]; role=places.new_zeros((1,1,c.D_MODEL)); role[...,0]=1
            torch.testing.assert_close(inputs[1][:,:m.max_seq_len],expected@m.embedding.weight+places+role,atol=0,rtol=0)
            if training:
                F.cross_entropy(logits[0,-2:,c.DIGIT:],torch.tensor([1,2])).backward()
                self.assertTrue(all(value.grad is not None and value.grad.abs().sum()>0 for value in outputs))
            else:
                self.assertTrue(((aux["digit_probabilities"]==0)|(aux["digit_probabilities"]==1)).all())
        for horizon,scale in ((1,1.),(2,.01),(3,.01)):
            torch.manual_seed(9); m=c.Model(spec()).train(); gated,aux=m(row(horizon)); target=torch.tensor([1,2])
            gl=F.cross_entropy(gated[0,-2:,c.DIGIT:],target); pl=F.cross_entropy(aux["ungated_logits"][0,-2:,c.DIGIT:],target)
            gg=torch.autograd.grad(gl,aux["ungated_logits"],retain_graph=True)[0]
            pg=torch.autograd.grad(pl,aux["ungated_logits"])[0]
            torch.testing.assert_close(gg,pg*scale,atol=2e-8,rtol=2e-6)
            torch.manual_seed(9); gated_model=c.Model(spec()).train()
            torch.manual_seed(9); plain_model=c.Model(spec()).train()
            gated_output=gated_model(row(horizon))[0]; plain_aux=plain_model(row(horizon))[1]
            F.cross_entropy(gated_output[0,-2:,c.DIGIT:],target).backward()
            F.cross_entropy(plain_aux["ungated_logits"][0,-2:,c.DIGIT:],target).backward()
            for ga,pa in zip(gated_model.parameters(),plain_model.parameters()):
                torch.testing.assert_close(ga.grad,pa.grad*scale,atol=2e-8,rtol=2e-5)
        m=c.Model(spec()).eval(); _,aux=m(row(64)); self.assertEqual(aux["active_updates"].item(),64)

    def test_invariance_bf16_and_capacity(self):
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
        m.train(); m.zero_grad()
        with torch.autocast("cpu",dtype=torch.bfloat16):
            output,_=m(row(3)); loss=F.cross_entropy(output[0,-2:,c.DIGIT:],torch.tensor([1,2]))
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
