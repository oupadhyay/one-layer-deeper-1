import unittest
from pathlib import Path
import torch
import torch.nn.functional as F
from benchmark import ModelSpec, OptimizerSpec, count_model_state_elements
from submission_validation import validate_submission_source
import competition_loop13_compressed_axial_cellular as candidate


def spec(length=13): return ModelSpec(17, length, 100_000)
def row(n=(3,2,3), x=(1,2,3), t=(1,), padding=0):
    return torch.tensor([[candidate.N, *(d+candidate.DIGIT for d in n), candidate.X,
                          *(d+candidate.DIGIT for d in x), candidate.T,
                          *(d+candidate.DIGIT for d in t)] + [candidate.PAD]*padding])


class CompressedAxialCellularTests(unittest.TestCase):
    def test_state_optimizer_and_source_contract(self):
        torch.manual_seed(74); model=candidate.build_model(spec())
        torch.manual_seed(74); repeat=candidate.build_model(spec())
        torch.manual_seed(74); wide=candidate.build_model(spec(32))
        self.assertEqual(count_model_state_elements(model), candidate.STATE_ELEMENTS)
        self.assertLess(count_model_state_elements(model), 100_000)
        self.assertEqual(set(model.state_dict()), set(wide.state_dict()))
        for key,value in model.state_dict().items():
            self.assertEqual(value.numpy().tobytes(), repeat.state_dict()[key].numpy().tobytes())
            self.assertEqual(value.shape, wide.state_dict()[key].shape)
        bundle=candidate.build_optimizer(model,OptimizerSpec(1600,"cpu"))
        self.assertEqual([g["weight_decay"] for g in bundle.optimizer.param_groups],[.02,0.])
        self.assertEqual(bundle.optimizer.defaults["lr"],8e-4)
        self.assertEqual(bundle.optimizer.defaults["betas"],(.9,.98))
        self.assertAlmostEqual(bundle.scheduler.get_last_lr()[0],4e-5)
        source=Path(candidate.__file__).read_text(encoding="utf-8")
        validate_submission_source(Path(candidate.__file__).name,source,256*1024,required_filename=None)
        for forbidden in ("Conv2d","2 * w","competition_submission","torch.load","remainder","carry"):
            self.assertNotIn(forbidden,source)

    def test_workspace_boundaries_locality_order_sharing_and_resets(self):
        model=candidate.build_model(spec()).train()
        inputs=torch.cat((row(padding=2),row(n=(9,0,0,1),x=(1,2,3,4),t=(3,))))
        axes=[]; identities=[]; resets=0
        original=model.cell.forward
        def observed(workspace,axis):
            axes.append(axis); identities.append(id(model.cell)); return original(workspace,axis)
        model.cell.forward=observed
        def init_hook(*_):
            nonlocal resets; resets+=1
        hook=model.initializer.register_forward_hook(init_hook)
        _,aux=model(inputs); hook.remove()
        self.assertEqual(aux["parsed_steps"].tolist(),[1,3]); self.assertEqual(aux["widths"].tolist(),[3,4])
        self.assertEqual(axes,["H","V","H","V"]*3); self.assertEqual(len(set(identities)),1)
        self.assertEqual(len(axes),12); self.assertEqual(resets,3)
        probabilities=F.one_hot(torch.tensor([[1,2,3,0],[1,2,3,4]]),10).float()
        workspace,valid=model._workspace(probabilities,torch.tensor([[3,2,3,0],[9,0,0,1]]),torch.tensor([3,4]))
        self.assertEqual(torch.count_nonzero(workspace[~valid]).item(),0)
        self.assertTrue(valid[0,:3,3].all()); self.assertFalse(valid[0,:,4].any())
        cell=candidate.AxialLocalCell().eval(); base=torch.zeros(1,7,8,32)
        changed=base.clone(); changed[0,3,4,0]=1
        for axis,allowed in (("H",[(3,3),(3,4),(3,5)]),("V",[(2,4),(3,4),(4,4)])):
            with torch.no_grad(): delta=(cell(changed,axis)-cell(base,axis)).abs().sum(-1)[0]
            outside=torch.ones_like(delta,dtype=torch.bool)
            for point in allowed: outside[point]=False
            self.assertEqual(torch.count_nonzero(delta[outside]).item(),0)

    def test_feedback_gradient_gate_padding_alignment_and_no_carry(self):
        for horizon,scale in ((1,1.),(2,.01),(3,.01)):
            torch.manual_seed(9); gated=candidate.build_model(spec()).train()
            logits,aux=gated(row(t=(horizon,))); self.assertTrue(torch.equal(aux["digit_probabilities"],aux["digit_probabilities"].round()))
            F.cross_entropy(logits[0,-3:,candidate.DIGIT:],torch.tensor([1,2,3])).backward()
            torch.manual_seed(9); plain=candidate.build_model(spec()).train()
            _,pa=plain(row(t=(horizon,))); F.cross_entropy(pa["ungated_logits"][0,-3:,candidate.DIGIT:],torch.tensor([1,2,3])).backward()
            for a,b in zip(gated.parameters(),plain.parameters()):
                torch.testing.assert_close(a.grad,b.grad*scale,atol=2e-8,rtol=4e-5)
        torch.manual_seed(17); model=candidate.build_model(spec()).eval()
        narrow=row(t=(2,),padding=2); wide=row(n=(9,0,0,1),x=(1,2,3,4),t=(3,))
        with torch.no_grad():
            together=model(torch.cat((narrow,wide)))[0]; separate=torch.cat((model(narrow)[0],model(wide)[0]))
        torch.testing.assert_close(together,separate,atol=3e-6,rtol=3e-6)
        self.assertTrue(torch.equal(together,model(torch.cat((narrow,wide)))[0]))

    def test_bf16_t64_max_length_and_capacity(self):
        model=candidate.build_model(spec()).to(torch.bfloat16).train()
        out,aux=model(row(t=(6,4))); loss=F.cross_entropy(out[0,-3:,candidate.DIGIT:],torch.tensor([1,2,3])); loss.backward()
        self.assertTrue(torch.isfinite(loss)); self.assertEqual(aux["parsed_steps"].item(),64)
        self.assertEqual(aux["vectorized_cell_calls"],256)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in model.parameters()))
        exact=row(n=(9,0,0,1),x=(1,2,3,4),t=(3,)); self.assertEqual(candidate.build_model(spec(12))(exact)[0].shape,(1,12,17))
        with self.assertRaises(ValueError): candidate.build_model(spec(11))(exact)
        x=torch.tensor([[2,10,9,10,3,8,12,4,8],[2,10,9,10,3,15,16,4,8],[2,10,9,10,3,11,9,4,8]])
        labels=torch.tensor([[9,9,12],[8,13,16],[8,11,16]]); positions=torch.tensor([[6,7,8]]*3); batch=torch.arange(3)[:,None]
        torch.manual_seed(74); model=candidate.build_model(spec()).train(); optimizer=torch.optim.AdamW(model.parameters(),lr=3e-3)
        for update in range(1,101):
            optimizer.zero_grad(set_to_none=True); selected=model(x)[0][batch,positions]
            F.cross_entropy(selected.flatten(0,1),labels.flatten()).backward(); optimizer.step()
            if selected.argmax(-1).eq(labels).all(): break
        self.assertTrue(selected.argmax(-1).eq(labels).all()); self.assertLessEqual(update,100)


if __name__=="__main__": unittest.main()
