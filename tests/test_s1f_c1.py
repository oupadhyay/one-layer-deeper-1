import hashlib, inspect, math, unittest
import torch
from benchmark.s1_c1 import CONFIG, OneStepHarness, S1GRUTransducer
from benchmark.s1f_c1 import S1FixedNumericContextTransducer, fixed_numeric_table, table_sha, _initialization_check

class S1FTests(unittest.TestCase):
    def test_table_exact_and_hash(self):
        t=fixed_numeric_table(); self.assertEqual((10,32),tuple(t.shape)); self.assertEqual(torch.float32,t.dtype)
        expected=[]
        for d in range(10): expected.append([d/9,math.sin(2*math.pi*d/10),math.cos(2*math.pi*d/10),math.sin(4*math.pi*d/10),math.cos(4*math.pi*d/10),math.sin(6*math.pi*d/10),math.cos(6*math.pi*d/10),1]+[0]*24)
        self.assertTrue(torch.equal(t,torch.tensor(expected,dtype=torch.float32)))
        self.assertEqual(table_sha(),hashlib.sha256(t.numpy().tobytes()).hexdigest())
    def test_buffer_persistent_nontrainable(self):
        m=S1FixedNumericContextTransducer(); self.assertIn("fixed_context_table",dict(m.named_buffers())); self.assertIn("fixed_context_table",m.state_dict()); self.assertFalse(m.fixed_context_table.requires_grad)
        m(torch.tensor([[0,2,2,2,2,1,3,2,2,2]])).sum().backward(); self.assertIsNone(m.fixed_context_table.grad)
    def test_locality_and_api_purity(self):
        m=S1FixedNumericContextTransducer().eval(); x=torch.tensor([[1,2,3,4]]); y=x.clone(); y[0,2]=9
        a,b=m.encode_context(x),m.encode_context(y); self.assertTrue(torch.equal(a[:,:2],b[:,:2])); self.assertTrue(torch.equal(a[:,3:],b[:,3:])); self.assertFalse(torch.equal(a[:,2],b[:,2]))
        ids=torch.tensor([[0,2,3,4,5,1,3,4,5,6]]); self.assertTrue(torch.equal(m(ids),OneStepHarness(m).run(ids)))
    def test_only_intervention_and_initialization(self):
        self.assertTrue(_initialization_check()); p=S1GRUTransducer(); m=S1FixedNumericContextTransducer()
        self.assertEqual(25930,sum(x.numel() for x in m.parameters())); self.assertEqual(type(p.cell),type(m.cell)); self.assertEqual(type(p.embedding),type(m.embedding)); self.assertEqual(type(p.digit_head),type(m.digit_head)); self.assertIs(m.encode_state.__func__,p.encode_state.__func__)
        self.assertIs(m.readout.__func__,p.readout.__func__); self.assertIs(m.forward.__func__,p.forward.__func__); self.assertEqual(list(inspect.signature(m.step.forward).parameters),["state","context"])
    def test_bad_context_shape(self):
        with self.assertRaises(ValueError): S1FixedNumericContextTransducer().encode_context(torch.zeros(2,3,dtype=torch.long))

if __name__ == "__main__": unittest.main()
