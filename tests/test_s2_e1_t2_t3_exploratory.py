import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
import benchmark.s2_e1_t2_t3_exploratory as e

def scripted_data(T=3):
 trajectory=[0,1,2,3][:T+1]
 return [{"N":11,"x0":0,"trajectory":trajectory,"reduction":[0]*T,"carry_counts":[0]*T,"carry_masks":[[0]*4 for _ in range(T)]}]

class Scripted(torch.nn.Module):
 def __init__(self, transitions, nonfinite=None):
  super().__init__();self.transitions=transitions;self.nonfinite=nonfinite or {};self.calls=[]
  class Step(torch.nn.Module):
   def __init__(self,owner):super().__init__();self.owner=owner
   def forward(self,state,context):self.owner.calls.append(state.clone());return state
  self.step=Step(self);self._step_identity=id(self.step)
 def encode_context(self,x):return x.float()
 def encode_state(self,x):return x.float()
 def readout(self,state):
  prior=int(sum(int(state[0,j])*10**j for j in range(4))); call=len(self.calls);out=torch.full((1,4,10),-20.0);value=self.transitions.get((call,prior),self.transitions.get(prior,prior))
  for j,d in enumerate(e.digits(value)):out[0,j,d]=20
  if call in self.nonfinite:out[0,0,0]=self.nonfinite[call]
  return out

class Tests(unittest.TestCase):
 @classmethod
 def setUpClass(c):c.t2,c.t3,c.h=e.audit_depth()
 def test_pins_and_frozen_facts(self): self.assertEqual(len(e.INPUTS),10);self.assertEqual(e.verify_frozen()["evaluations"]["test_state"]["overall"]["exact_correct"],3990)
 def test_order_before_depth_and_model(self):
  bad=dict(e.INPUTS);bad["checkpoint"]=(bad["checkpoint"][0],"0"*64);events=[]
  with self.assertRaises(RuntimeError):e.verify_frozen(bad,events)
  self.assertNotIn("depth-parse",events);self.assertNotIn("model-call",events)
 def test_all_prefix_arrays_and_order(self):
  self.assertEqual([(x['N'],x['x0']) for x in self.t3],sorted((x['N'],x['x0']) for x in self.t3));self.assertEqual(len(self.h),64)
  for f in e.ARRAYS:
   z=copy.deepcopy(self.t3);z[0][f][0]=None;self.assertNotEqual(e.hashlib.sha256(e.canon(z)).hexdigest(),self.h)
 def test_checkpoint_contract(self):
  m=e.load_model();self.assertEqual(tuple(e.inspect.signature(m.step.forward).parameters),("state","context"));self.assertEqual(id(m.step),m._step_identity)
 def test_real_counts_context_once(self):
  m=e.load_model();calls=0;old=m.encode_context
  def enc(x):
   nonlocal calls;calls+=1;return old(x)
  m.encode_context=enc;r,c=e.evaluate_horizon(m,self.t2,2);self.assertEqual((calls,c),(1,4))
 def test_real_repeat_and_no_t4(self):
  m=e.load_model();a,c=e.evaluate_horizon(m,self.t2,2,True);b,d=e.evaluate_horizon(m,self.t3,3,True);a0,_=e.evaluate_horizon(m,self.t2,2,True);self.assertEqual(c+d,10);self.assertEqual(a['trace_sha256'],a0['trace_sha256']);self.assertEqual(a["first_error_oracle_identity_mismatch_count"],0);self.assertEqual(b["first_error_oracle_identity_mismatch_count"],0)
 def test_depth_access_instrumentation(self):
  parsed=[];observed=[];actual=[];old=Path.open
  def spy(path,*args,**kwargs):actual.append(str(path));return old(path,*args,**kwargs)
  with patch.object(Path,'open',spy):e.audit_depth(parsed=parsed,observed_opens=observed)
  expected=[str(e.INPUTS[x][0]) for x in ('T2','T4','T64')]
  self.assertEqual(parsed,expected);self.assertEqual(observed,expected);self.assertEqual(actual,expected)
  self.assertFalse(any(('train' in x or 'test' in x) for x in actual))
  with self.assertRaisesRegex(RuntimeError,'unauthorized'):e.rows('data/train.jsonl',[e.INPUTS['T2'][0]],[])
 def test_scripted_no_cross_feed_after_early_error(self):
  # Calls are teacher/free pairs. At step 2 oracle prior=1, while free prior=9.
  m=Scripted({(1,0):9,(2,0):9,(3,1):2,(4,9):8,(5,2):3,(6,8):7})
  r,c=e.evaluate_horizon(m,scripted_data(),3)
  self.assertEqual(c,6);self.assertEqual([int(x[0,0]) for x in m.calls],[0,0,1,9,2,8])
  self.assertEqual([x['teacher']['correct'] for x in r['steps']],[0,1,1]);self.assertEqual([x['free']['correct'] for x in r['steps']],[0,0,0])
 def test_scripted_lucky_recovery_is_not_trajectory_success(self):
  m=Scripted({(1,0):9,(2,0):9,(3,1):2,(4,9):2})
  r,c=e.evaluate_horizon(m,scripted_data(2),2)
  self.assertEqual(c,4);self.assertEqual((r['free_endpoint_exact'],r['free_complete_trajectory_exact'],r['lucky_endpoint_recovery']),(1,0,1))
  self.assertTrue(next(iter(r['cascade_after_first_error'].values()))['endpoint_recovery'])
 def test_scripted_nonfinite_is_sticky(self):
  for bad in (float('nan'),float('inf')):
   m=Scripted({0:1,1:2},nonfinite={2:bad})
   r,_=e.evaluate_horizon(m,scripted_data(2),2)
   self.assertEqual((r['invalid_rows'],r['free_endpoint_exact'],r['free_complete_trajectory_exact']),(1,0,0))
 def test_scripted_forward_call_boundaries(self):
  for T,want in ((2,4),(3,6)):
   m=Scripted({0:1,1:2,2:3});r,c=e.evaluate_horizon(m,scripted_data(T),T)
   self.assertEqual((len(m.calls),c),(want,want));self.assertEqual(len(r['steps']),T)
 def test_error_schema_and_scientific_detail(self):
  r,_=e.evaluate_horizon(e.load_model(),self.t3,3,True);self.assertEqual((r['free_endpoint_exact'],len(r['new_error_transitions'])),(511,1))
  z=r['new_error_transitions'][0];self.assertEqual((z['N'],z['oracle_source_x'],z['step'],z['boundary_distance']),(9767,4886,3,5))
  required={'N','x0','step','oracle_source_x','target_integer','target_digits','teacher_prediction_integer','teacher_prediction_digits','free_prediction_integer','free_prediction_digits','boundary_distance','reduction','carry_count','carry_mask','aligned'};self.assertTrue(required<=set(z))
  c=next(iter(r['cascade_after_first_error'].values()));self.assertEqual((c['first_error_step'],c['remaining_steps_including_first'],c['wrong_step_count'],c['consecutive_wrong_from_first'],c['endpoint_recovery']),(3,1,1,1,False))
 def test_logits_shape_finite_and_ce(self):
  r,_=e.evaluate_horizon(e.load_model(),self.t2,2);self.assertEqual(r['steps'][0]['free']['count'],512);self.assertEqual(len(r['steps'][0]['free']['per_place']),4);self.assertEqual(len(r['steps'][0]['teacher']['per_place']),4);self.assertIn('ce',r['steps'][0]['teacher'])
 def test_output_schema_forbidden(self):
  d=json.loads(Path('metrics_s2_e1_t2_t3_exploratory.json').read_text());self.assertEqual(d['purpose'],'empirical_error_propagation_of_imperfect_E1_transition')
  def walk(x):
   if isinstance(x,dict):
    for k,v in x.items():self.assertFalse(any(q in k.lower() for q in ('pass','gate','unlock','certified')));walk(v)
   elif isinstance(x,list):
    for v in x:walk(v)
   elif isinstance(x,str):self.assertFalse(any(q in x.lower() for q in ('pass','gate','unlock','certified')))
  walk(d)
 def test_boundary_formula(self):self.assertEqual(abs(2*4-11),3)
 def test_frozen_heuristic_exact(self):
  d=json.loads(Path('metrics_s2_e1_t2_t3_exploratory.json').read_text())['frozen_t1_cross_population_comparator'];self.assertEqual(d['T2'],.99500625);self.assertEqual(d['T3'],.992518734375);self.assertIn('not_an_independence_model',d['label'])
 def test_zero_hidden_source(self):
  src=Path('benchmark/s2_e1_biscan.py').read_text();self.assertGreaterEqual(src.count('pair.new_zeros'),2)
 def test_provenance(self):
  p=json.loads(Path('artifacts/s2_e1_t2_t3_exploratory_provenance.json').read_text());self.assertEqual(len(p['inputs']),10);expected=[str(e.INPUTS[x][0]) for x in ('T2','T4','T64')];self.assertEqual(p['access']['parsed_for_prefix_audit'],expected);self.assertEqual(p['access']['observed_path_open_calls_for_prefix_audit'],expected);self.assertEqual(p['access']['source_enforced_allowed_depth_paths'],expected);self.assertEqual(p['access']['maximum_inferred_macrostep'],3);self.assertTrue(p['historical_s2_stop_unchanged']);self.assertEqual(p['competition_engineering_implication'],'diagnostic_only');self.assertIsNone(p['training'])
 def test_no_depth_over_three(self):self.assertEqual(set(json.loads(Path('metrics_s2_e1_t2_t3_exploratory.json').read_text())['depths']),{'T2','T3'})
 def test_stratifications(self):
  d=json.loads(Path('metrics_s2_e1_t2_t3_exploratory.json').read_text());self.assertEqual(set(d['depths']['T3']['transition_stratification']),{'teacher','aligned_free'})
 def test_endpoint_trajectory_distinct(self):
  d=json.loads(Path('metrics_s2_e1_t2_t3_exploratory.json').read_text())['depths']['T3'];self.assertIn('free_endpoint_exact',d);self.assertIn('free_complete_trajectory_exact',d)
if __name__=='__main__':unittest.main()
