"""Pinned, read-only E1 T2/T3 error-propagation evaluator."""
from __future__ import annotations
import argparse, hashlib, inspect, json, math, platform, sys
from collections import Counter, defaultdict
from pathlib import Path
import torch
import torch.nn.functional as F
from benchmark.s2_e1_biscan import S2BiScan

ROOT=Path("data/generated/s2_doublemod_seen_dense_v1"); CHECKPOINT=Path("artifacts/s2_e1_biscan_seed79.pt")
INPUTS={
 "checkpoint":(CHECKPOINT,"d0b6ef957d1876d226b2be7021d72986529f0c5053e4d17f35cb77099b9e14fb"),
 "e1_runner":(Path("benchmark/s2_e1_biscan.py"),"62ce733484eee46b313e550ee88bd1c2274a1b5ff2d8d6c87b5c251d030aa98a"),
 "e1_metrics":(Path("metrics_s2_e1_biscan_seed79.jsonl"),"2687220cfcee25e98fa449ff6c993b957309b44389151a8b0ff54a436a238793"),
 "e1_provenance":(Path("artifacts/s2_e1_biscan_seed79_provenance.json"),"39073932d08b6e74d54dec3f5b87d04e33c2319b7d9c53a42dfa94ebd5dc9249"),
 "e1_predeclaration":(Path("artifacts/s2_e1_biscan_seed79_predeclaration.json"),"f7c93a840c3b6ec99ddc4afdc207603285b103599ba5dc43afa029700d47e98e"),
 "dense_config":(ROOT/"dataset_config.json","10c26a868b02403b68f7f0348771b52866fcd1de8fb1978a6ce2f412327f2a98"),
 "manifest":(ROOT/"artifact_manifest.json","817e2df936909427934fdbe8f59ea1d596649c3e14eb2facedce6c97c64bf578"),
 "T2":(ROOT/"depth_t_2.jsonl","e8d54f79c3e51657d89138268fccc09950cdb6a43090c37fe5a08c81dac87435"),
 "T4":(ROOT/"depth_t_4.jsonl","8bed5b1df217a26035c9888bbdb634d24aed056b049b6178d03053ee0f25b951"),
 "T64":(ROOT/"depth_t_64.jsonl","1bec3d8b712620a91072a6e52e39f88f0476bf03c0120ca701a20b81f9718d97")}
ARRAYS=("trajectory","reduction","carry_counts","carry_masks")
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canon(x): return json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()
def digits(x): return [x//10**i%10 for i in range(4)]
def rows(p, allowed=None, parsed=None):
 p=Path(p)
 if allowed is not None and p not in {Path(x) for x in allowed}:raise RuntimeError("unauthorized depth parse: "+str(p))
 if parsed is not None:parsed.append(str(p))
 with p.open(encoding="utf-8") as f:return [json.loads(x) for x in f]

def verify_frozen(inputs=INPUTS, event=None):
 """This function must complete before depth JSON decoding or model construction."""
 for name,(path,want) in inputs.items():
  if event:event.append("hash:"+name)
  if sha(path)!=want: raise RuntimeError("input pin mismatch: "+name)
 if event:event.append("metrics-parse")
 final=[json.loads(x) for x in Path(inputs["e1_metrics"][0]).read_text(encoding="utf-8").splitlines() if x.strip()][-1]
 tr=final["evaluations"]["train"]["overall"]; te=final["evaluations"]["test_state"]["overall"]
 a=final["threshold"]["all_abs1"]
 if not (final["kind"]=="final" and tr==tr and tr["count"]==tr["exact_correct"]==34316 and te["count"]==4000 and te["exact_correct"]==3990 and len(final["test_errors"])==10 and len(a)==16 and sum(x["exact"] for x in a)==7 and final["pass_for_independent_review"] is False): raise RuntimeError("frozen metrics facts mismatch")
 return final

def audit_depth(inputs=INPUTS,event=None,parsed=None,observed_opens=None):
 if event:event.append("depth-parse")
 allowed=[inputs[f"T{t}"][0] for t in (2,4,64)]; parsed=[] if parsed is None else parsed
 observed_opens=[] if observed_opens is None else observed_opens
 original_open=Path.open
 def monitored_open(path,*args,**kwargs):
  observed_opens.append(str(path))
  return original_open(path,*args,**kwargs)
 Path.open=monitored_open
 try:
  src={t:rows(inputs[f"T{t}"][0],allowed,parsed) for t in (2,4,64)}
 finally:
  Path.open=original_open
 maps={t:{(r["N"],r["x0"]):r for r in src[t]} for t in src}
 if any(len(src[t])!=512 or len(maps[t])!=512 for t in src):raise RuntimeError("depth cardinality")
 keys=sorted(maps[64]);
 if any(sorted(maps[t])!=keys for t in (2,4)):raise RuntimeError("key prefix")
 for k in keys:
  for field in ARRAYS:
   for small,big,n in ((2,4,3 if field=="trajectory" else 2),(2,64,3 if field=="trajectory" else 2),(4,64,5 if field=="trajectory" else 4)):
    if maps[small][k][field]!=maps[big][k][field][:n]:raise RuntimeError(f"{field} prefix T{small}/T{big}")
 out=[]
 for k in keys:
  z=dict(maps[64][k]); z.update(T=3,split="derived_depth_t_3");
  for f in ARRAYS:z[f]=z[f][:4 if f=="trajectory" else 3]
  z["y"]=z["trajectory"][-1]; z["labels"]=digits(z["y"]); out.append(z)
 return [maps[2][k] for k in keys],out,hashlib.sha256(canon(out)).hexdigest()

def load_model(path=CHECKPOINT,event=None):
 if event:event.append("model-call")
 p=torch.load(path,map_location="cpu",weights_only=True); cfg=p.get("config",{})
 expected={"seed":79,"optimizer_steps":5040}
 if any(cfg.get(k)!=v for k,v in expected.items()) or p.get("parameter_count")!=63178 or p.get("optimizer_steps")!=5040 or p.get("examples_seen")!=1281212 or any(k in p for k in ("optimizer","optimizer_state","optimizer_state_dict")):raise RuntimeError("checkpoint metadata")
 m=S2BiScan().cpu().float(); m.load_state_dict(p["state_dict"],strict=True); m.eval().requires_grad_(False)
 if tuple(inspect.signature(m.step.forward).parameters)!=("state","context") or sum(x.numel() for x in m.parameters())!=63178:raise RuntimeError("step identity")
 m._step_identity=id(m.step); return m

def _bucket_add(d,key,ok,ce):
 x=d.setdefault(str(key),{"count":0,"correct":0,"ce_sum":0.0});x["count"]+=1;x["correct"]+=int(ok);x["ce_sum"]+=ce
def _finish(d): return {k:{"count":v["count"],"correct":v["correct"],"ce":v["ce_sum"]/v["count"]} for k,v in sorted(d.items())}

def _trace(h, label, tensor):
 h.update(label.encode()+b"\0"); z=tensor.detach().cpu().contiguous();h.update(str(z.dtype).encode()+str(tuple(z.shape)).encode()+z.numpy().tobytes())

def _place_metrics(logits,predictions,target,bad):
 ce=F.cross_entropy(logits.flatten(0,1),target.flatten(),reduction="none").view(len(target),4)
 return [{"count":len(target),"correct":int((predictions[:,p].eq(target[:,p])&~bad).sum()),"ce":float(ce[:,p].double().mean())} for p in range(4)]

@torch.no_grad()
def evaluate_horizon(model,data,T,capture_trace=False):
 if id(model.step)!=model._step_identity:raise RuntimeError("step replaced")
 n=len(data); ctx_digits=torch.tensor([digits(r["N"]) for r in data]); context=model.encode_context(ctx_digits); context.requires_grad_(False)
 free=torch.tensor([digits(r["x0"]) for r in data]); invalid_any=torch.zeros(n,dtype=torch.bool); records=[]; steps=[]
 free_ok_history=[]; teacher_ok_history=[]; teacher_prefix_errors=[]; trace=hashlib.sha256() if capture_trace else None
 strat={lane:{f:[defaultdict(dict) for _ in range(T)] for f in ("N","reduction","carry_count","carry_mask")} for lane in ("teacher","aligned_free")}
 for s in range(T):
  target=torch.tensor([digits(r["trajectory"][s+1]) for r in data]); oracle=torch.tensor([digits(r["trajectory"][s]) for r in data])
  logits_t=model.readout(model.step(model.encode_state(oracle),context)); logits_f=model.readout(model.step(model.encode_state(free),context))
  if logits_t.shape!=(n,4,10) or logits_f.shape!=(n,4,10):raise RuntimeError("logits shape")
  bad_t=~torch.isfinite(logits_t).all((1,2)); bad_f=~torch.isfinite(logits_f).all((1,2)); safe_t=torch.nan_to_num(logits_t);safe_f=torch.nan_to_num(logits_f)
  pt=safe_t.argmax(-1);pf=safe_f.argmax(-1); pt[bad_t]=0;pf[bad_f]=0
  if trace:
   for label,z in (("teacher_logits",logits_t),("teacher_predictions",pt),("free_logits",logits_f),("free_predictions",pf)):_trace(trace,label,z)
  ce_t=F.cross_entropy(safe_t.flatten(0,1),target.flatten(),reduction="none").view(n,4).mean(1);ce_f=F.cross_entropy(safe_f.flatten(0,1),target.flatten(),reduction="none").view(n,4).mean(1)
  ot=pt.eq(target).all(1)&~bad_t; of=pf.eq(target).all(1)&~bad_f; aligned=torch.ones(n,dtype=torch.bool) if s==0 else torch.stack(free_ok_history).all(0)
  invalid_any|=bad_f; free=pf.clone(); free[bad_f]=0
  free_ok_history.append(of);teacher_ok_history.append(ot);teacher_prefix_errors.append(~ot)
  steps.append({"step":s+1,"free":{"count":n,"correct":int(of.sum()),"ce":float(ce_f.double().mean()),"per_place":_place_metrics(safe_f,pf,target,bad_f),"invalid_rows":int(bad_f.sum()),"invalid_events":int(bad_f.sum())},"free_oracle_aligned_prefix":{"eligible":int(aligned.sum()),"correct":int((of&aligned).sum()),"ce":float(ce_f[aligned].double().mean())},"teacher":{"count":n,"correct":int(ot.sum()),"ce":float(ce_t.double().mean()),"per_place":_place_metrics(safe_t,pt,target,bad_t),"invalid_rows":int(bad_t.sum()),"invalid_events":int(bad_t.sum())}})
  vals=(lambda r:(r["N"],r["reduction"][s],r["carry_counts"][s],"".join(map(str,r["carry_masks"][s]))))
  for i,r in enumerate(data):
   if not bool(ot[i]) or not bool(of[i]):
    source=r["trajectory"][s]; td=target[i].tolist(); tpd=pt[i].tolist(); fpd=pf[i].tolist()
    records.append({"N":r["N"],"x0":r["x0"],"step":s+1,"oracle_source_x":source,"target_integer":r["trajectory"][s+1],"target_digits":td,"teacher_prediction_integer":sum(v*10**j for j,v in enumerate(tpd)),"teacher_prediction_digits":tpd,"free_prediction_integer":sum(v*10**j for j,v in enumerate(fpd)),"free_prediction_digits":fpd,"boundary_distance":abs(2*source-r["N"]),"reduction":r["reduction"][s],"carry_count":r["carry_counts"][s],"carry_mask":r["carry_masks"][s],"aligned":bool(aligned[i]),"teacher_wrong":not bool(ot[i]),"free_wrong":not bool(of[i]),"teacher_invalid":bool(bad_t[i]),"free_invalid":bool(bad_f[i])})
   for lane,mask,ok,ce in (("teacher",True,ot[i],ce_t[i]),("aligned_free",bool(aligned[i]),of[i],ce_f[i])):
    if mask:
     for field,val in zip(("N","reduction","carry_count","carry_mask"),vals(r)):_bucket_add(strat[lane][field][s],val,bool(ok),float(ce))
 free_mat=torch.stack(free_ok_history); teach_mat=torch.stack(teacher_ok_history); trajectory=free_mat.all(0)&~invalid_any; endpoint=free_mat[-1]&~invalid_any
 first=[]; mismatch=0; cascade={}; new_errors=[]
 for i,r in enumerate(data):
  ff=next((s+1 for s in range(T) if not bool(free_mat[s,i])),None); tf=next((s+1 for s in range(T) if not bool(teach_mat[s,i])),None)
  first.append("none" if ff is None else str(ff)); mismatch+=int(ff is not None and ff!=tf)
  if ff is not None:
   outcomes=[bool(free_mat[s,i]) for s in range(ff-1,T)]; consecutive=next((j for j,v in enumerate(outcomes) if v),len(outcomes))
   cascade[str((r["N"],r["x0"]))]={"first_error_step":ff,"remaining_steps_including_first":T-ff+1,"wrong_step_count":sum(not v for v in outcomes),"consecutive_wrong_from_first":consecutive,"endpoint_recovery":bool(endpoint[i])}
 byn={}
 for N in sorted({r["N"] for r in data}):
  ix=[i for i,r in enumerate(data) if r["N"]==N];byn[str(N)]={"count":len(ix),"endpoint_correct":int(endpoint[ix].sum()),"trajectory_correct":int(trajectory[ix].sum())}
 histogram=Counter(first)
 return {"count":n,"free_endpoint_exact":int(endpoint.sum()),"free_complete_trajectory_exact":int(trajectory.sum()),"lucky_endpoint_recovery":int((endpoint&~free_mat.all(0)&~invalid_any).sum()),"invalid_rows":int(invalid_any.sum()),"invalid_events":sum(x["free"]["invalid_events"] for x in steps),"first_error_histogram":{str(s):histogram[str(s)] for s in range(1,T+1)}|{"none":histogram["none"]},"first_error_oracle_identity_mismatch_count":mismatch,"cascade_after_first_error":cascade,"steps":steps,"transition_stratification":{l:{f:[_finish(x) for x in arr] for f,arr in fs.items()} for l,fs in strat.items()},"endpoint_trajectory_by_N":byn,"new_error_transitions":records,"teacher_marginal_product_noncertifying":math.prod(x["teacher"]["correct"]/n for x in steps),"trace_sha256":trace.hexdigest() if trace else None},2*T

def run(metrics_path="metrics_s2_e1_t2_t3_exploratory.json",provenance_path="artifacts/s2_e1_t2_t3_exploratory_provenance.json"):
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 parsed=[];observed_opens=[];frozen=verify_frozen();t2,t3,derived=audit_depth(parsed=parsed,observed_opens=observed_opens);model=load_model();before={k:v.clone() for k,v in model.state_dict().items()}
 a2,c2=evaluate_horizon(model,t2,2,True);a3,c3=evaluate_horizon(model,t3,3,True);b2,d2=evaluate_horizon(model,t2,2,True);b3,d3=evaluate_horizon(model,t3,3,True)
 old={(x["N"],x["x"]) for x in frozen["test_errors"]}; overlap_report={}; union=set()
 for name,data,T in (("T2",t2,2),("T3",t3,3)):
  edge={(r["N"],x) for r in data for x in r["trajectory"][:T]};union|=edge;overlap_report[name]={"oracle_source_edge_count":len(edge),"frozen_error_count":len(old),"intersection_count":len(old&edge)}
 overlap_report["union"]={"oracle_source_edge_count":len(union),"frozen_error_count":len(old),"intersection_count":len(old&union)}
 if any(v["intersection_count"] for v in overlap_report.values()):raise RuntimeError("frozen T1 error/source-edge overlap")
 pure=all(torch.equal(v,model.state_dict()[k]) for k,v in before.items());repeat=a2["trace_sha256"]==b2["trace_sha256"] and a3["trace_sha256"]==b3["trace_sha256"]
 if not repeat:raise RuntimeError("bitwise repeat mismatch")
 access={"source_enforced_allowed_depth_paths":[str(INPUTS[x][0]) for x in ("T2","T4","T64")],"observed_path_open_calls_for_prefix_audit":observed_opens,"parsed_for_prefix_audit":parsed,"model_inference_inputs":{"T2":"pinned depth_t_2 rows","T3":"derived in memory from pinned depth_t_64 prefixes"},"maximum_inferred_macrostep":3}
 facts={"purpose":"empirical_error_propagation_of_imperfect_E1_transition","exploratory_only":True,"certification":False,"historical_s2_stop_unchanged":True,"competition_engineering_implication":"diagnostic_only","depths":{"T2":a2,"T3":a3},"frozen_e1_review_eligible_false":True,"frozen_t1_cross_population_comparator":{"label":"noncertifying_not_an_independence_model_or_prediction_interval","T2":.99500625,"T3":.992518734375},"frozen_t1_edge_overlap":overlap_report,"derived_t3_sha256":derived,"access":access,"model_calls":{"initial":c2+c3,"repeat":d2+d3,"T4":0},"repeat_equal":repeat,"state_nonmutation":pure}
 prov={"purpose":facts["purpose"],"exploratory_only":True,"certification":False,"historical_s2_stop_unchanged":True,"competition_engineering_implication":"diagnostic_only","inputs":{k:{"path":str(p),"sha256":h} for k,(p,h) in INPUTS.items()},"access":access,"environment":{"python":sys.version,"torch":torch.__version__,"platform":platform.platform(),"cpu_fp32":True,"deterministic":torch.are_deterministic_algorithms_enabled()},"derived_t3_sha256":derived,"trace_sha256":{"T2":a2["trace_sha256"],"T3":a3["trace_sha256"]},"source_sha256":sha(__file__),"test_sha256":sha(Path("tests/test_s2_e1_t2_t3_exploratory.py")),"training":None,"model_calls":facts["model_calls"],"repeat_equal":repeat,"state_nonmutation":pure}
 Path(metrics_path).write_text(json.dumps(facts,sort_keys=True,indent=2)+"\n",encoding="utf-8",newline="\n");Path(provenance_path).write_text(json.dumps(prov,sort_keys=True,indent=2)+"\n",encoding="utf-8",newline="\n");return facts
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--metrics",default="metrics_s2_e1_t2_t3_exploratory.json");p.add_argument("--provenance",default="artifacts/s2_e1_t2_t3_exploratory_provenance.json");a=p.parse_args();print(json.dumps(run(a.metrics,a.provenance),indent=2))
