"""Deterministic S2 doubling-mod-prime dataset construction and independent G0 audit."""
from __future__ import annotations

import argparse, hashlib, inspect, json, math, random, subprocess, tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

VERSION = STAGE = "s2_doublemod_seen_v1"
SEED = 20260730
WIDTH = 4
RUNGS = (2, 4, 8, 16, 32, 64)
EXPECTED_MODULI = [8009, 8263, 8501, 8753, 9001, 9257, 9533, 9767]
EXPECTED_COUNTS = {"train": 16000, "test_state": 4000,
                   **{f"depth_t_{t}": 512 for t in RUNGS}}

def digits(v: int) -> list[int]:
    if not 0 <= v <= 9999: raise ValueError("requires four-digit unsigned value")
    return [(v // 10**p) % 10 for p in range(4)]

def decode(ds: Sequence[int]) -> int:
    if len(ds) != 4 or any(type(d) is not int or not 0 <= d <= 9 for d in ds):
        raise ValueError("canonical state is exactly four decimal digits")
    return sum(d * 10**p for p, d in enumerate(ds))

def transition(x: int, n: int) -> int: return (2*x) % n
def input_ids(x: int, n: int) -> list[int]:
    return [0, *[d+2 for d in digits(x)], 1, *[d+2 for d in digits(n)]]

def carry_mask(x: int) -> list[int]:
    carry = 0; out = []
    for d in digits(x):
        carry = int(2*d + carry >= 10); out.append(carry)
    return out

def _prime(n: int) -> bool:
    if n < 2 or n % 2 == 0: return n == 2
    return all(n % d for d in range(3, math.isqrt(n)+1, 2))

def multiplicative_order_2(n: int) -> int:
    if not _prime(n): raise ValueError("order requested for nonprime")
    v = 2 % n
    for k in range(1, n):
        if v == 1: return k
        v = (v * 2) % n
    raise AssertionError("order not found")

def derive_moduli() -> list[int]:
    result=[]
    for lo in range(8000, 10000, 250):
        result.append(next(n for n in range(lo | 1, lo+250, 2)
                           if _prime(n) and multiplicative_order_2(n) >= 512))
    if result != EXPECTED_MODULI: raise RuntimeError(f"modulus derivation drift: {result}")
    return result

def _sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def _git(root: Path, *args: str) -> str:
    try: return subprocess.check_output(["git", *args], cwd=root, text=True).strip() or "clean"
    except (OSError, subprocess.CalledProcessError): return "unavailable"

def _boundary(n: int, delta: int) -> list[int]:
    return [x for x in range(n) if abs(2*x-n) == delta]

def _record(split: str, n: int, x: int, t: int) -> dict[str, Any]:
    tr=[x]; masks=[]
    for _ in range(t): masks.append(carry_mask(tr[-1])); tr.append((2*tr[-1]) % n)
    return {"stage":STAGE,"split":split,"N":n,"x0":x,"T":t,"y":tr[-1],
            "trajectory":tr,"reduction":[int(2*z>=n) for z in tr[:-1]],
            "carry_masks":masks,"carry_counts":[sum(m) for m in masks],
            "input_ids":input_ids(x,n),"labels":digits(tr[-1]),"seed":SEED}

def _allocate(n: int, excluded: set[int], rng: random.Random) -> tuple[list[int],list[int]]:
    train_forced=set(_boundary(n,3)+[0,n-1]); test_forced=set(_boundary(n,1))
    assert not (train_forced|test_forced)&excluded
    train=[]; test=[]
    for reduction in (0,1):
        tf=[x for x in train_forced if int(2*x>=n)==reduction]
        vf=[x for x in test_forced if int(2*x>=n)==reduction]
        pool=[x for x in range(n) if x not in excluded|train_forced|test_forced and int(2*x>=n)==reduction]
        rng.shuffle(pool)
        train += tf + pool[:1000-len(tf)]
        test += vf + pool[1000-len(tf):1000-len(tf)+250-len(vf)]
    return train,test

def generate(output_dir: str|Path, seed: int=SEED) -> dict[str,Any]:
    if seed != SEED: raise ValueError("frozen stage accepts only seed 20260730")
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True); rng=random.Random(seed)
    moduli=derive_moduli(); roots=[]; used_by_n={}
    for n in moduli:
        reserved=set(_boundary(n,1)+_boundary(n,3)+[0,n-1]); candidates=list(range(1,n)); rng.shuffle(candidates)
        used=set(); selected=[]
        for root in candidates:
            tr=[root]
            for _ in range(64): tr.append((2*tr[-1])%n)
            edges=set(tr[:-1])
            if len(tr)==len(set(tr)) and len(edges)==64 and not edges&(used|reserved):
                selected.append(root); used |= edges
                if len(selected)==64: break
        if len(selected)!=64: raise RuntimeError("depth root allocation failed")
        roots += [(n,r) for r in selected]; used_by_n[n]=used
    splits={k:[] for k in EXPECTED_COUNTS}
    for n in moduli:
        train,test=_allocate(n,used_by_n[n],rng)
        splits["train"] += [_record("train",n,x,1) for x in train]
        splits["test_state"] += [_record("test_state",n,x,1) for x in test]
    for t in RUNGS: splits[f"depth_t_{t}"]=[_record(f"depth_t_{t}",n,r,t) for n,r in roots]
    for rows in splits.values(): rng.shuffle(rows)
    source=Path(__file__).resolve(); repo=source.parents[1]
    config={"stage":STAGE,"seed":seed,"operation":"y=(2*x)%N","moduli":moduli,
      "orders":{str(n):multiplicative_order_2(n) for n in moduli},"bins":"[8000,10000), 8 bins width 250",
      "width":4,"decimal_order":"LSD-first","input_layout":"[STATE,x[4],MODULUS,N[4]]",
      "model_visible_fields":["x0","N"],"split_counts":{k:len(v) for k,v in splits.items()},
      "depth_roots_per_modulus":64,"depth_rungs":list(RUNGS),"boundary_policy":"abs(2x-N)=3 train; =1 test",
      "singleton_policy":"x=0 and x=N-1 train-only; both excluded from depth; no test x=0",
      "generator_path":"data/s2_doublemod_seen_v1.py","generator_sha256":_sha(source),
      "repo_head":_git(repo,"rev-parse","HEAD"),"generator_git_status":_git(repo,"status","--short","--","data/s2_doublemod_seen_v1.py")}
    for name,rows in splits.items():
        (out/f"{name}.jsonl").write_text("".join(json.dumps(r,sort_keys=True,separators=(",",":"))+"\n" for r in rows),encoding="utf-8",newline="\n")
    (out/"dataset_config.json").write_text(json.dumps(config,sort_keys=True,indent=2)+"\n",encoding="utf-8",newline="\n")
    files={p.name:_sha(p) for p in sorted(out.iterdir()) if p.name!="artifact_manifest.json"}
    (out/"artifact_manifest.json").write_text(json.dumps({"algorithm":"sha256","excluded":["artifact_manifest.json"],"files":files},sort_keys=True,indent=2)+"\n",encoding="utf-8",newline="\n")
    return config

def _load(root:Path,name:str):
    return [json.loads(x) for x in (root/f"{name}.jsonl").read_text(encoding="utf-8").splitlines()]

def _stats(rows, field):
    ans=[]
    for p in range(4):
        c=Counter(digits(r[field])[p] for r in rows); z=len(rows)
        ans.append({"entropy_bits":-sum((v/z)*math.log2(v/z) for v in c.values()),"modal_frequency":max(c.values())/z})
    return ans

def audit(root_dir:str|Path, regenerate:bool=True)->dict[str,Any]:
    root=Path(root_dir); cfg=json.loads((root/"dataset_config.json").read_text(encoding="utf-8-sig")); man=json.loads((root/"artifact_manifest.json").read_text())
    # Independent trial division and repeated-multiplication audit.
    independently=[]; orders={}
    for lo in range(8000,10000,250):
        for n in range(lo|1,lo+250,2):
            prime=n>1 and all(n%d for d in range(3,math.isqrt(n)+1,2))
            if prime:
                v=1; order=0
                while True:
                    v=(v+v)%n; order+=1
                    if v==1: break
                if order>=512: independently.append(n); orders[str(n)]=order; break
    assert independently==EXPECTED_MODULI==cfg["moduli"] and orders==cfg["orders"]
    assert cfg["generator_sha256"]==_sha(Path(__file__).resolve())
    for name,h in man["files"].items(): assert _sha(root/name)==h
    splits={k:_load(root,k) for k in EXPECTED_COUNTS}; assert {k:len(v) for k,v in splits.items()}==EXPECTED_COUNTS
    for name,rows in splits.items():
        for r in rows:
            tr=[r["x0"]]; masks=[]
            for _ in range(r["T"]):
                ds=[(tr[-1]//10**p)%10 for p in range(4)]; carry=0; mask=[]
                for d in ds: carry=int(d*2+carry>9); mask.append(carry)
                masks.append(mask); tr.append((tr[-1]+tr[-1])%r["N"])
            assert 0<=r["x0"]<r["N"] and r["trajectory"]==tr and r["y"]==tr[-1]
            assert r["labels"]==digits(tr[-1]) and r["input_ids"]==input_ids(r["x0"],r["N"]) and r["carry_masks"]==masks
    train={(r["N"],r["x0"]) for r in splits["train"]}; test={(r["N"],r["x0"]) for r in splits["test_state"]}
    depth={(r["N"],x) for r in splits["depth_t_64"] for x in r["trajectory"][:-1]}
    assert len(train)==16000 and len(test)==4000 and len(depth)==32768 and not train&test and not depth&(train|test)
    balance={}; boundary={}
    for n in EXPECTED_MODULI:
        balance[str(n)]={s:{str(c):sum(r["N"]==n and r["reduction"][0]==c for r in splits[s]) for c in (0,1)} for s in ("train","test_state")}
        assert balance[str(n)]=={"train":{"0":1000,"1":1000},"test_state":{"0":250,"1":250}}
        assert (n,0) in train and (n,n-1) in train and (n,0) not in test|depth
        boundary[str(n)]={"train_abs3":_boundary(n,3),"test_abs1":_boundary(n,1),"x_N_minus_1":"train_reduction"}
        assert all((n,x) in train for x in _boundary(n,3)) and all((n,x) in test for x in _boundary(n,1))
    full={(r["N"],r["x0"]):r for r in splits["depth_t_64"]}; depth_step={}
    for n in EXPECTED_MODULI:
        rows=[r for r in splits["depth_t_64"] if r["N"]==n]; assert len(rows)==64
        assert all(len(set(r["trajectory"]))==65 for r in rows)
        per=[{str(c):sum(r["reduction"][i]==c for r in rows) for c in (0,1)} for i in range(64)]
        assert sum(x["0"] for x in per)>0 and sum(x["1"] for x in per)>0; depth_step[str(n)]=per
    for t in RUNGS[:-1]:
        assert all(r["trajectory"]==full[(r["N"],r["x0"])]["trajectory"][:t+1] for r in splits[f"depth_t_{t}"])
    regen=True
    if regenerate:
        with tempfile.TemporaryDirectory() as td:
            generate(td); regen=sorted(p.name for p in root.iterdir())==sorted(p.name for p in Path(td).iterdir()) and all((root/p.name).read_bytes()==p.read_bytes() for p in Path(td).iterdir())
        assert regen
    distributions={name:{f:_stats(rows,f) for f in ("x0","y")} for name,rows in splits.items()}
    diagnostics={}
    for name,rows in splits.items():
        diagnostics[name]={
          "carry_mask_counts":dict(sorted(Counter("".join(map(str,m)) for r in rows for m in r["carry_masks"]).items())),
          "carry_count_counts":dict(sorted(Counter(c for r in rows for c in r["carry_counts"]).items())),
          "reduction_counts":dict(sorted(Counter(c for r in rows for c in r["reduction"]).items())),
          "leading_zero_output_frequency":sum(r["labels"][3]==0 for r in rows)/len(rows)}
    per_modulus_step={}
    for name,rows in splits.items():
        per_modulus_step[name]={}
        for n in EXPECTED_MODULI:
            selected=[r for r in rows if r["N"]==n]
            per_modulus_step[name][str(n)]=[
              {"reduction":dict(Counter(r["reduction"][i] for r in selected)),
               "carry_masks":dict(Counter("".join(map(str,r["carry_masks"][i])) for r in selected)),
               "carry_counts":dict(Counter(r["carry_counts"][i] for r in selected)),
               "leading_zero_output_frequency":sum(digits(r["trajectory"][i+1])[3]==0 for r in selected)/len(selected)}
              for i in range(selected[0]["T"])]
    conditional={name:{str(n):max(Counter(r["y"] for r in rows if r["N"]==n).values())/sum(r["N"]==n for r in rows) for n in EXPECTED_MODULI} for name,rows in splits.items()}
    return {"passed":True,"config_sha256":_sha(root/"dataset_config.json"),"artifact_hashes":man["files"],"split_counts":EXPECTED_COUNTS,"orders":orders,"balance":balance,"boundary_allocation":boundary,"unique_depth_edges":len(depth),"nested":True,"depth_reduction_per_step":depth_step,"diagnostics":diagnostics,"per_modulus_step_diagnostics":per_modulus_step,"position_stats":distributions,"conditional_N_exact_marginal_baseline":conditional,"byte_identical_regeneration":regen}

class StrictAPIContract:
    def __init__(self, model:Any):
        if tuple(inspect.signature(model.step.forward).parameters)!=("state","context"): raise TypeError("step must be exactly (state, context)")
        if not callable(model.encode_state) or not callable(model.encode_context): raise TypeError("separate encoders required")
        self.model=model; self.identity=id(model.step)
    def canonical_reset(self,predicted_digits:Sequence[int]):
        if id(self.model.step)!=self.identity: raise AssertionError("step identity changed")
        return self.model.encode_state(digits(decode(predicted_digits)))

def cli():
    p=argparse.ArgumentParser(); p.add_argument("action",choices=("generate","audit")); p.add_argument("output"); a=p.parse_args()
    print(json.dumps(generate(a.output) if a.action=="generate" else audit(a.output),indent=2,sort_keys=True))
if __name__=="__main__": cli()
