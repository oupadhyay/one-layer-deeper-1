"""Authorized S2 C1/G1 seed-78 one-step Transformer control."""
from __future__ import annotations
import argparse, hashlib, inspect, json, random, time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence
from unittest.mock import patch
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

@dataclass(frozen=True)
class Config:
    control:str="s2_c1_transformer_v1"; seed:int=78; width:int=128; heads:int=4
    blocks:int=4; ff:int=512; batch_size:int=256; epochs:int=80
    learning_rate:float=1e-3; weight_decay:float=1e-4; gradient_clip:float=1.0
    eval_batch_size:int=1024; max_seconds:float=1800.0
CONFIG=Config()

class Projected(Dataset):
    def __init__(self,ids,labels): self.input_ids=torch.tensor(ids); self.labels=torch.tensor(labels)
    def __len__(self): return len(self.labels)
    def __getitem__(self,i): return {"input_ids":self.input_ids[i],"labels":self.labels[i]}

class Block(nn.Module):
    def __init__(self):
        super().__init__(); self.ln1=nn.LayerNorm(128); self.attn=nn.MultiheadAttention(128,4,dropout=0,batch_first=True)
        self.ln2=nn.LayerNorm(128); self.ff=nn.Sequential(nn.Linear(128,512),nn.GELU(),nn.Linear(512,128))
    def forward(self,x):
        z=self.ln1(x); x=x+self.attn(z,z,z,need_weights=False)[0]; return x+self.ff(self.ln2(x))

class OuterStep(nn.Module):
    def __init__(self): super().__init__(); self.blocks=nn.ModuleList(Block() for _ in range(4))
    def forward(self,state,context):
        x=torch.cat((state,context),1)
        for block in self.blocks: x=block(x)
        return x[:,:4]

class S2Transformer(nn.Module):
    def __init__(self):
        super().__init__(); self.embedding=nn.Embedding(10,128); self.role_embedding=nn.Embedding(2,128)
        self.place_embedding=nn.Embedding(4,128); self.step=OuterStep(); self.readout=nn.Linear(128,10)
    def _encode(self,d,role):
        if d.ndim!=2 or d.shape[1]!=4: raise ValueError("canonical digits must be Bx4")
        p=torch.arange(4,device=d.device); return self.embedding(d)+self.role_embedding.weight[role]+self.place_embedding(p)
    def encode_state(self,d): return self._encode(d,0)
    def encode_context(self,d): return self._encode(d,1)
    def canonical(self,ids):
        if ids.ndim!=2 or ids.shape[1]!=10 or not torch.all(ids[:,0]==0) or not torch.all(ids[:,5]==1): raise ValueError("layout marker")
        a,b=ids[:,1:5],ids[:,6:10]
        if not torch.all((a>=2)&(a<=11)) or not torch.all((b>=2)&(b<=11)): raise ValueError("digit token")
        return a-2,b-2
    def forward(self,ids):
        x,n=self.canonical(ids); return self.readout(self.step(self.encode_state(x),self.encode_context(n)))

class Harness:
    def __init__(self,m):
        if tuple(inspect.signature(m.step.forward).parameters)!=("state","context"): raise TypeError("signature")
        self.model=m; self.identity=id(m.step)
    def run(self,ids):
        if id(self.model.step)!=self.identity: raise RuntimeError("identity")
        x,n=self.model.canonical(ids); return self.model.readout(self.model.step(self.model.encode_state(x),self.model.encode_context(n)))

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify(root):
    root=Path(root); man=json.loads((root/"artifact_manifest.json").read_text())
    hashes={n:sha(root/n) for n in man["files"]}
    if hashes!=man["files"]: raise RuntimeError("manifest mismatch")
    return hashes
def load_projected(root,opened):
    p=Path(root)/"train.jsonl"; opened.append(p.name); ids=[]; labels=[]
    with p.open() as f:
        for line in f:
            r=json.loads(line); ids.append(r["input_ids"]); labels.append(r["labels"])
    return Projected(ids,labels)
def load_rows(root,name,opened):
    if name not in ("train","test_state"): raise ValueError("one-step only")
    p=Path(root)/f"{name}.jsonl"; opened.append(p.name)
    with p.open() as f: return [json.loads(x) for x in f]

@torch.no_grad()
def predict(m,rows,dev):
    out=[]
    for i in range(0,len(rows),CONFIG.eval_batch_size):
        ids=torch.tensor([r["input_ids"] for r in rows[i:i+CONFIG.eval_batch_size]],device=dev); out+=m(ids).argmax(-1).cpu().tolist()
    return out
def metrics(rows,preds,logits=None):
    exact=[p==r["labels"] for p,r in zip(preds,rows)]
    groups=lambda key:{str(v):{"count":sum(key(r)==v for r in rows),"exact_correct":sum(ok for r,ok in zip(rows,exact) if key(r)==v)} for v in sorted({key(r) for r in rows})}
    return {"count":len(rows),"exact_correct":sum(exact),"by_N":groups(lambda r:r["N"]),"by_reduction":groups(lambda r:r["reduction"][0]),
      "by_carry_count":groups(lambda r:r["carry_counts"][0]),"by_carry_mask":groups(lambda r:"".join(map(str,r["carry_masks"][0]))),
      "leading_zero":{"count":sum(r["labels"][3]==0 for r in rows),"exact_correct":sum(ok for r,ok in zip(rows,exact) if r["labels"][3]==0)}}

def run(data_root,metrics_path,checkpoint_path,provenance_path,device="cpu"):
    random.seed(78); torch.manual_seed(78); torch.use_deterministic_algorithms(True); dev=torch.device(device)
    hashes=verify(data_root); opt=[]; ds=load_projected(data_root,opt)
    model=S2Transformer().to(dev); count=sum(p.numel() for p in model.parameters()); optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    gen=torch.Generator().manual_seed(78); loader=DataLoader(ds,batch_size=256,shuffle=True,generator=gen); start=time.perf_counter(); steps=0; curve=[]
    for epoch in range(1,81):
        total=0.
        for b in loader:
            if time.perf_counter()-start>=1800: raise TimeoutError("30 minute hard cap")
            optimizer.zero_grad(set_to_none=True); logits=model(b["input_ids"].to(dev)); loss=F.cross_entropy(logits.flatten(0,1),b["labels"].to(dev).flatten())
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1); optimizer.step(); steps+=1; total+=loss.item()*len(b["labels"])
        curve.append({"epoch":epoch,"optimizer_steps":steps,"mean_ce":total/len(ds)})
    if steps!=5040: raise RuntimeError("step drift")
    model.eval(); [p.requires_grad_(False) for p in model.parameters()]; Path(checkpoint_path).parent.mkdir(parents=True,exist_ok=True)
    torch.save({"state_dict":model.state_dict(),"config":asdict(CONFIG),"parameter_count":count,"optimizer_steps":steps},checkpoint_path)
    frozen=S2Transformer(); frozen.load_state_dict(torch.load(checkpoint_path,map_location="cpu",weights_only=True)["state_dict"],strict=True)
    opened=[]; train=load_rows(data_root,"train",opened); test=load_rows(data_root,"test_state",opened); pt=predict(model,train,dev); pv=predict(model,test,dev)
    ev={"train":metrics(train,pt),"test_state":metrics(test,pv)}; test_errors=len(test)-ev["test_state"]["exact_correct"]
    threshold=[(r,p) for r,p in zip(test,pv) if abs(2*r["x0"]-r["N"])==1]+[(r,p) for r,p in zip(train,pt) if abs(2*r["x0"]-r["N"])==3]
    sample=torch.tensor([r["input_ids"] for r in test[:128]],device=dev); before={n:v.clone() for n,v in model.state_dict().items()}; a=model(sample); b=Harness(model).run(sample); c=model(sample)
    purity={"direct_harness":torch.equal(a,b),"repeated":torch.equal(a,c),"nonmutation":all(torch.equal(v,model.state_dict()[n]) for n,v in before.items())}
    gates={"train_exact":ev["train"]["exact_correct"]==16000,"test_errors_at_most_4":test_errors<=4,
      "each_N_at_most_1":all(v["count"]-v["exact_correct"]<=1 for v in ev["test_state"]["by_N"].values()),
      "nonreduction_at_most_2":ev["test_state"]["by_reduction"]["0"]["count"]-ev["test_state"]["by_reduction"]["0"]["exact_correct"]<=2,
      "reduction_at_most_2":ev["test_state"]["by_reduction"]["1"]["count"]-ev["test_state"]["by_reduction"]["1"]["exact_correct"]<=2,
      "threshold_16_exact":len(threshold)==16 and all(p==r["labels"] for r,p in threshold),"access":opt==["train.jsonl"] and opened==["train.jsonl","test_state.jsonl"],"purity":all(purity.values())}
    errors=[{"index":i,"x":r["x0"],"N":r["N"],"target":r["labels"],"prediction":p} for i,(r,p) in enumerate(zip(test,pv)) if p!=r["labels"]][:32]
    singleton=[{"N":r["N"],"x":r["x0"],"prediction":p,"target":r["labels"]} for r,p in zip(train,pt) if r["x0"] in (0,r["N"]-1)]
    final={"kind":"final","control":CONFIG.control,"config":asdict(CONFIG),"parameter_count":count,"optimizer_steps":steps,"elapsed_seconds":time.perf_counter()-start,
      "curve":curve,"evaluations":ev,"threshold_rows":len(threshold),"test_errors":errors,"blackout_highlights":errors[:8],"singleton_predictions":singleton,"gates":gates,"g1_pass":all(gates.values()),"purity":purity,"checkpoint_sha256":sha(checkpoint_path)}
    provenance={"config":asdict(CONFIG),"parameter_count":count,"dataset_hashes":hashes,"optimization_parsed_jsonl":opt,"post_freeze_parsed_jsonl":opened,"depth_parsed_jsonl":[],"runner_sha256":sha(__file__),"test_sha256":sha(Path(__file__).parents[1]/"tests/test_s2_c1.py"),"checkpoint_sha256":sha(checkpoint_path),"source_statuses":{"runner":"uncommitted authorized","test":"uncommitted authorized"},"device":str(dev),"dtype":"torch.float32","outer_step_applications":1}
    Path(metrics_path).write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in curve+[final]),encoding="utf-8",newline="\n"); Path(provenance_path).write_text(json.dumps(provenance,sort_keys=True,indent=2)+"\n",encoding="utf-8",newline="\n")
    return final

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--data-root",default="data/generated/s2_doublemod_seen_v1"); p.add_argument("--metrics",default="metrics_s2_c1_transformer_seed78.jsonl"); p.add_argument("--checkpoint",default="artifacts/s2_c1_transformer_seed78.pt"); p.add_argument("--provenance",default="artifacts/s2_c1_transformer_seed78_provenance.json"); a=p.parse_args(); print(json.dumps(run(a.data_root,a.metrics,a.checkpoint,a.provenance),indent=2))
