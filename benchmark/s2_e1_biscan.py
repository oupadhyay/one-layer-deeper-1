"""Predeclared S2-E1 bidirectional scan v1 (seed 79).

Importing this module is inert.  The sole optimization is available only through
``run`` or the command-line entry point.
"""
from __future__ import annotations

import argparse, hashlib, inspect, json, os, random, time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

SEED = 79
PARAMETER_COUNT = 63_178
TARGET_STEPS, TARGET_EXAMPLES = 5_040, 1_281_212
DATASET_PINS = {
    "artifact_manifest.json": "817e2df936909427934fdbe8f59ea1d596649c3e14eb2facedce6c97c64bf578",
    "dataset_config.json": "10c26a868b02403b68f7f0348771b52866fcd1de8fb1978a6ce2f412327f2a98",
    "train.jsonl": "4f1db72713873ba564f3b21faefc8f132d0f201b267b6870a3452c8ee3c1441c",
    "test_state.jsonl": "4accb941940d5ff9d513b3d887c20a589e9c6c348b9833db64129b01fc65fcb2",
}
DENSE_G0_SHA256 = "0ed5dee741f528ea6ab9db61e566dcc91a5e576c7004e98d8f7438baa2cf2a9b"

@dataclass(frozen=True)
class Config:
    control: str = "S2-E1 bidirectional_scan_v1"
    seed: int = SEED
    batch_size: int = 256
    learning_rate: float = 1e-3
    betas: tuple[float, float] = (.9, .999)
    eps: float = 1e-8
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    optimizer_steps: int = TARGET_STEPS
    full_epochs: int = 37
    partial_epoch_batches: int = 45
    max_seconds: float = 1500.0
    device: str = "cpu"
    dtype: str = "torch.float32"
CONFIG = Config()

class Projected(Dataset):
    """The complete and only optimization-time projection."""
    def __init__(self, input_ids, labels):
        self.input_ids = torch.tensor(input_ids, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self): return len(self.labels)
    def __getitem__(self, i): return {"input_ids": self.input_ids[i], "labels": self.labels[i]}

class BidirectionalStep(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_cell = nn.GRUCell(64, 64)
        self.up_cell = nn.GRUCell(128, 64)
    def forward(self, state, context):
        pair = torch.cat((state, context), dim=-1)
        hidden = pair.new_zeros((pair.shape[0], 64)); down = [None] * 4
        for place in (3, 2, 1, 0):
            hidden = self.down_cell(pair[:, place], hidden); down[place] = hidden
        hidden = pair.new_zeros((pair.shape[0], 64)); output = []
        for place in (0, 1, 2, 3):
            hidden = self.up_cell(torch.cat((pair[:, place], down[place]), -1), hidden)
            output.append(hidden)
        return torch.stack(output, dim=1)

class S2BiScan(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(10, 32)
        self.step = BidirectionalStep()
        self.readout = nn.Linear(64, 10)
    def encode_state(self, digits): return self.embedding(digits)
    def encode_context(self, digits): return self.embedding(digits)
    def canonical(self, ids):
        if ids.ndim != 2 or ids.shape[1] != 10: raise ValueError("canonical input must be Bx10")
        if not torch.all(ids[:, 0] == 0) or not torch.all(ids[:, 5] == 1): raise ValueError("layout marker")
        state, context = ids[:, 1:5], ids[:, 6:10]
        if not torch.all((state >= 2) & (state <= 11)) or not torch.all((context >= 2) & (context <= 11)):
            raise ValueError("digit token")
        return state - 2, context - 2
    def forward(self, ids):
        state, context = self.canonical(ids)
        return self.readout(self.step(self.encode_state(state), self.encode_context(context)))

class Harness:
    def __init__(self, model):
        if tuple(inspect.signature(model.step.forward).parameters) != ("state", "context"): raise TypeError("step signature")
        self.model, self.identity = model, id(model.step)
    def run(self, ids):
        if id(self.model.step) != self.identity: raise RuntimeError("step replaced")
        state, context = self.model.canonical(ids)
        return self.model.readout(self.model.step(self.model.encode_state(state), self.model.encode_context(context)))

def initialize(model: S2BiScan) -> S2BiScan:
    torch.manual_seed(SEED)
    nn.init.normal_(model.embedding.weight, 0., .02)
    for cell in (model.step.down_cell, model.step.up_cell):
        for block in cell.weight_ih.chunk(3, 0): nn.init.xavier_uniform_(block)
        for block in cell.weight_hh.chunk(3, 0): nn.init.orthogonal_(block)
        nn.init.zeros_(cell.bias_ih); nn.init.zeros_(cell.bias_hh)
    nn.init.xavier_uniform_(model.readout.weight); nn.init.zeros_(model.readout.bias)
    return model

def initialized_model():
    torch.manual_seed(SEED)
    model = initialize(S2BiScan()).cpu().float()
    if sum(p.numel() for p in model.parameters()) != PARAMETER_COUNT: raise RuntimeError("parameter count drift")
    return model

def canonical_state_hash(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode()); digest.update(b"\0"); digest.update(str(value.dtype).encode())
        digest.update(b"\0"); digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
        digest.update(b"\0"); digest.update(value.numpy().tobytes())
    return digest.hexdigest()

def sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def verify_files(root, log):
    root = Path(root); manifest_path = root / "artifact_manifest.json"
    if sha256(manifest_path) != DATASET_PINS["artifact_manifest.json"]: raise RuntimeError("manifest pin drift")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual = {}
    for name in manifest["files"]:
        log.append(name); actual[name] = sha256(root / name)  # opaque bytes only
    if actual != manifest["files"]: raise RuntimeError("manifest file drift")
    if any(sha256(root / n) != h for n, h in DATASET_PINS.items() if n != "artifact_manifest.json"): raise RuntimeError("dataset pin drift")
    return actual

def load_rows(root, name, log):
    if name not in ("train", "test_state"): raise ValueError("depth access forbidden")
    log.append(f"{name}.jsonl")
    with (Path(root) / f"{name}.jsonl").open(encoding="utf-8") as source: return [json.loads(line) for line in source]

def project_train(root, log):
    rows = load_rows(root, "train", log)
    return Projected([r["input_ids"] for r in rows], [r["labels"] for r in rows])

def schedule(batch_sizes):
    if batch_sizes != [256] * 134 + [12]: raise RuntimeError("loader geometry drift")
    return 37 * 135 + 45, 37 * sum(batch_sizes) + sum(batch_sizes[:45])

@torch.no_grad()
def infer(model, rows):
    return torch.cat([model(torch.tensor([r["input_ids"] for r in rows[i:i+1024]])) for i in range(0, len(rows), 1024)])

def _basic(rows, logits):
    labels=torch.tensor([r["labels"] for r in rows]); pred=logits.argmax(-1); loss=F.cross_entropy(logits.flatten(0,1),labels.flatten(),reduction="none").view(-1,4); correct=pred.eq(labels); exact=correct.all(1)
    def summary(ix):
        return {"count":len(ix),"exact_correct":int(exact[ix].sum()),"per_place":[{"ce":float(loss[ix,p].double().mean()),"correct":int(correct[ix,p].sum())} for p in range(4)]}
    def groups(key): return {str(v):summary([i for i,r in enumerate(rows) if key(r)==v]) for v in sorted({key(r) for r in rows},key=str)}
    report={"overall":summary(list(range(len(rows)))),"by_N":groups(lambda r:r["N"]),"by_reduction":groups(lambda r:r["reduction"][0]),"by_carry_count":groups(lambda r:r["carry_counts"][0]),"by_carry_mask":groups(lambda r:"".join(map(str,r["carry_masks"][0]))),"by_leading_zero":groups(lambda r:r["labels"][3]==0)}
    return report,pred,loss

def _selected(rows, logits, distance):
    report,pred,loss=_basic(rows,logits); top2=logits.topk(2,-1).values
    selected=[{"index":i,"N":r["N"],"x":r["x0"],"reduction":r["reduction"][0],"target":r["labels"],"prediction":pred[i].tolist(),"exact":pred[i].tolist()==r["labels"],"per_place_ce":loss[i].tolist(),"top2_margin":(top2[i,:,0]-top2[i,:,1]).tolist()} for i,r in enumerate(rows) if abs(2*r["x0"]-r["N"])==distance]
    return report,pred,loss,selected

def run(data_root, metrics_path, checkpoint_path, provenance_path, predeclaration_path="artifacts/s2_e1_biscan_seed79_predeclaration.json"):
    """Execute exactly one predeclared run; any failed condition raises immediately."""
    entry=time.perf_counter(); torch.set_num_threads(1); torch.set_num_interop_threads(1); random.seed(SEED); torch.manual_seed(SEED); torch.use_deterministic_algorithms(True)
    pre=json.loads(Path(predeclaration_path).read_text(encoding="utf-8")); verified=[]; hashes=verify_files(data_root,verified)
    model=initialized_model(); init_hash=canonical_state_hash(model)
    if pre["initial_state_sha256"] != init_hash: raise RuntimeError("initial hash not predeclared")
    opt_access=[]; dataset=project_train(data_root,opt_access)
    plain=DataLoader(dataset,batch_size=256,shuffle=False,num_workers=0,drop_last=False); sizes=[len(b["labels"]) for b in plain]
    if schedule(sizes)!=(TARGET_STEPS,TARGET_EXAMPLES): raise RuntimeError("schedule drift")
    kwargs=dict(lr=1e-3,betas=(.9,.999),eps=1e-8,weight_decay=1e-4); sig=inspect.signature(torch.optim.AdamW)
    if "foreach" in sig.parameters: kwargs["foreach"]=False
    if "fused" in sig.parameters: kwargs["fused"]=False
    optimizer=torch.optim.AdamW(model.parameters(),**kwargs); loader=DataLoader(dataset,batch_size=256,shuffle=True,generator=torch.Generator().manual_seed(SEED),num_workers=0,drop_last=False)
    curve=[]; steps=examples=0
    for epoch in range(1,39):
        ce_sum=grad_sum=count=0.
        for index,batch in enumerate(loader):
            if epoch==38 and index==45: break
            if time.perf_counter()-entry>=1500: raise TimeoutError("1500 second run-entry wall cap")
            optimizer.zero_grad(set_to_none=True); logits=model(batch["input_ids"]); loss=F.cross_entropy(logits.flatten(0,1),batch["labels"].flatten())
            loss.backward(); norm=nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step(); n=len(batch["labels"]); steps+=1; examples+=n; ce_sum+=float(loss)*n; grad_sum+=float(norm); count+=1
        curve.append({"epoch":epoch,"mean_ce":ce_sum/(sum(sizes) if epoch<=37 else sum(sizes[:45])),"optimizer_steps":steps,"examples_seen":examples,"mean_preclip_grad_norm":grad_sum/count,"elapsed_seconds":time.perf_counter()-entry})
    del optimizer
    if (steps,examples)!=(TARGET_STEPS,TARGET_EXAMPLES): raise RuntimeError("execution drift")
    model.eval().requires_grad_(False); Path(checkpoint_path).parent.mkdir(parents=True,exist_ok=True)
    metadata={"config":asdict(CONFIG),"parameter_count":PARAMETER_COUNT,"optimizer_steps":steps,"examples_seen":examples,"initial_state_sha256":init_hash}
    torch.save({"state_dict":model.state_dict(),**metadata},checkpoint_path)
    payload=torch.load(checkpoint_path,map_location="cpu",weights_only=True)
    if {k:payload[k] for k in metadata} != metadata: raise RuntimeError("checkpoint metadata drift")
    restored=S2BiScan(); restored.load_state_dict(payload["state_dict"],strict=True); restored.eval().requires_grad_(False)
    post=[]; train=load_rows(data_root,"train",post); test=load_rows(data_root,"test_state",post); train_logits=infer(restored,train); test_logits=infer(restored,test)
    tr,tp,tl,abs3=_selected(train,train_logits,3); te,vp,vl,abs1=_selected(test,test_logits,1)
    sample=torch.tensor([r["input_ids"] for r in test[:128]]); before={n:t.clone() for n,t in restored.state_dict().items()}; direct=restored(sample); harness=Harness(restored).run(sample); repeat=restored(sample)
    purity={"direct_harness":torch.equal(direct,harness),"repeat":torch.equal(direct,repeat),"nonmutation":all(torch.equal(t,restored.state_dict()[n]) for n,t in before.items()),"shape":list(direct.shape)==[128,4,10]}
    errors=[{"index":i,"N":r["N"],"x":r["x0"],"target":r["labels"],"prediction":vp[i].tolist(),"per_place_ce":vl[i].tolist()} for i,r in enumerate(test) if vp[i].tolist()!=r["labels"]]
    gates={"train_34316":tr["overall"]["exact_correct"]==34316,"test_4000":te["overall"]["exact_correct"]==4000,"each_N_500":all(x["count"]==x["exact_correct"]==500 for x in te["by_N"].values()),"reduction_2000":all(x["count"]==x["exact_correct"]==2000 for x in te["by_reduction"].values()),
           "abs1_16_exact":len(abs1)==16 and all(x["exact"] for x in abs1),
           "abs1_structure":len(abs1)==16 and sum(x["reduction"] for x in abs1)==8 and all(sum(x["N"]==n for x in abs1)==2 for n in (8009,8263,8501,8753,9001,9257,9533,9767)),
           "abs3_16_exact":len(abs3)==16 and all(x["exact"] for x in abs3),"integrity":sum(p.numel() for p in restored.parameters())==PARAMETER_COUNT and steps==TARGET_STEPS and examples==TARGET_EXAMPLES and time.perf_counter()-entry<1500,"access":opt_access==["train.jsonl"] and post==["train.jsonl","test_state.jsonl"],"depth_inference_zero":True,"purity":all(purity.values())}
    tensor_hash=lambda x:hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    e1_pass=all(gates.values())
    final={"kind":"final","config":asdict(CONFIG),"curve":curve,"optimizer_steps":steps,"examples_seen":examples,"elapsed_seconds":time.perf_counter()-entry,"evaluations":{"train":tr,"test_state":te},"threshold":{"all_abs1":abs1,"all_abs3":abs3},"test_errors":errors,"singleton_predictions":[{"N":r["N"],"x":r["x0"],"target":r["labels"],"prediction":tp[i].tolist()} for i,r in enumerate(train) if r["x0"] in (0,r["N"]-1)],"label_hashes":{"train":tensor_hash(torch.tensor([r["labels"] for r in train])),"test":tensor_hash(torch.tensor([r["labels"] for r in test]))},"prediction_hashes":{"train":tensor_hash(tp),"test":tensor_hash(vp)},"timings":{"total":time.perf_counter()-entry},"gates":gates,"pass_for_independent_review":e1_pass,"checkpoint_sha256":sha256(checkpoint_path)}
    repo=Path(__file__).parents[1]; provenance={"environment":{"python":os.sys.version,"torch":torch.__version__,"cpu_fp32":True,"intra_threads":torch.get_num_threads(),"interop_threads":torch.get_num_interop_threads()},"dataset_hashes":hashes,"dataset_pins":DATASET_PINS,"dense_g0_sha256":DENSE_G0_SHA256,"initial_state_sha256":init_hash,"runner_sha256":sha256(__file__),"test_sha256":sha256(repo/"tests/test_s2_e1_biscan.py"),"predeclaration_sha256":sha256(predeclaration_path),"source_statuses":{"runner":"predeclared greenfield","test":"predeclared greenfield"},"C1":{"checkpoint_sha256":"78114f47b9516862cca5ae6191dad06318d3a81d1b52aab3d27f2b0d33c18ab7","g1_pass":False},"D1":{"checkpoint_sha256":"b321c4d310d555abeb420cd3e3df9c41a9d3bc698c686b4eeb84ec27ddcea9b2","g1_pass":False},"optimization_parsed_jsonl":opt_access,"post_freeze_parsed_jsonl":post,"hash_verified_files":verified,"depth_parsed_jsonl":[],"depth_inference":0,"outer_step_applications_per_forward":1,"weights_only_strict_restore":True,"no_retry_scheduler_aux_selection_depth":True,"checkpoint_sha256":sha256(checkpoint_path),"pass_for_independent_review":e1_pass}
    Path(metrics_path).write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in curve+[final]),encoding="utf-8",newline="\n"); Path(provenance_path).write_text(json.dumps(provenance,sort_keys=True,indent=2)+"\n",encoding="utf-8",newline="\n")
    return final

if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--data-root",default="data/generated/s2_doublemod_seen_dense_v1"); parser.add_argument("--metrics",default="metrics_s2_e1_biscan_seed79.jsonl"); parser.add_argument("--checkpoint",default="artifacts/s2_e1_biscan_seed79.pt"); parser.add_argument("--provenance",default="artifacts/s2_e1_biscan_seed79_provenance.json"); parser.add_argument("--predeclaration",default="artifacts/s2_e1_biscan_seed79_predeclaration.json"); args=parser.parse_args(); print(json.dumps(run(args.data_root,args.metrics,args.checkpoint,args.provenance,args.predeclaration),indent=2))
