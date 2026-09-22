"""Authorized S2-D1 dense density control; training runs only via explicit CLI/run."""
from __future__ import annotations

import argparse, hashlib, inspect, json, random, time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from benchmark.s2_c1 import Harness, Projected, S2Transformer

SOURCE_SHA256 = "283a1033e78f1e241554e495d366d30aa2033278ab08d6726a512072fb164087"
PARENT_CHECKPOINT_SHA256 = "78114f47b9516862cca5ae6191dad06318d3a81d1b52aab3d27f2b0d33c18ab7"
PARENT_METRICS_SHA256 = "34c47580000dfcfceb976dad68df46dbcef0e7c192942fcfecc6214cb84893c8"
PARENT_PROVENANCE_SHA256 = "08b2e093700c5ce8dc9d05b49f1e802d3a270c8d1789be4a12c86d310871744d"
TARGET_STEPS, TARGET_EXAMPLES = 5040, 1_281_212

@dataclass(frozen=True)
class Config:
    control: str = "S2-D1 dense_density_control seed78"
    seed: int = 78
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    optimizer_steps: int = TARGET_STEPS
    full_epochs: int = 37
    partial_epoch_batches: int = 45
    max_seconds: float = 1800.0
    device: str = "cpu"
    dtype: str = "torch.float32"
CONFIG = Config()

def sha256(path: str | Path) -> str: return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def verify_architecture_source() -> None:
    source = Path(inspect.getsourcefile(S2Transformer) or "")
    if source.name != "s2_c1.py" or sha256(source) != SOURCE_SHA256:
        raise RuntimeError("imported S2 C1 architecture source drift")

def initialized_model() -> S2Transformer:
    verify_architecture_source(); torch.manual_seed(CONFIG.seed); dense = S2Transformer()
    torch.manual_seed(CONFIG.seed); sparse = S2Transformer()
    if not all(torch.equal(dense.state_dict()[n], sparse.state_dict()[n]) for n in dense.state_dict()):
        raise RuntimeError("dense/sparse seed-78 initialization mismatch")
    if sum(p.numel() for p in dense.parameters()) != 796426: raise RuntimeError("parameter-count drift")
    return dense

def schedule(batch_sizes: list[int]) -> tuple[int, int]:
    if len(batch_sizes) != 135 or batch_sizes[:-1] != [256]*134 or batch_sizes[-1] != 12:
        raise RuntimeError("dense loader geometry drift")
    return 37*len(batch_sizes)+45, 37*sum(batch_sizes)+sum(batch_sizes[:45])

def load_split(root: Path, name: str, access: list[str]) -> list[dict[str, Any]]:
    if name not in {"train", "test_state"}: raise ValueError("D1 forbids depth parse/inference")
    access.append(f"{name}.jsonl")
    with (root/f"{name}.jsonl").open(encoding="utf-8") as handle: return [json.loads(line) for line in handle]

def project_train(root: Path, access: list[str]) -> Projected:
    rs = load_split(root, "train", access)
    return Projected([r["input_ids"] for r in rs], [r["labels"] for r in rs])

def summarize(rows, logits):
    labels=torch.tensor([r["labels"] for r in rows]); pred=logits.argmax(-1); losses=F.cross_entropy(logits.flatten(0,1),labels.flatten(),reduction="none").view(-1,4)
    correct=pred.eq(labels); exact=correct.all(1)
    def basic(ix):
        return {"count":len(ix),"exact_correct":int(exact[ix].sum()),"exact_accuracy":float(exact[ix].double().mean()),
                "per_place":[{"place":p,"ce":float(losses[ix,p].double().mean()),"correct":int(correct[ix,p].sum()),"accuracy":float(correct[ix,p].double().mean())} for p in range(4)]}
    def grouped(key):
        return {str(v):basic([i for i,r in enumerate(rows) if key(r)==v]) for v in sorted({key(r) for r in rows},key=str)}
    return {"overall":basic(list(range(len(rows)))),"by_N":grouped(lambda r:r["N"]),"by_reduction":grouped(lambda r:r["reduction"][0]),
            "by_carry_count":grouped(lambda r:r["carry_counts"][0]),"by_carry_mask":grouped(lambda r:"".join(map(str,r["carry_masks"][0]))),
            "leading_zero":basic([i for i,r in enumerate(rows) if r["labels"][3]==0])}, pred, losses

@torch.no_grad()
def infer(model, rows):
    return torch.cat([model(torch.tensor([r["input_ids"] for r in rows[i:i+1024]])) for i in range(0,len(rows),1024)])

def run(data_root: str|Path, metrics_path: str|Path, checkpoint_path: str|Path, provenance_path: str|Path) -> dict[str,Any]:
    """Execute the single authorized run. No retry, selection, scheduler, or depth access."""
    root=Path(data_root); man=json.loads((root/"artifact_manifest.json").read_text(encoding="utf-8"))
    if {n:sha256(root/n) for n in man["files"]} != man["files"]: raise RuntimeError("dataset manifest mismatch")
    random.seed(78); torch.use_deterministic_algorithms(True); model=initialized_model().cpu().float(); optimization_access=[]
    ds=project_train(root,optimization_access); generator=torch.Generator().manual_seed(78)
    loader=DataLoader(ds,batch_size=256,shuffle=True,generator=generator,drop_last=False)
    sizes=[len(batch["labels"]) for batch in DataLoader(ds,batch_size=256,shuffle=False,drop_last=False)]
    if schedule(sizes)!=(TARGET_STEPS,TARGET_EXAMPLES): raise RuntimeError("schedule drift")
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4); start=time.perf_counter(); steps=seen=0; curve=[]
    for epoch in range(1,39):
        epoch_loss=0.; batches=loader if epoch<=37 else iter(loader)
        for index,batch in enumerate(batches):
            if epoch==38 and index==45: break
            if time.perf_counter()-start>=1800: raise TimeoutError("30 minute hard cap; no retry")
            optimizer.zero_grad(set_to_none=True); logits=model(batch["input_ids"]); loss=F.cross_entropy(logits.flatten(0,1),batch["labels"].flatten())
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step(); steps+=1; seen+=len(batch["labels"]); epoch_loss+=loss.item()*len(batch["labels"])
        curve.append({"epoch":epoch,"optimizer_steps":steps,"examples_seen":seen,"mean_ce":epoch_loss/(len(ds) if epoch<=37 else sum(sizes[:45]))})
    if (steps,seen)!=(TARGET_STEPS,TARGET_EXAMPLES): raise RuntimeError("execution schedule drift")
    model.eval().requires_grad_(False); Path(checkpoint_path).parent.mkdir(parents=True,exist_ok=True)
    torch.save({"state_dict":model.state_dict(),"config":asdict(CONFIG),"parameter_count":796426,"optimizer_steps":steps,"examples_seen":seen},checkpoint_path)
    payload=torch.load(checkpoint_path,map_location="cpu",weights_only=True)
    if payload["config"]!=asdict(CONFIG) or payload["parameter_count"]!=796426 or payload["optimizer_steps"]!=TARGET_STEPS or payload["examples_seen"]!=TARGET_EXAMPLES:
        raise RuntimeError("checkpoint metadata drift")
    restored=S2Transformer(); restored.load_state_dict(payload["state_dict"],strict=True); restored.eval().requires_grad_(False)
    post=[]; train=load_split(root,"train",post); test=load_split(root,"test_state",post); tl=infer(restored,train); vl=infer(restored,test)
    tr,tp,tce=summarize(train,tl); te,vp,vce=summarize(test,vl)
    def selected(rs,pred,loss,d): return [{"index":i,"N":r["N"],"x":r["x0"],"target":r["labels"],"prediction":pred[i].tolist(),"per_place_ce":loss[i].tolist()} for i,r in enumerate(rs) if abs(2*r["x0"]-r["N"])==d]
    abs1,abs3=selected(test,vp,vce,1),selected(train,tp,tce,3)
    errors=[{"index":i,"N":r["N"],"x":r["x0"],"target":r["labels"],"prediction":vp[i].tolist()} for i,r in enumerate(test) if vp[i].tolist()!=r["labels"]]
    singleton=[{"N":r["N"],"x":r["x0"],"target":r["labels"],"prediction":tp[i].tolist()} for i,r in enumerate(train) if r["x0"] in (0,r["N"]-1)]
    sample=torch.tensor([r["input_ids"] for r in test[:128]]); before={n:t.clone() for n,t in restored.state_dict().items()}; a=restored(sample); b=Harness(restored).run(sample); c=restored(sample)
    purity={"direct_harness":torch.equal(a,b),"repeat":torch.equal(a,c),"nonmutation":all(torch.equal(t,restored.state_dict()[n]) for n,t in before.items())}
    gates={"train_exact":tr["overall"]["exact_correct"]==34316,"test_errors_at_most_4":len(errors)<=4,"each_N_at_most_1":all(v["count"]-v["exact_correct"]<=1 for v in te["by_N"].values()),
           "nonreduction_at_most_2":te["by_reduction"]["0"]["count"]-te["by_reduction"]["0"]["exact_correct"]<=2,"reduction_at_most_2":te["by_reduction"]["1"]["count"]-te["by_reduction"]["1"]["exact_correct"]<=2,
           "test_threshold_abs1_16_exact":len(abs1)==16 and all(x["target"]==x["prediction"] for x in abs1),
           "train_threshold_abs3_16_exact":len(abs3)==16 and all(x["target"]==x["prediction"] for x in abs3),
           "imported_architecture_and_initialization":True,"checkpoint_load":True,
           "access":optimization_access==["train.jsonl"] and post==["train.jsonl","test_state.jsonl"],"purity":all(purity.values())}
    final={"kind":"final","config":asdict(CONFIG),"parameter_count":796426,"optimizer_steps":steps,"examples_seen":seen,"curve":curve,"evaluations":{"train":tr,"test_state":te},"threshold":{"all_abs1":abs1,"all_abs3":abs3},"singleton_predictions":singleton,"test_errors":{"total":len(errors),"rows":errors},"gates":gates,"g1_pass":all(gates.values()),"purity":purity,"checkpoint_sha256":sha256(checkpoint_path)}
    repo=Path(__file__).parents[1]; dataset_source=repo/"data/s2_doublemod_seen_dense_v1.py"; dataset_config=root/"dataset_config.json"; dataset_manifest=root/"artifact_manifest.json"
    provenance={"runner_sha256":sha256(__file__),"runner_test_sha256":sha256(repo/"tests/test_s2_d1.py"),"architecture_import":"benchmark.s2_c1.S2Transformer","architecture_source_sha256":SOURCE_SHA256,"initialized_named_tensors_bitwise_equal_to_fresh_sparse_seed78":True,
      "parameter_count":796426,"optimizer_steps":steps,"examples_seen":seen,"sparse_examples_seen":1_280_000,"example_count_delta_percent":(seen/1_280_000-1)*100,
      "dataset_source_sha256":sha256(dataset_source),"dataset_config_sha256":sha256(dataset_config),"dataset_manifest_sha256":sha256(dataset_manifest),"dataset_hashes":man["files"],
      "parent_artifacts":{"checkpoint_sha256":PARENT_CHECKPOINT_SHA256,"metrics_sha256":PARENT_METRICS_SHA256,"provenance_sha256":PARENT_PROVENANCE_SHA256,"sparse_c1_g1_pass":False,"sparse_c1_test_errors":11},"optimization_parsed_jsonl":optimization_access,"post_freeze_parsed_jsonl":post,"depth_parsed_jsonl":[],
      "dataset_diff":"dense train exact complement of test and T64 source edges; test/depth byte-copied; most non-threshold test states now lie between dense train neighbors, while threshold outputs remain uniquely held out","model_diff":"none; exact imported class","config_diff":"same seed78/batch256/AdamW/clip/fp32 CPU and fixed 5040 steps; dense contents/order and epochs-per-example differ; 37 full epochs plus 45 batches of epoch38","claim_scope":"density/capacity control, not sparse-learning success","device":"cpu","dtype":"torch.float32","no_retry_selection_scheduler":True,"weights_only_strict_checkpoint_load":True,"checkpoint_sha256":sha256(checkpoint_path)}
    Path(metrics_path).write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in curve+[final]),encoding="utf-8",newline="\n"); Path(provenance_path).write_text(json.dumps(provenance,sort_keys=True,indent=2)+"\n",encoding="utf-8",newline="\n")
    return final

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--data-root",default="data/generated/s2_doublemod_seen_dense_v1"); p.add_argument("--metrics",default="metrics_s2_d1_dense_density_control_seed78.jsonl"); p.add_argument("--checkpoint",default="artifacts/s2_d1_dense_density_control_seed78.pt"); p.add_argument("--provenance",default="artifacts/s2_d1_dense_density_control_seed78_provenance.json"); a=p.parse_args(); print(json.dumps(run(a.data_root,a.metrics,a.checkpoint,a.provenance),indent=2))
