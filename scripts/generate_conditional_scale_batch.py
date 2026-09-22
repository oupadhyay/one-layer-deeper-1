"""Generate the predeclared Medium and Hard conditional-transducer batch."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

MEDIUM_VARIANTS = (
    ("focal05_lr9", 64, 32, 256, "9e-4", 8, ".5"),
    ("plain_lr9", 64, 32, 256, "9e-4", 8, None),
    ("focal10_lr9", 64, 32, 256, "9e-4", 8, "1."),
    ("focal05_lr6", 64, 32, 256, "6e-4", 8, ".5"),
    ("focal05_batch128", 64, 32, 128, "9e-4", 8, ".5"),
    ("focal05_d96", 96, 48, 256, "9e-4", 8, ".5"),
    ("focal05_lr10", 64, 32, 256, "1e-3", 8, ".5"),
    ("focal05_warm32", 64, 32, 256, "9e-4", 32, ".5"),
    ("focal05_batch512", 64, 32, 512, "9e-4", 8, ".5"),
    ("focal05_d128", 128, 64, 256, "9e-4", 8, ".5"),
)


def source(name: str, d_model: int, hidden: int, batch: int, lr: str,
           warmup: int, focal: str | None, dynamic: bool = False) -> str:
    width = "None" if dynamic else "5"
    contract = (
        'if L>s.config.max_seq_len or bool(((nw<1)|(xw<1)|(t>64)).any()):raise ValueError("invalid Hard contract")'
        if dynamic else
        'if L>s.config.max_seq_len or bool(((nw<1)|(nw>W)|(xw<1)|(xw>W)|(t>64)).any()):raise ValueError("invalid Medium contract")'
    )
    slots = "L" if dynamic else "W"
    loss_import = ",TokenLossBatch" if focal is not None else ""
    loss = ""
    loss_argument = ""
    if focal is not None:
        loss = (
            "def token_training_loss(batch:TokenLossBatch):\n"
            " ce=F.cross_entropy(batch.logits.transpose(1,2),batch.labels,ignore_index=-100,reduction='none');"
            f"v=batch.valid_mask;return (ce*(1-torch.exp(-ce)).pow({focal})*v).sum()/v.sum()\n"
        )
        loss_argument = ",token_training_loss=token_training_loss"
    return f'''"""Conditional global transducer scale batch: {name}."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn
from benchmark import ModelSpec,OptimizerSpec,OptimizerBundle,Submission{loss_import},assert_model_state
W,D,H={width},{d_model},{hidden}
class C:
 def __init__(s,vocab_size,max_seq_len):s.vocab_size,s.max_seq_len=vocab_size,max_seq_len
class RMS(nn.Module):
 def __init__(s,n):super().__init__();s.weight=nn.Parameter(torch.ones(n))
 def forward(s,x):return x*(x.float().square().mean(-1,keepdim=True)+1e-6).rsqrt().to(x.dtype)*s.weight
class Transition(nn.Module):
 def __init__(s):
  super().__init__();s.digit=nn.Embedding(10,D);s.presence=nn.Embedding(2,D);s.scan=nn.GRU(D,H,batch_first=True,bidirectional=True);s.value=nn.Linear(D,D,bias=False);s.content=nn.Linear(2*D,D);s.film=nn.Linear(D,2*D);s.norm=RMS(D);s.out=nn.Linear(D,10)
 def encode(s,digit,present):
  state=s.digit(digit)+s.presence(present.long());return s.scan(state)[0]
 def forward(s,nstate,np,x,xp):
  state=x.to(s.digit.weight.dtype)@s.digit.weight;state=state+s.presence(xp.long());xs=s.scan(state)[0];weight=np[...,None].to(nstate.dtype);summary=(nstate*weight).sum(1)/weight.sum(1).clamp_min(1.);ctx=s.value(summary)[:,None,:].expand_as(xs);h=F.gelu(s.content(torch.cat((xs,ctx),-1)));scale,shift=s.film(ctx).chunk(2,-1);h=s.norm(h)*(1+torch.tanh(scale))+shift;return s.out(h),h
class Model(nn.Module):
 def __init__(s,spec):super().__init__();s.config=C(spec.vocab_size,spec.max_seq_len);s.transition=Transition()
 @staticmethod
 def parse(ids,mask):
  d=ids.ge(7)&ids.lt(17)&mask;m=ids.eq(2).long()+2*ids.eq(3).long()+3*ids.eq(4).long();r=torch.cummax(m,1).values*d.long();i=torch.arange(ids.shape[1],device=ids.device);same=r[:,:,None].eq(r[:,None,:])&r[:,:,None].gt(0);p=(same&(i[None,None,:]>i[None,:,None])&d[:,None,:]).sum(-1);v=(ids-7).clamp(0,9);t=(v*torch.pow(ids.new_tensor(10),p)*r.eq(3)).sum(1).long();return r,p,t
 def forward(s,ids,attention_mask=None):
  B,L=ids.shape;mask=ids.ne(0) if attention_mask is None else attention_mask.bool();r,p,t=s.parse(ids,mask);nw,xw=r.eq(1).sum(1),r.eq(2).sum(1)
  {contract}
  slots=torch.arange({slots},device=ids.device);v=(ids-7).clamp(0,9)
  def field(k):q=r.eq(k)[:,:,None]&p[:,:,None].eq(slots);return (v[:,:,None]*q).sum(1).long(),q.any(1)
  n,np=field(1);x,xp=field(2);ns=s.transition.encode(n,np);prob=F.one_hot(x,10).to(s.transition.digit.weight.dtype);end=prob.clamp_min(1e-8).log();present=xp;runs=int(t.max().item());state=None
  for i in range(runs):
   proposed,state=s.transition(ns,np,prob,present);active=(t>i)[:,None,None];end=torch.where(active,proposed,end);prob=torch.where(active,proposed.softmax(-1),prob);present=torch.where(active[:,:,0],np,present)
  target=mask.sum(1)[:,None]-1-slots[None];active=slots[None]<nw[:,None];place=F.one_hot(target.clamp(0,L-1),L).to(end.dtype)*active[...,None];placed=torch.bmm(place.transpose(1,2),end);occupied=place.sum(1).bool();digit=F.pad(placed,(7,s.config.vocab_size-17),value=-10000.);logits=end.new_full((B,L,s.config.vocab_size),-10000.);return torch.where(occupied[...,None],digit,logits),{{'widths':nw,'x_widths':xw,'steps':t,'macrosteps':runs,'state':state}}
def build_model(spec):
 m=Model(spec);assert_model_state(m,spec);return m
def build_optimizer(m,spec):
 d,n=[],[]
 for name,p in m.named_parameters():(d if p.ndim==2 and 'embedding' not in name else n).append(p)
 o=torch.optim.AdamW([{{'params':d,'weight_decay':.01}},{{'params':n,'weight_decay':0.}}],lr={lr},betas=(.9,.95),eps=1e-8,capturable=spec.device_type=='cuda');return OptimizerBundle(o,torch.optim.lr_scheduler.LambdaLR(o,lambda u:min((u+1)/{warmup},1.)))
{loss}SUBMISSION=Submission(build_model,build_optimizer,batch_size={batch},eval_batch_size=512,max_steps=None{loss_argument})
'''


def main() -> None:
    for name, d_model, hidden, batch, lr, warmup, focal in MEDIUM_VARIANTS:
        target = ROOT / "submissions" / f"conditional_medium_{name}_v1" / "submission.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            source(name, d_model, hidden, batch, lr, warmup, focal),
            encoding="utf-8",
        )
    target = ROOT / "submissions" / "conditional_dynamic_hard_v1" / "submission.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        source("dynamic_hard", 96, 48, 256, "9e-4", 8, ".5", dynamic=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
