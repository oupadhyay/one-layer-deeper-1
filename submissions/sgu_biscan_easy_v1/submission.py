"""Tied bidirectional learned place-scan SGU for Easy E5."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn
from benchmark import ModelSpec,OptimizerSpec,OptimizerBundle,Submission,assert_model_state
W,D,H=4,112,224
class C:
 def __init__(s,vocab_size,max_seq_len): s.vocab_size,s.max_seq_len=vocab_size,max_seq_len
class RMS(nn.Module):
 def __init__(s,n): super().__init__();s.weight=nn.Parameter(torch.ones(n))
 def forward(s,x): return x*(x.float().square().mean(-1,keepdim=True)+1e-6).rsqrt().to(x.dtype)*s.weight
class Block(nn.Module):
 def __init__(s):
  super().__init__();s.norm=RMS(D);s.expand=nn.Linear(D,2*H);s.gnorm=nn.LayerNorm(H);s.scan_in=nn.Linear(2*H,32,bias=False);s.scan=nn.GRU(32,32,batch_first=True,bidirectional=True);s.scan_out=nn.Linear(64,2*H,bias=False);s.role=nn.Parameter(torch.empty(W,2,2));s.bias=nn.Parameter(torch.ones(W,2));s.project=nn.Linear(H,D);nn.init.uniform_(s.role,-1.25e-4,1.25e-4)
 def forward(s,x):
  u,v=F.gelu(s.expand(s.norm(x))).chunk(2,-1);v=s.gnorm(v).reshape(x.shape[0],W,2,H);z=s.scan_in(v.reshape(x.shape[0],W,2*H));z,_=s.scan(z);p=s.scan_out(z).reshape(x.shape[0],W,2,H);g=(p+torch.einsum('por,bprh->bpoh',s.role,v)+s.bias[None,:,:,None]).reshape_as(u);return x+s.project(u*g)
class Transition(nn.Module):
 def __init__(s): super().__init__();s.digit=nn.Embedding(10,D);s.role=nn.Embedding(2,D);s.place=nn.Embedding(W,D);s.presence=nn.Embedding(2,D);s.block=Block();s.outnorm=RMS(D);s.out=nn.Linear(D,10)
 def forward(s,n,np,x,xp):
  dt=s.digit.weight.dtype;a=s.digit(n)*np[...,None].to(dt);b=(x.to(dt)@s.digit.weight)*xp[...,None].to(dt);p=s.place.weight[None];r=torch.stack((a+s.role.weight[0]+p+s.presence(np.long()),b+s.role.weight[1]+p+s.presence(xp.long())),2).reshape(n.shape[0],2*W,D);r=s.block(r);return s.out(s.outnorm(r[:,1::2])),r
class Model(nn.Module):
 def __init__(s,spec): super().__init__();s.config=C(spec.vocab_size,spec.max_seq_len);s.transition=Transition()
 @staticmethod
 def parse(ids,mask):
  d=ids.ge(7)&ids.lt(17)&mask;m=ids.eq(2).long()+2*ids.eq(3).long()+3*ids.eq(4).long();r=torch.cummax(m,1).values*d.long();i=torch.arange(ids.shape[1],device=ids.device);same=r[:,:,None].eq(r[:,None,:])&r[:,:,None].gt(0);p=(same&(i[None,None,:]>i[None,:,None])&d[:,None,:]).sum(-1);v=(ids-7).clamp(0,9);t=(v*torch.pow(ids.new_tensor(10),p)*r.eq(3)).sum(1).long();return r,p,t
 def forward(s,ids,attention_mask=None):
  B,L=ids.shape;mask=ids.ne(0) if attention_mask is None else attention_mask.bool();r,p,t=s.parse(ids,mask);nw,xw=r.eq(1).sum(1),r.eq(2).sum(1)
  if L>s.config.max_seq_len or bool(((nw<1)|(nw>W)|(xw<1)|(xw>W)|(t>64)).any()): raise ValueError('invalid input contract')
  slots=torch.arange(W,device=ids.device);v=(ids-7).clamp(0,9)
  def field(k): q=r.eq(k)[:,:,None]&p[:,:,None].eq(slots);return (v[:,:,None]*q).sum(1).long(),q.any(1)
  n,np=field(1);x,xp=field(2);prob=F.one_hot(x,10).to(s.transition.digit.weight.dtype);end=prob.clamp_min(1e-8).log();present=xp;runs=3 if s.training else int(t.max().item());reg=None
  for i in range(runs):
   proposed,reg=s.transition(n,np,prob,present);active=(t>i)[:,None,None];end=torch.where(active,proposed,end);prob=torch.where(active,proposed.softmax(-1),prob);present=torch.where(active[:,:,0],np,present)
  target=mask.sum(1)[:,None]-1-slots[None];active=slots[None]<nw[:,None];place=F.one_hot(target.clamp(0,L-1),L).to(end.dtype)*active[...,None];placed=torch.bmm(place.transpose(1,2),end);occupied=place.sum(1).bool();digit=F.pad(placed,(7,s.config.vocab_size-17),value=-10000.);logits=end.new_full((B,L,s.config.vocab_size),-10000.);return torch.where(occupied[...,None],digit,logits),{'widths':nw,'x_widths':xw,'steps':t,'macrosteps':runs,'register':reg}
def build_model(spec):
 m=Model(spec);a=assert_model_state(m,spec)
 if a!=135346: raise RuntimeError(f'state element drift: {a}')
 return m
def build_optimizer(m,spec):
 d,n=[],[]
 for name,p in m.named_parameters(): (d if p.ndim==2 and 'embedding' not in name else n).append(p)
 o=torch.optim.AdamW([{'params':d,'weight_decay':.01},{'params':n,'weight_decay':0.}],lr=6e-4,betas=(.9,.95),eps=1e-8,capturable=spec.device_type=='cuda');return OptimizerBundle(o,torch.optim.lr_scheduler.LambdaLR(o,lambda u:min((u+1)/32,1.)))
SUBMISSION=Submission(build_model,build_optimizer,batch_size=256,eval_batch_size=512,max_steps=None)
