"""Conditional optimizer screen: lr5."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn
from benchmark import ModelSpec,OptimizerSpec,OptimizerBundle,Submission,assert_model_state
W,D,H=4,64,32
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
  if L>s.config.max_seq_len or bool(((nw<1)|(nw>W)|(xw<1)|(xw>W)|(t>64)).any()):raise ValueError('invalid Easy contract')
  slots=torch.arange(W,device=ids.device);v=(ids-7).clamp(0,9)
  def field(k):q=r.eq(k)[:,:,None]&p[:,:,None].eq(slots);return (v[:,:,None]*q).sum(1).long(),q.any(1)
  n,np=field(1);x,xp=field(2);ns=s.transition.encode(n,np);prob=F.one_hot(x,10).to(s.transition.digit.weight.dtype);end=prob.clamp_min(1e-8).log();present=xp;runs=int(t.max().item());state=None
  for i in range(runs):
   proposed,state=s.transition(ns,np,prob,present);active=(t>i)[:,None,None];end=torch.where(active,proposed,end);prob=torch.where(active,proposed.softmax(-1),prob);present=torch.where(active[:,:,0],np,present)
  target=mask.sum(1)[:,None]-1-slots[None];active=slots[None]<nw[:,None];place=F.one_hot(target.clamp(0,L-1),L).to(end.dtype)*active[...,None];placed=torch.bmm(place.transpose(1,2),end);occupied=place.sum(1).bool();digit=F.pad(placed,(7,s.config.vocab_size-17),value=-10000.);logits=end.new_full((B,L,s.config.vocab_size),-10000.);return torch.where(occupied[...,None],digit,logits),{'widths':nw,'x_widths':xw,'steps':t,'macrosteps':runs,'state':state}
def build_model(spec):
 m=Model(spec);actual=assert_model_state(m,spec)
 if actual!=40970:raise RuntimeError(f'state element drift: {actual}')
 return m
def build_optimizer(m,spec):
 decay,no_decay=[],[]
 for name,p in m.named_parameters():(decay if p.ndim==2 and 'embedding' not in name else no_decay).append(p)
 opt=torch.optim.AdamW([{'params':decay,'weight_decay':.01},{'params':no_decay,'weight_decay':0.}],lr=5e-4,betas=(.9,.95),eps=1e-8,capturable=spec.device_type=='cuda');return OptimizerBundle(opt,torch.optim.lr_scheduler.LambdaLR(opt,lambda u:min((u+1)/32,1.)))
SUBMISSION=Submission(build_model,build_optimizer,batch_size=256,eval_batch_size=512,max_steps=None)
