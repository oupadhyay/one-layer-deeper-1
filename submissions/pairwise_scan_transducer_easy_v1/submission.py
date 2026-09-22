"""Pairwise digit-relation scan for the Easy tier."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn
from benchmark import ModelSpec,OptimizerSpec,OptimizerBundle,Submission,TokenLossBatch,assert_model_state
W,D,H=4,48,96
class C:
 def __init__(s,vocab_size,max_seq_len):s.vocab_size,s.max_seq_len=vocab_size,max_seq_len
class RMS(nn.Module):
 def __init__(s,n):super().__init__();s.weight=nn.Parameter(torch.ones(n))
 def forward(s,x):return x*(x.float().square().mean(-1,keepdim=True)+1e-6).rsqrt().to(x.dtype)*s.weight
class Transition(nn.Module):
 def __init__(s):
  super().__init__();s.digit=nn.Embedding(10,D);s.xdigit=nn.Embedding(10,D);s.presence=nn.Embedding(2,D);s.nplace=nn.Embedding(W,D);s.xplace=nn.Embedding(W,D);s.relation=nn.Linear(3*D,H);s.ninit=nn.Linear(D,H);s.scan=nn.GRU(H,H,batch_first=True);s.norm=RMS(H);s.out=nn.Linear(H,10)
 def encode(s,n,np):
  p=s.nplace.weight[None];e=s.digit(n)+p+s.presence(np.long());w=np[...,None].to(e.dtype);summary=(e*w).sum(1)/w.sum(1).clamp_min(1.);return e,torch.tanh(s.ninit(summary))[None]
 def forward(s,ne,np,x,xp,initial):
  xe=x.to(s.xdigit.weight.dtype)@s.xdigit.weight;xe=xe+s.xplace.weight[None]+s.presence(xp.long());B=xe.shape[0];ni=ne[:,None].expand(B,W,W,D);xi=xe[:,:,None].expand(B,W,W,D);delta=(s.xplace.weight[:,None]-s.nplace.weight[None])[None].expand(B,W,W,D);pair=F.gelu(s.relation(torch.cat((ni,xi,delta),-1)));weight=np[:,None,:,None].to(pair.dtype);related=(pair*weight).sum(2)/weight.sum(2).clamp_min(1.);state,_=s.scan(related,initial);return s.out(s.norm(state)),state
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
  n,np=field(1);x,xp=field(2);ne,initial=s.transition.encode(n,np);prob=F.one_hot(x,10).to(s.transition.xdigit.weight.dtype);end=prob.clamp_min(1e-8).log();present=xp;runs=3 if s.training else int(t.max().item());state=None
  for i in range(runs):
   proposed,state=s.transition(ne,np,prob,present,initial);active=(t>i)[:,None,None];end=torch.where(active,proposed,end);prob=torch.where(active,proposed.softmax(-1),prob);present=torch.where(active[:,:,0],np,present)
  target=mask.sum(1)[:,None]-1-slots[None];active=slots[None]<nw[:,None];place=F.one_hot(target.clamp(0,L-1),L).to(end.dtype)*active[...,None];placed=torch.bmm(place.transpose(1,2),end);occupied=place.sum(1).bool();digit=F.pad(placed,(7,s.config.vocab_size-17),value=-10000.);logits=end.new_full((B,L,s.config.vocab_size),-10000.);return torch.where(occupied[...,None],digit,logits),{'widths':nw,'x_widths':xw,'steps':t,'macrosteps':runs,'state':state}
def build_model(spec):
 m=Model(spec);assert_model_state(m,spec);return m
def build_optimizer(m,spec):
 d,n=[],[]
 for p in m.parameters():(d if p.ndim==2 else n).append(p)
 opt=torch.optim.AdamW([{'params':d,'weight_decay':.01},{'params':n,'weight_decay':0.}],lr=1e-3,betas=(.9,.95),eps=1e-8,capturable=spec.device_type=='cuda');return OptimizerBundle(opt,torch.optim.lr_scheduler.LambdaLR(opt,lambda u:min((u+1)/8,1.)))
def token_training_loss(batch:TokenLossBatch):
 ce=F.cross_entropy(batch.logits.transpose(1,2),batch.labels,ignore_index=-100,reduction='none');v=batch.valid_mask;mean=(ce*v).sum()/v.sum();w=ce.masked_fill(~v,-1e4).softmax(1);return mean+.5*(w*ce).sum(1).mean()
SUBMISSION=Submission(build_model,build_optimizer,batch_size=256,eval_batch_size=512,max_steps=None,token_training_loss=token_training_loss)
