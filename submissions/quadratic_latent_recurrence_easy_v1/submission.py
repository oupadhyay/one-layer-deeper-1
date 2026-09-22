"""Learned quadratic whole-state recurrence for the Easy tier."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn
from benchmark import OptimizerBundle,Submission,TokenLossBatch,assert_model_state
W,D,R,H=4,24,128,128
class C:
 def __init__(s,vocab_size,max_seq_len):s.vocab_size,s.max_seq_len=vocab_size,max_seq_len
class RMS(nn.Module):
 def __init__(s,n):super().__init__();s.weight=nn.Parameter(torch.ones(n))
 def forward(s,x):return x*(x.float().square().mean(-1,keepdim=True)+1e-6).rsqrt().to(x.dtype)*s.weight
class Transition(nn.Module):
 def __init__(s):
  super().__init__();s.ndigit=nn.Embedding(10,D);s.xdigit=nn.Embedding(10,D);s.presence=nn.Embedding(2,D);s.nplace=nn.Embedding(W,D);s.xplace=nn.Embedding(W,D);s.ncontext=nn.Sequential(nn.Linear(W*D,H),nn.GELU(),nn.Linear(H,3*R));s.quadratic=nn.Linear(W*D,R);s.linear=nn.Linear(W*D,R);s.mix=nn.Sequential(RMS(R),nn.Linear(R,H),nn.GELU(),RMS(H),nn.Linear(H,W*10))
 def encode_n(s,n,np):return (s.ndigit(n)+s.nplace.weight[None]+s.presence(np.long())).flatten(1)
 def forward(s,ncontext,x,xp):
  xe=x.to(s.xdigit.weight.dtype)@s.xdigit.weight+s.xplace.weight[None]+s.presence(xp.long());a,b,c=s.ncontext(ncontext).chunk(3,-1);q=s.quadratic(xe.flatten(1));u=torch.tanh(a)*q.square()+torch.tanh(b)*s.linear(xe.flatten(1))+c;return s.mix(u).view(-1,W,10)
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
  n,np=field(1);x,xp=field(2);ncontext=s.transition.encode_n(n,np);prob=F.one_hot(x,10).to(s.transition.xdigit.weight.dtype);end=prob.clamp_min(1e-8).log();present=xp;runs=3 if s.training else int(t.max().item())
  for step in range(runs):
   proposed=s.transition(ncontext,prob,present);active=(t>step)[:,None,None];end=torch.where(active,proposed,end);prob=torch.where(active,proposed.softmax(-1),prob);present=torch.where(active[:,:,0],np,present)
  target=mask.sum(1)[:,None]-1-slots[None];active=slots[None]<nw[:,None];place=F.one_hot(target.clamp(0,L-1),L).to(end.dtype)*active[...,None];placed=torch.bmm(place.transpose(1,2),end);occupied=place.sum(1).bool();digit=F.pad(placed,(7,s.config.vocab_size-17),value=-10000.);logits=end.new_full((B,L,s.config.vocab_size),-10000.);return torch.where(occupied[...,None],digit,logits),{'widths':nw,'x_widths':xw,'steps':t,'macrosteps':runs}
def build_model(spec):m=Model(spec);assert_model_state(m,spec);return m
def build_optimizer(m,spec):
 d,n=[],[]
 for p in m.parameters():(d if p.ndim==2 else n).append(p)
 o=torch.optim.AdamW([{'params':d,'weight_decay':.01},{'params':n,'weight_decay':0.}],lr=1e-3,betas=(.9,.95),eps=1e-8,capturable=spec.device_type=='cuda');return OptimizerBundle(o,torch.optim.lr_scheduler.LambdaLR(o,lambda u:min((u+1)/8,1.)))
def token_training_loss(batch:TokenLossBatch):
 ce=F.cross_entropy(batch.logits.transpose(1,2),batch.labels,ignore_index=-100,reduction='none');v=batch.valid_mask;mean=(ce*v).sum()/v.sum();w=ce.masked_fill(~v,-1e4).softmax(1);return mean+.5*(w*ce).sum(1).mean()
SUBMISSION=Submission(build_model,build_optimizer,batch_size=256,eval_batch_size=512,max_steps=None,token_training_loss=token_training_loss)
