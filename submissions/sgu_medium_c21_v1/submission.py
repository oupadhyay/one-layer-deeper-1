"""Standalone Medium M3 axial SGU with the C21 normalization recipe."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from benchmark import ModelSpec, OptimizerSpec, OptimizerBundle, Submission, assert_model_state
PAD,N_MARK,X_MARK,T_MARK,DIGIT_BASE=0,2,3,4,7
WIDTH,D_MODEL,HIDDEN,MAX_T=5,112,224,64
STATE_ELEMENTS,NEG=79_809,-10_000.0
class Config:
 def __init__(self,vocab_size,max_seq_len): self.vocab_size,self.max_seq_len=vocab_size,max_seq_len
class RMSNorm(nn.Module):
 def __init__(self,width,eps=1e-6): super().__init__(); self.weight=nn.Parameter(torch.ones(width)); self.eps=eps
 def forward(self,x): return x*(x.float().square().mean(-1,keepdim=True)+self.eps).rsqrt().to(x.dtype)*self.weight
class SpatialGatingBlock(nn.Module):
 def __init__(self):
  super().__init__(); self.norm=RMSNorm(D_MODEL); self.expand=nn.Linear(D_MODEL,2*HIDDEN); self.gate_norm=nn.LayerNorm(HIDDEN); self.place_operator=nn.Parameter(torch.empty(WIDTH,WIDTH)); self.role_operator=nn.Parameter(torch.empty(WIDTH,2,2)); self.spatial_bias=nn.Parameter(torch.ones(WIDTH,2)); self.project=nn.Linear(HIDDEN,D_MODEL)
  nn.init.uniform_(self.place_operator,-1.25e-4,1.25e-4); nn.init.uniform_(self.role_operator,-1.25e-4,1.25e-4)
 def forward(self,x):
  u,v=F.gelu(self.expand(self.norm(x))).chunk(2,-1); v=self.gate_norm(v).reshape(v.shape[0],WIDTH,2,HIDDEN); g=(torch.einsum("op,bprh->borh",self.place_operator,v)+torch.einsum("por,bprh->bpoh",self.role_operator,v)+self.spatial_bias[None,:,:,None]).reshape_as(u); return x+self.project(u*g)
class SGURegister(nn.Module):
 def __init__(self):
  super().__init__(); self.digit_embedding=nn.Embedding(10,D_MODEL); self.role_embedding=nn.Embedding(2,D_MODEL); self.place_embedding=nn.Embedding(WIDTH,D_MODEL); self.presence_embedding=nn.Embedding(2,D_MODEL); self.block=SpatialGatingBlock(); self.readout_norm=RMSNorm(D_MODEL); self.readout=nn.Linear(D_MODEL,10)
 def forward(self,n,np,x,xp):
  dt=self.digit_embedding.weight.dtype; a=self.digit_embedding(n)*np[...,None].to(dt); b=(x.to(dt)@self.digit_embedding.weight)*xp[...,None].to(dt); p=self.place_embedding.weight[None]; r=torch.stack((a+self.role_embedding.weight[0]+p+self.presence_embedding(np.long()),b+self.role_embedding.weight[1]+p+self.presence_embedding(xp.long())),2).reshape(n.shape[0],10,D_MODEL); r=self.block(r); return self.readout(self.readout_norm(r[:,1::2])),r
class Model(nn.Module):
 def __init__(self,spec): super().__init__(); self.config=Config(spec.vocab_size,spec.max_seq_len); self.transition=SGURegister()
 @staticmethod
 def parse(ids,mask):
  d=ids.ge(7)&ids.lt(17)&mask; mark=ids.eq(2).long()+2*ids.eq(3).long()+3*ids.eq(4).long(); role=torch.cummax(mark,1).values*d.long(); ix=torch.arange(ids.shape[1],device=ids.device); same=role[:,:,None].eq(role[:,None,:])&role[:,:,None].gt(0); place=(same&(ix[None,None,:]>ix[None,:,None])&d[:,None,:]).sum(-1); val=(ids-7).clamp(0,9); t=(val*torch.pow(ids.new_tensor(10),place)*role.eq(3)).sum(1).long(); return role,place,t
 def forward(self,ids,attention_mask=None):
  B,L=ids.shape
  if L>self.config.max_seq_len: raise ValueError("input sequence exceeds max_seq_len")
  mask=ids.ne(0) if attention_mask is None else attention_mask.bool(); role,place,steps=self.parse(ids,mask); nw,xw=role.eq(1).sum(1),role.eq(2).sum(1)
  if bool(((nw<1)|(nw>5)|(xw<1)|(xw>5)).any()): raise ValueError("N and X widths must be between one and five")
  if bool((steps>64).any()): raise ValueError("T exceeds 64")
  slots=torch.arange(5,device=ids.device); val=(ids-7).clamp(0,9)
  def field(k):
   q=role.eq(k)[:,:,None]&place[:,:,None].eq(slots); return (val[:,:,None]*q).sum(1).long(),q.any(1)
  n,np=field(1); x,xp=field(2); prob=F.one_hot(x,10).to(self.transition.digit_embedding.weight.dtype); end=prob.clamp_min(1e-8).log(); present=xp; runs=int(steps.max().item()); reg=None
  for i in range(runs):
   proposed,reg=self.transition(n,np,prob,present); active=(steps>i)[:,None,None]; end=torch.where(active,proposed,end); prob=torch.where(active,proposed.softmax(-1),prob); present=torch.where(active[:,:,0],np,present)
  target=mask.sum(1)[:,None]-1-slots[None]; active=slots[None]<nw[:,None]; placement=F.one_hot(target.clamp(0,L-1),L).to(end.dtype)*active[...,None]; placed=torch.bmm(placement.transpose(1,2),end); occupied=placement.sum(1).bool(); digit=F.pad(placed,(7,self.config.vocab_size-17),value=NEG); logits=end.new_full((B,L,self.config.vocab_size),NEG); logits=torch.where(occupied[...,None],digit,logits); return logits,{"widths":nw,"x_widths":xw,"steps":steps,"macrosteps":runs,"register":reg}
def build_model(spec):
 m=Model(spec); actual=assert_model_state(m,spec)
 if actual!=STATE_ELEMENTS: raise RuntimeError(f"state element drift: {actual}")
 return m
def build_optimizer(model,spec):
 d,n=[],[]
 for name,p in model.named_parameters(): (d if p.ndim==2 and "embedding" not in name else n).append(p)
 o=torch.optim.AdamW([{"params":d,"weight_decay":.01},{"params":n,"weight_decay":0.}],lr=6e-4,betas=(.9,.95),eps=1e-8,capturable=spec.device_type=="cuda"); return OptimizerBundle(o,torch.optim.lr_scheduler.LambdaLR(o,lambda u:min((u+1)/32,1.)))
SUBMISSION=Submission(build_model,build_optimizer,batch_size=256,eval_batch_size=512,max_steps=None)
