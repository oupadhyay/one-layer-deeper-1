"""Two-microtick learned role--filler tensor-product transition."""
import torch
import torch.nn.functional as F
from torch import nn
from benchmark import OptimizerBundle,Submission,assert_model_state
PAD,N,X,T,ANS,DIGIT=0,2,3,4,5,7
P,FILLER,ROLE,C,MAX_T,TICKS=4,32,16,4,64,2
STATE=14186
class Config:
 def __init__(s,spec):s.vocab_size,s.max_seq_len=spec.vocab_size,spec.max_seq_len
class Cell(nn.Module):
 def __init__(s):
  super().__init__();s.left=nn.Parameter(torch.empty(C,2,FILLER,FILLER));s.right=nn.Parameter(torch.empty(C,2,ROLE,ROLE));s.channel_gate=nn.Parameter(torch.zeros(C,FILLER,ROLE));s.residual_gate=nn.Parameter(torch.zeros(FILLER,ROLE));nn.init.xavier_uniform_(s.left);nn.init.xavier_uniform_(s.right)
 def kron(s,z,b):return torch.einsum('cij,bjk,ckl->bcil',s.left[:,b],z,s.right[:,b])
 def forward(s,z):
  update=(s.kron(z,0)*torch.sigmoid(s.kron(z,1)+s.channel_gate[None])).sum(1);g=torch.sigmoid(s.residual_gate)[None];return z+g*(torch.tanh(update)-z)
class Model(nn.Module):
 def __init__(s,spec):
  super().__init__();s.config=Config(spec);s.digit=nn.Embedding(10,FILLER);s.presence=nn.Embedding(2,FILLER);s.field=nn.Embedding(2,ROLE);s.place=nn.Embedding(P,ROLE);s.base=nn.Parameter(torch.empty(FILLER,ROLE));s.cell=Cell();s.query=nn.Parameter(torch.empty(P,ROLE));s.out=nn.Linear(FILLER,10);nn.init.normal_(s.base,std=.02);nn.init.normal_(s.query,std=.02)
 @staticmethod
 def layout(ids,mask):
  d=ids.ge(DIGIT)&ids.lt(DIGIT+10)&mask;m=(ids.eq(N)|ids.eq(X)|ids.eq(T)|ids.eq(ANS)).long();field=m.cumsum(1)*d;i=torch.arange(ids.shape[1],device=ids.device);same=field[:,:,None].eq(field[:,None,:]);place=(same&(i[None,None,:]>i[None,:,None])&d[:,None,:]).sum(2);return field,place,d
 @classmethod
 def parse(cls,ids,mask):
  field,place,d=cls.layout(ids,mask);v=(ids-DIGIT).clamp(0,9);slots=torch.arange(P,device=ids.device)
  def take(k):q=field[:,:,None].eq(k)&place[:,:,None].eq(slots);return (q*v[:,:,None]).sum(1).long(),q.any(1)
  n,np=take(1);x,xp=take(2);steps=(v*torch.pow(ids.new_tensor(10),place)*field.eq(3)).sum(1).long();return n,np,x,xp,steps,field,d
 def bind(s,d,p,k,prob=None):
  filler=s.digit(d) if prob is None else prob@s.digit.weight;filler=filler+s.presence(p.long());role=s.field.weight[k][None,None]+s.place.weight[None];return torch.einsum('bpf,bpr->bfr',filler,role.expand(d.shape[0],-1,-1))
 def transition(s,nm,prob,p):
  dummy=torch.zeros(prob.shape[:2],device=prob.device,dtype=torch.long);z=s.base[None]+nm+s.bind(dummy,p,1,prob)
  for _ in range(TICKS):z=s.cell(z)
  return s.out(torch.einsum('bfr,pr->bpf',z,s.query)),z
 def forward(s,ids,attention_mask=None):
  B,L=ids.shape;mask=ids.ne(PAD) if attention_mask is None else attention_mask.bool();n,np,x,xp,steps,field,d=s.parse(ids,mask);nw,xw=np.sum(1),xp.sum(1)
  if L>s.config.max_seq_len or bool(((nw<1)|(nw>P)|(xw<1)|(xw>P)|(steps>MAX_T)).any()):raise ValueError('invalid Easy contract')
  nm=s.bind(n,np,0);prob=F.one_hot(x,10).to(s.base.dtype);end=prob.clamp_min(1e-7).log();runs=3 if s.training else int(steps.max().item());state=None
  for i in range(runs):
   proposed,state=s.transition(nm,prob,xp);active=(steps>i)[:,None,None];prob=torch.where(active,proposed.softmax(-1),prob);end=torch.where(active,proposed,end)
  pos=torch.arange(L,device=ids.device)[None];slot=(mask.sum(1)[:,None]-1-pos).clamp(0,P-1);selected=end.gather(1,slot[...,None].expand(-1,-1,10));logits=selected.new_full((B,L,s.config.vocab_size),-1e4);logits[:,:,DIGIT:DIGIT+10]=selected;return logits,{'steps':steps,'macrosteps':runs,'microticks':runs*TICKS,'state':state}
def build_model(spec):
 m=Model(spec);actual=assert_model_state(m,spec)
 if actual!=STATE:raise RuntimeError(f'state drift: {actual}')
 return m
def build_optimizer(m,spec):
 d,n=[],[]
 for name,p in m.named_parameters():(d if p.ndim==2 and 'embedding' not in name else n).append(p)
 o=torch.optim.AdamW([{'params':d,'weight_decay':.01},{'params':n,'weight_decay':0.}],lr=6e-4,betas=(.9,.95),eps=1e-8,capturable=spec.device_type=='cuda');return OptimizerBundle(o,torch.optim.lr_scheduler.LambdaLR(o,lambda u:min((u+1)/32,1.)))
SUBMISSION=Submission(build_model,build_optimizer,batch_size=256,eval_batch_size=512,max_steps=None)
