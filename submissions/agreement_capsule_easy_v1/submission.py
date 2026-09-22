"""Agreement-routed output capsules for Easy E5."""
import torch
import torch.nn.functional as F
from torch import nn
from benchmark import OptimizerBundle,Submission,assert_model_state
PAD,N,X,T,DIGIT=0,2,3,4,7
P,D,H,ROUNDS,MAX_T=4,32,64,3,64
STATE=7306
class Config:
 def __init__(s,spec):s.vocab_size,s.max_seq_len=spec.vocab_size,spec.max_seq_len
class RMS(nn.Module):
 def __init__(s,n):super().__init__();s.weight=nn.Parameter(torch.ones(n))
 def forward(s,x):return x*(x.float().square().mean(-1,keepdim=True)+1e-6).rsqrt().to(x.dtype)*s.weight
class Transition(nn.Module):
 def __init__(s):
  super().__init__();s.digit=nn.Embedding(10,D);s.presence=nn.Embedding(2,D);s.role=nn.Embedding(2,D);s.place=nn.Embedding(P,D);s.query=nn.Parameter(torch.empty(P,D));s.vote1=nn.Linear(2*D,H);s.vote2=nn.Linear(H,D);s.norm=RMS(D);s.out=nn.Linear(D,10);nn.init.normal_(s.query,std=.02)
 def forward(s,n,np,x,xp):
  ne=s.digit(n)+s.presence(np.long())+s.role.weight[0]+s.place.weight[None]
  xe=x.to(s.digit.weight.dtype)@s.digit.weight;xe=xe+s.presence(xp.long())+s.role.weight[1]+s.place.weight[None]
  source=torch.cat((ne,xe),1);present=torch.cat((np,xp),1);q=s.query[None,None].expand(source.shape[0],source.shape[1],-1,-1);src=source[:,:,None].expand_as(q);votes=s.vote2(F.gelu(s.vote1(torch.cat((src,q),-1))));route=votes.new_zeros(votes.shape[:-1]);caps=None
  for _ in range(ROUNDS):
   assign=route.float().softmax(2).to(votes.dtype)*present[:,:,None].to(votes.dtype);caps=(assign[...,None]*votes).sum(1)/assign.sum(1).clamp_min(1e-6)[...,None];caps=torch.tanh(s.norm(caps));route=route+(votes*caps[:,None]).sum(-1)
  return s.out(caps),{'assignments':assign,'capsules':caps}
class Model(nn.Module):
 def __init__(s,spec):super().__init__();s.config=Config(spec);s.transition=Transition()
 @staticmethod
 def parse(ids,mask):
  d=ids.ge(DIGIT)&ids.lt(DIGIT+10)&mask;m=ids.eq(N).long()+2*ids.eq(X).long()+3*ids.eq(T).long();r=torch.cummax(m,1).values*d.long();i=torch.arange(ids.shape[1],device=ids.device);same=r[:,:,None].eq(r[:,None,:])&r[:,:,None].gt(0);p=(same&(i[None,None,:]>i[None,:,None])&d[:,None,:]).sum(-1);v=(ids-DIGIT).clamp(0,9);slots=torch.arange(P,device=ids.device)
  def field(k):q=r[:,:,None].eq(k)&p[:,:,None].eq(slots);return (v[:,:,None]*q).sum(1).long(),q.any(1)
  n,np=field(1);x,xp=field(2);steps=(v*torch.pow(ids.new_tensor(10),p)*r.eq(3)).sum(1).long();return n,np,x,xp,steps
 def forward(s,ids,attention_mask=None):
  B,L=ids.shape;mask=ids.ne(PAD) if attention_mask is None else attention_mask.bool();n,np,x,xp,steps=s.parse(ids,mask);nw,xw=np.sum(1),xp.sum(1)
  if L>s.config.max_seq_len or bool(((nw<1)|(nw>P)|(xw<1)|(xw>P)|(steps>MAX_T)).any()):raise ValueError('invalid Easy contract')
  prob=F.one_hot(x,10).to(s.transition.digit.weight.dtype);end=prob.clamp_min(1e-8).log();runs=3 if s.training else int(steps.max().item());detail=None
  for i in range(runs):
   proposed,detail=s.transition(n,np,prob,xp);active=(steps>i)[:,None,None];end=torch.where(active,proposed,end);prob=torch.where(active,proposed.softmax(-1),prob)
  slots=torch.arange(P,device=ids.device);target=mask.sum(1)[:,None]-1-slots[None];active=slots[None]<nw[:,None];place=F.one_hot(target.clamp(0,L-1),L).to(end.dtype)*active[...,None];placed=torch.bmm(place.transpose(1,2),end);occupied=place.sum(1).bool();digit=F.pad(placed,(DIGIT,s.config.vocab_size-DIGIT-10),value=-1e4);logits=end.new_full((B,L,s.config.vocab_size),-1e4);return torch.where(occupied[...,None],digit,logits),{'steps':steps,'macrosteps':runs,'routing':detail}
def build_model(spec):
 m=Model(spec);actual=assert_model_state(m,spec)
 if actual!=STATE:raise RuntimeError(f'state drift: {actual}')
 return m
def build_optimizer(m,spec):
 d,n=[],[]
 for name,p in m.named_parameters():(d if p.ndim==2 and 'embedding' not in name else n).append(p)
 o=torch.optim.AdamW([{'params':d,'weight_decay':.01},{'params':n,'weight_decay':0.}],lr=8e-4,betas=(.9,.95),eps=1e-8,capturable=spec.device_type=='cuda');return OptimizerBundle(o,torch.optim.lr_scheduler.LambdaLR(o,lambda u:min((u+1)/16,1.)))
SUBMISSION=Submission(build_model,build_optimizer,batch_size=256,eval_batch_size=512,max_steps=None)
