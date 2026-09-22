"""Equal-mixture compressed-axial and factorized-pair place transition."""
import math
import torch
import torch.nn.functional as F
from torch import nn
from benchmark import OptimizerBundle, Submission, assert_model_state

PAD, N, X, T, ANS, DIGIT = 0, 2, 3, 4, 5, 7
MAX_STEPS, C, D = 64, 32, 64

class Config:
    def __init__(self, spec):
        self.vocab_size, self.max_seq_len = spec.vocab_size, spec.max_seq_len

class AxialCell(nn.Module):
    def __init__(self):
        super().__init__(); self.norm=nn.RMSNorm(C); self.gates=nn.Linear(3*C,2*C); self.candidate=nn.Linear(3*C,C)
    def forward(self, x, axis):
        z=self.norm(x)
        if axis=="H": a,b=F.pad(z[:,:,:-1],(0,0,1,0)),F.pad(z[:,:,1:],(0,0,0,1))
        else: a,b=F.pad(z[:,:-1],(0,0,0,0,1,0)),F.pad(z[:,1:],(0,0,0,0,0,1))
        r,u=self.gates(torch.cat((z,a,b),-1)).chunk(2,-1); r,u=r.sigmoid(),u.sigmoid()
        return (1-u)*x+u*torch.tanh(self.candidate(torch.cat((z,r*a,r*b),-1)))

class AxialExpert(nn.Module):
    def __init__(self):
        super().__init__(); self.embedding=nn.Embedding(10,16); self.initializer=nn.Sequential(nn.Linear(70,C),nn.GELU(),nn.Linear(C,C)); self.cell=AxialCell(); self.readout=nn.Linear(C,10)
    def forward(self,p,nd,widths):
        b,w=p.shape[:2]; cols=w+1; s=p@self.embedding.weight; ns=self.embedding(nd)
        ei=s[:,:,None].expand(-1,-1,cols,-1); ni=ns[:,:,None].expand_as(ei)
        zero=s.new_zeros((b,w,1,16)); ej=torch.cat((s[:,None].expand(-1,w,-1,-1),zero),2); nj=torch.cat((ns[:,None].expand(-1,w,-1,-1),zero),2)
        rows=torch.arange(w,device=s.device)[None,:,None]; columns=torch.arange(cols,device=s.device)[None,None,:]; aw=widths[:,None,None]
        vr=rows<aw; pair=columns<aw; boundary=columns==aw; bc=boundary.expand(-1,w,-1)[...,None]
        ej=torch.where(bc,torch.zeros_like(ej),ej); nj=torch.where(bc,torch.zeros_like(nj),nj)
        flags=torch.stack((vr.expand(-1,-1,cols),pair.expand(-1,w,-1),boundary.expand(-1,w,-1),(rows==0).expand(b,-1,cols),(columns==0).expand(b,w,-1),(rows==aw-1).expand(-1,-1,cols)),-1).to(s.dtype)
        valid=vr&(pair|boundary); x=self.initializer(torch.cat((ei,ej,ni,nj,flags),-1))*valid[...,None]
        for axis in ("H","V","H","V"): x=self.cell(x,axis)*valid[...,None]
        index=widths[:,None,None,None].expand(-1,w,1,C)
        return self.readout(x.gather(2,index).squeeze(2))

class PairEncoder(nn.Module):
    def __init__(self,m):
        super().__init__(); self.mlp=nn.Sequential(nn.Linear(2*D,D),nn.GELU(),nn.Linear(D,D)); self.row_place=nn.Embedding(m,D); self.column_place=nn.Embedding(m,D)
    def forward(self,s,w):
        b=s.shape[0]; x=self.mlp(torch.cat((s[:,:,None].expand(-1,-1,w,-1),s[:,None].expand(-1,w,-1,-1)),-1))
        return (x+self.row_place.weight[None,:w,None]+self.column_place.weight[None,None,:w]).reshape(b,w*w,D)

class Reducer(nn.Module):
    def __init__(self):
        super().__init__(); self.norm1,self.norm2,self.norm3=nn.RMSNorm(D),nn.RMSNorm(D),nn.RMSNorm(D); self.qkv=nn.Linear(D,3*D,bias=False); self.self_out=nn.Linear(D,D,bias=False); self.q=nn.Linear(D,D,bias=False); self.kv=nn.Linear(D,2*D,bias=False); self.cross_out=nn.Linear(D,D,bias=False); self.ffn=nn.Sequential(nn.Linear(D,2*D),nn.GELU(),nn.Linear(2*D,D))
    @staticmethod
    def attention(q,k,v,mask=None):
        score=torch.matmul(q,k.transpose(-1,-2))*(D**-.5)
        if mask is not None: score=score.masked_fill(~mask[:,None],-1e4)
        return torch.matmul(score.softmax(-1),v)
    def forward(self,work,context,mask,work_mask):
        q,k,v=self.qkv(self.norm1(work)).chunk(3,-1); work=work+self.self_out(self.attention(q,k,v,work_mask)); q=self.q(self.norm2(work)); k,v=self.kv(context).chunk(2,-1); work=work+self.cross_out(self.attention(q,k,v,mask)); return work+self.ffn(self.norm3(work))

class PairExpert(nn.Module):
    def __init__(self,m):
        super().__init__(); self.digit_embedding=nn.Embedding(10,D); self.relative_place=nn.Embedding(m,D); self.n_role=nn.Parameter(torch.empty(D)); self.output_tokens=nn.Parameter(torch.empty(m,D)); self.scratch_tokens=nn.Parameter(torch.empty(m,D)); self.phase_embedding=nn.Embedding(1,D); self.pair_encoder=PairEncoder(m); self.reducer=Reducer(); self.readout_norm=nn.RMSNorm(D); self.readout=nn.Linear(D,10); nn.init.normal_(self.n_role,std=.02); nn.init.normal_(self.output_tokens,std=.02); nn.init.normal_(self.scratch_tokens,std=.02)
    def forward(self,p,nd,place_mask):
        b,w=p.shape[:2]; state=p@self.digit_embedding.weight+self.relative_place.weight[None,:w]; pairs=self.pair_encoder(state,w); pm=(place_mask[:,:,None]&place_mask[:,None,:]).reshape(b,-1); nc=self.digit_embedding(nd)+self.relative_place.weight[None,:w]+self.n_role; context=torch.cat((pairs,nc),1); mask=torch.cat((pm,place_mask),1); work=torch.cat((state+self.output_tokens[None,:w],self.scratch_tokens[None,:w].expand(b,-1,-1)),1); work_mask=torch.cat((place_mask,place_mask),1); work=self.reducer(work+self.phase_embedding.weight[0],context,mask,work_mask); return self.readout(self.readout_norm(work[:,:w]))

class Model(nn.Module):
    def __init__(self,spec):
        super().__init__(); self.config=Config(spec); self.max_seq_len=spec.max_seq_len; self.vocab_size=spec.vocab_size; self.axial=AxialExpert(); self.pair=PairExpert(spec.max_seq_len)
    @staticmethod
    def parse(ids,mask):
        digit=(ids>=DIGIT)&(ids<DIGIT+10)&mask; marker=(ids==N)|(ids==X)|(ids==T)|(ids==ANS); role=torch.cumsum(marker.long(),1)*digit; i=torch.arange(ids.shape[1],device=ids.device); same=role[:,:,None].eq(role[:,None,:]); place=(same&(i[None,None]>i[None,:,None])&digit[:,None]).sum(2); values=(ids-DIGIT).clamp(0,9); td=digit&role.eq(3); steps=((values*torch.pow(ids.new_tensor(10),place))*td).sum(1).clamp(0,MAX_STEPS); return role,place,steps
    def forward(self,input_ids,attention_mask=None):
        if input_ids.shape[1]>self.max_seq_len: raise ValueError("sequence exceeds max_seq_len")
        mask=input_ids.ne(PAD) if attention_mask is None else attention_mask.bool(); role,places,steps=self.parse(input_ids,mask); values=(input_ids-DIGIT).clamp(0,9); slots=torch.arange(self.max_seq_len,device=input_ids.device)
        def field(r):
            a=role[:,:,None].eq(r)&places[:,:,None].eq(slots); return (a.to(values.dtype)*values[:,:,None]).sum(1).long(),a.any(1)
        nd,nmask=field(1); xd,_=field(2); widths=nmask.sum(1); w=int(widths.max().item()); nd,xd=nd[:,:w],xd[:,:w]; pm=torch.arange(w,device=input_ids.device)[None]<widths[:,None]; nd=torch.where(pm,nd,torch.zeros_like(nd)); xd=torch.where(pm,xd,torch.zeros_like(xd)); p=F.one_hot(xd,10).to(self.axial.embedding.weight.dtype); endpoint=torch.log(p.clamp_min(1e-7)); js_sum=endpoint.sum()*0; js_count=0
        for macro in range(int(steps.max().item())):
            la=self.axial(p,nd,widths).float(); lp=self.pair(p,nd,pm).float(); lpa=F.log_softmax(la,-1); lpp=F.log_softmax(lp,-1); fused=torch.logaddexp(lpa+math.log(.5),lpp+math.log(.5)); pa,pb=lpa.exp(),lpp.exp(); mixture=.5*(pa+pb); js=.5*((pa*(lpa-torch.log(mixture.clamp_min(1e-30)))).sum(-1)+(pb*(lpp-torch.log(mixture.clamp_min(1e-30)))).sum(-1)); active=(steps>macro)[:,None]&pm; js_sum=js_sum+(js*active).sum(); js_count+=int(active.sum().item()); hard=F.one_hot(fused.argmax(-1),10).to(fused.dtype); feedback=hard-fused.exp().detach()+fused.exp() if self.training else hard; zero=F.one_hot(torch.zeros_like(xd),10).to(feedback.dtype); feedback=torch.where(pm[...,None],feedback,zero); active3=(steps>macro)[:,None,None]; p=torch.where(active3,feedback,p); endpoint=torch.where(active3,fused,endpoint)
        length=input_ids.shape[1]; pos=torch.arange(length,device=input_ids.device)[None]; slot=mask.sum(1)[:,None]-1-pos; slot=torch.minimum(slot.clamp_min(0),(widths-1).clamp_min(0)[:,None]); selected=endpoint.gather(1,slot[...,None].expand(-1,-1,10)); logits=endpoint.new_full((input_ids.shape[0],length,self.vocab_size),-1e4); logits[:,:,DIGIT:DIGIT+10]=selected; return logits,{"parsed_steps":steps,"widths":widths,"macrosteps":int(steps.max().item()),"axial_calls":int(steps.max().item()),"pair_calls":int(steps.max().item()),"branch_js":js_sum/max(js_count,1)}

def token_training_loss(logits,labels,auxiliary):
    if logits.ndim!=2 or labels.ndim!=1 or logits.shape[0]!=labels.shape[0]: raise ValueError("selected logits must be [N,V] and labels [N] with matching N")
    return F.cross_entropy(logits,labels)+.01*auxiliary["branch_js"]

def build_model(spec):
    model=Model(spec); assert_model_state(model,spec); return model
def build_optimizer(model,spec):
    axial=list(model.axial.parameters()); pair=[p for n,p in model.named_parameters() if not n.startswith("axial.")]; optimizer=torch.optim.AdamW([{"params":axial,"lr":1e-3},{"params":pair,"lr":1e-4}],betas=(.9,.95),eps=1e-8,weight_decay=.1,capturable=spec.device_type=="cuda"); return OptimizerBundle(optimizer)
SUBMISSION=Submission(build_model,build_optimizer,token_training_loss,batch_size=64,eval_batch_size=128,max_steps=None)
