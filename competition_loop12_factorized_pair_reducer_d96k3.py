"""D96 factorized ordered digit-pair context with three tied reducer phases."""

import torch
import torch.nn.functional as F
from torch import nn
from benchmark import OptimizerBundle, Submission, assert_model_state

PAD, N, X, T, ANS, DIGIT = 0, 2, 3, 4, 5, 7
D, PLACES, MICROPHASES, MAX_STEPS, SCRATCH = 96, 4, 3, 64, 8
STATE_ELEMENTS = 143_722


class Config:
    def __init__(self, spec):
        self.vocab_size, self.max_seq_len = spec.vocab_size, spec.max_seq_len


class PairEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2*D, D), nn.GELU(), nn.Linear(D, D))
        self.row_place, self.column_place = nn.Embedding(PLACES, D), nn.Embedding(PLACES, D)

    def forward(self, state):
        b,p,_=state.shape
        left=state[:,:,None,:].expand(-1,-1,p,-1); right=state[:,None,:,:].expand(-1,p,-1,-1)
        encoded=self.mlp(torch.cat((left,right),-1))
        encoded=encoded+self.row_place.weight[None,:,None]+self.column_place.weight[None,None,:]
        return encoded.reshape(b,p*p,D)


class TiedReducer(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1,self.norm2,self.norm3=nn.RMSNorm(D),nn.RMSNorm(D),nn.RMSNorm(D)
        self.qkv=nn.Linear(D,3*D,bias=False); self.self_out=nn.Linear(D,D,bias=False)
        self.q=nn.Linear(D,D,bias=False); self.kv=nn.Linear(D,2*D,bias=False); self.cross_out=nn.Linear(D,D,bias=False)
        self.ffn=nn.Sequential(nn.Linear(D,2*D),nn.GELU(),nn.Linear(2*D,D))

    @staticmethod
    def attention(q,k,v,mask=None):
        score=torch.matmul(q,k.transpose(-1,-2))*(D**-.5)
        if mask is not None: score=score.masked_fill(~mask[:,None],-1e4)
        return torch.matmul(score.softmax(-1),v)

    def forward(self,work,context,context_mask):
        z=self.norm1(work); q,k,v=self.qkv(z).chunk(3,-1)
        work=work+self.self_out(self.attention(q,k,v))
        q=self.q(self.norm2(work)); k,v=self.kv(context).chunk(2,-1)
        work=work+self.cross_out(self.attention(q,k,v,context_mask))
        return work+self.ffn(self.norm3(work))


class Model(nn.Module):
    def __init__(self,spec):
        super().__init__(); self.config=Config(spec); self.max_seq_len=spec.max_seq_len; self.vocab_size=spec.vocab_size
        self.digit_embedding=nn.Embedding(10,D); self.relative_place=nn.Embedding(PLACES,D)
        self.n_role=nn.Parameter(torch.empty(D)); self.output_tokens=nn.Parameter(torch.empty(PLACES,D))
        self.scratch_tokens=nn.Parameter(torch.empty(SCRATCH,D)); self.phase_embedding=nn.Embedding(MICROPHASES,D)
        self.pair_encoder=PairEncoder(); self.reducer=TiedReducer(); self.readout_norm=nn.RMSNorm(D); self.readout=nn.Linear(D,10)
        for p in (self.n_role,self.output_tokens,self.scratch_tokens): nn.init.normal_(p,std=.02)

    @staticmethod
    def parse(input_ids,mask):
        digit=(input_ids>=DIGIT)&(input_ids<DIGIT+10)&mask
        marker=(input_ids==N)|(input_ids==X)|(input_ids==T)|(input_ids==ANS)
        role=torch.cumsum(marker.long(),1)*digit; index=torch.arange(input_ids.shape[1],device=input_ids.device)
        same=role[:,:,None].eq(role[:,None,:]); place=(same&(index[None,None]>index[None,:,None])&digit[:,None]).sum(2)
        values=(input_ids-DIGIT).clamp(0,9); slots=torch.arange(PLACES,device=input_ids.device)
        def field(which):
            assignment=role[:,:,None].eq(which)&place[:,:,None].eq(slots)
            return torch.einsum("blp,bl->bp",assignment.long(),values),assignment.any(1)
        nd,nmask=field(1); xd,_=field(2); td=digit&role.eq(3)
        steps=((values*torch.pow(input_ids.new_tensor(10),place))*td).sum(1).clamp(0,MAX_STEPS)
        return nd.long(),xd.long(),nmask.sum(1),steps

    def prepare(self,ids,attention_mask=None):
        if ids.shape[1]>self.max_seq_len: raise ValueError("sequence exceeds max_seq_len")
        mask=ids.ne(PAD) if attention_mask is None else attention_mask.bool(); nd,xd,widths,steps=self.parse(ids,mask)
        pm=torch.arange(PLACES,device=ids.device)[None]<widths[:,None]
        return mask,torch.where(pm,nd,0),torch.where(pm,xd,0),widths,steps,pm

    def _contexts(self,probabilities,nd,place_mask):
        state=probabilities@self.digit_embedding.weight+self.relative_place.weight[None]
        pairs=self.pair_encoder(state); pair_mask=(place_mask[:,:,None]&place_mask[:,None,:]).reshape(state.shape[0],-1)
        nc=self.digit_embedding(nd)+self.relative_place.weight[None]+self.n_role
        return state,torch.cat((pairs,nc),1),torch.cat((pair_mask,place_mask),1)

    def debug_execution(self,input_ids,attention_mask=None):
        _,_,_,widths,steps,_=self.prepare(input_ids,attention_mask)
        return {"parsed_steps":steps,"widths":widths,"active_macrosteps":int(steps.sum()),"pair_token_count":int((widths*widths*steps).sum()),"reducer_calls":int(MICROPHASES*steps.max()),"scratch_resets":int(steps.max())}

    def forward(self,input_ids,attention_mask=None):
        mask,nd,xd,widths,steps,pm=self.prepare(input_ids,attention_mask)
        probabilities=F.one_hot(xd,10).to(self.digit_embedding.weight.dtype); zero=F.one_hot(torch.zeros_like(xd),10).to(probabilities.dtype)
        endpoint=torch.log(probabilities.clamp_min(1e-7)); feedback_states=[]
        for macro in range(int(steps.max().item())):
            state,context,cm=self._contexts(probabilities,nd,pm); outputs=state+self.output_tokens[None]
            scratch=self.scratch_tokens[None].expand(input_ids.shape[0],-1,-1); work=torch.cat((outputs,scratch),1)
            for phase in range(MICROPHASES): work=self.reducer(work+self.phase_embedding.weight[phase],context,cm)
            digit_logits=self.readout(self.readout_norm(work[:,:PLACES])); soft=digit_logits.softmax(-1); hard=F.one_hot(soft.argmax(-1),10).to(soft.dtype)
            feedback=hard-soft.detach()+soft if self.training else hard; feedback=torch.where(pm[...,None],feedback,zero)
            active=(steps>macro)[:,None,None]; probabilities=torch.where(active,feedback,probabilities); endpoint=torch.where(active,digit_logits,endpoint); feedback_states.append(probabilities)
        positions=torch.arange(input_ids.shape[1],device=input_ids.device)[None]; slot=mask.sum(1)[:,None]-1-positions
        slot=torch.minimum(slot.clamp_min(0),(widths-1).clamp_min(0)[:,None]); selected=endpoint.gather(1,slot[...,None].expand(-1,-1,10))
        logits=endpoint.new_full((input_ids.shape[0],input_ids.shape[1],self.vocab_size),-1e4); logits[:,:,DIGIT:DIGIT+10]=selected
        return logits,{**self.debug_execution(input_ids,attention_mask),"digit_probabilities":probabilities,"feedback_states":feedback_states,"plain_endpoint_ce":True}


def build_model(spec):
    model=Model(spec); assert_model_state(model,spec); actual=sum(v.numel() for v in model.state_dict().values())
    if actual!=STATE_ELEMENTS: raise RuntimeError(f"state element drift: {actual}")
    return model


def build_optimizer(model,spec):
    decay,no_decay=[],[]
    for name,p in model.named_parameters(): (decay if p.ndim==2 and "embedding" not in name else no_decay).append(p)
    optimizer=torch.optim.AdamW([{"params":decay,"weight_decay":.02},{"params":no_decay,"weight_decay":0.}],lr=8e-4,betas=(.9,.98))
    return OptimizerBundle(optimizer,torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:min(1.,(step+1)/20)))


SUBMISSION=Submission(build_model,build_optimizer,batch_size=64,eval_batch_size=128,max_steps=1600)
