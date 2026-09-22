"""Local Easy-E5 triadic candidate-energy transducer."""

import torch
from torch import nn
from benchmark import OptimizerBundle, Submission, assert_model_state

PAD, N, X, T, ANS, DIGIT = 0, 2, 3, 4, 5, 7
D, HEADS, MAX_PLACES, MAX_STEPS = 80, 4, 16, 64
STATE_ELEMENTS = 118_240


class Config:
    def __init__(self, spec):
        self.vocab_size = spec.vocab_size
        self.max_seq_len = spec.max_seq_len


class CrossAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(D, D, bias=False)
        self.kv = nn.Linear(D, 2 * D, bias=False)
        self.out = nn.Linear(D, D, bias=False)

    def forward(self, query, memory, mask):
        b, qn, _ = query.shape
        mn = memory.shape[1]
        q = self.q(query).reshape(b, qn, HEADS, D // HEADS).transpose(1, 2)
        k, v = self.kv(memory).chunk(2, -1)
        k = k.reshape(b, mn, HEADS, D // HEADS).transpose(1, 2)
        v = v.reshape(b, mn, HEADS, D // HEADS).transpose(1, 2)
        score = torch.matmul(q, k.transpose(-1, -2)) * ((D // HEADS) ** -.5)
        score = score.masked_fill(~mask[:, None, None, :], -1e4)
        value = torch.matmul(score.softmax(-1), v).transpose(1, 2).reshape(b, qn, D)
        return self.out(value)


class Model(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.config = Config(spec)
        self.max_seq_len = spec.max_seq_len
        self.vocab_size = spec.vocab_size
        self.digit_embedding = nn.Embedding(10, D)
        self.place_embedding = nn.Embedding(MAX_PLACES, D)
        self.left_role = nn.Parameter(torch.empty(D))
        self.right_role = nn.Parameter(torch.empty(D))
        self.n_role = nn.Parameter(torch.empty(D))
        self.output_role = nn.Parameter(torch.empty(D))
        self.pair_mlp = nn.Sequential(nn.Linear(2 * D, 2 * D), nn.GELU(), nn.Linear(2 * D, D))
        self.triad_mlp = nn.Sequential(nn.Linear(2 * D, 2 * D), nn.GELU(), nn.Linear(2 * D, D))
        self.cross_attention = CrossAttention()
        self.query_norm = nn.RMSNorm(D)
        self.value_norm = nn.RMSNorm(D)
        self.query_projection = nn.Linear(D, D, bias=False)
        self.value_projection = nn.Linear(D, D, bias=False)
        for role in (self.left_role, self.right_role, self.n_role, self.output_role):
            nn.init.normal_(role, std=.02)

    @staticmethod
    def parse(input_ids, mask):
        digit = (input_ids >= DIGIT) & (input_ids < DIGIT + 10) & mask
        marker = (input_ids == N) | (input_ids == X) | (input_ids == T) | (input_ids == ANS)
        role = torch.cumsum(marker.long(), 1) * digit
        index = torch.arange(input_ids.shape[1], device=input_ids.device)
        same = role[:, :, None].eq(role[:, None, :])
        place = (same & (index[None, None] > index[None, :, None]) & digit[:, None]).sum(2)
        values = (input_ids - DIGIT).clamp(0, 9)
        slots = torch.arange(MAX_PLACES, device=input_ids.device)

        def field(which):
            assignment = role[:, :, None].eq(which) & place[:, :, None].eq(slots)
            return (assignment.to(values.dtype) * values[:, :, None]).sum(1).long(), assignment.any(1)

        nd, nmask = field(1)
        xd, xmask = field(2)
        td = digit & role.eq(3)
        steps = ((values * torch.pow(input_ids.new_tensor(10), place)) * td).sum(1).clamp(0, MAX_STEPS)
        return nd, xd, nmask, xmask, steps

    def prepare(self, ids, attention_mask=None):
        if ids.shape[1] > self.max_seq_len:
            raise ValueError("sequence exceeds max_seq_len")
        mask = ids.ne(PAD) if attention_mask is None else attention_mask.bool()
        nd, xd, nmask, xmask, steps = self.parse(ids, mask)
        return mask, nd, xd, nmask, xmask, steps

    def triadic_memory(self, probabilities, nd, nmask):
        b, width, _ = probabilities.shape
        places = self.place_embedding.weight[None, :width]
        x = torch.matmul(probabilities, self.digit_embedding.weight) + places
        left = (x + self.left_role)[:, :, None, :].expand(-1, -1, width, -1)
        right = (x + self.right_role)[:, None, :, :].expand(-1, width, -1, -1)
        pairs = self.pair_mlp(torch.cat((left, right), -1))
        pairs = torch.nn.functional.normalize(pairs, dim=-1)
        n_tokens = self.digit_embedding(nd) + places + self.n_role
        pairs = pairs[:, :, :, None, :].expand(-1, -1, -1, width, -1)
        ordered_n = n_tokens[:, None, None, :, :].expand(-1, width, width, -1, -1)
        memory = self.triad_mlp(torch.cat((pairs, ordered_n), -1)).reshape(b, width ** 3, D)
        mask = (nmask[:, :, None, None] & nmask[:, None, :, None] &
                nmask[:, None, None, :]).reshape(b, width ** 3)
        return memory * mask[:, :, None], mask

    def candidate_energies(self, queries, attended):
        query = self.query_projection(self.query_norm(queries))
        value = self.value_projection(self.value_norm(attended))
        return (query * value).sum(-1) * (D ** -.5)

    def transition(self, probabilities, nd, nmask):
        b, width, _ = probabilities.shape
        memory, memory_mask = self.triadic_memory(probabilities, nd, nmask)
        query = (self.place_embedding.weight[:width, None, :] + self.output_role +
                 self.digit_embedding.weight[None, :, :]).reshape(1, width * 10, D).expand(b, -1, -1)
        attended = self.cross_attention(query, memory, memory_mask)
        logits = self.candidate_energies(query, attended).reshape(b, width, 10)
        return logits.softmax(-1), logits

    def forward(self, input_ids, attention_mask=None):
        mask, nd, xd, nmask, _, steps = self.prepare(input_ids, attention_mask)
        active_places = int(nmask.sum(1).max().item())
        if not 1 <= active_places <= MAX_PLACES:
            raise ValueError(f"active modulus places must be in 1..{MAX_PLACES}")
        nd, xd, nmask = nd[:, :active_places], xd[:, :active_places], nmask[:, :active_places]
        probabilities = self.digit_embedding.weight.new_zeros((*xd.shape, 10)).scatter(2, xd[..., None], 1)
        endpoint = probabilities.new_zeros(probabilities.shape)
        macrosteps = int(steps.max().item())
        for macro in range(macrosteps):
            candidate, candidate_logits = self.transition(probabilities, nd, nmask)
            active = (steps > macro)[:, None, None]
            probabilities = torch.where(active, candidate, probabilities)
            endpoint = torch.where(active, candidate_logits, endpoint)
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)[None]
        widths = nmask.sum(1)
        slot = mask.sum(1)[:, None] - 1 - positions
        slot = torch.minimum(slot.clamp_min(0), (widths - 1).clamp_min(0)[:, None])
        selected = endpoint.gather(1, slot[..., None].expand(-1, -1, 10))
        logits = endpoint.new_full((*input_ids.shape, self.vocab_size), -1e4)
        logits[:, :, DIGIT:DIGIT + 10] = selected
        return logits, {"parsed_steps": steps, "widths": widths, "macrosteps": macrosteps,
                        "transition_calls": macrosteps, "active_places": active_places,
                        "triad_tokens": active_places ** 3}


def build_model(spec):
    model = Model(spec)
    actual = assert_model_state(model, spec)
    if actual != STATE_ELEMENTS:
        raise RuntimeError(f"state element drift: {actual}")
    return model


def build_optimizer(model, spec):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        (decay if parameter.ndim == 2 and "embedding" not in name else no_decay).append(parameter)
    optimizer = torch.optim.AdamW([
        {"params": decay, "weight_decay": .02},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=5e-4, betas=(.9, .98), eps=1e-8, capturable=spec.device_type == "cuda")
    return OptimizerBundle(optimizer, None)


SUBMISSION = Submission(build_model, build_optimizer, batch_size=512, eval_batch_size=512, max_steps=None)
