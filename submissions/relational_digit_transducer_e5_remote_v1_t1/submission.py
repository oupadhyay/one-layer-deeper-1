"""A fixed-capacity relational digit transducer with continuous latent recurrence."""

import torch
from torch import nn
from benchmark import OptimizerBundle, Submission, assert_model_state

PAD, N, X, T, ANS, DIGIT = 0, 2, 3, 4, 5, 7
D, HEADS, MAX_PLACES, SCRATCH, MAX_STEPS = 64, 4, 16, 4, 64
STATE_ELEMENTS = 160_202


class Config:
    def __init__(self, spec):
        self.vocab_size = spec.vocab_size
        self.max_seq_len = spec.max_seq_len


class Attention(nn.Module):
    def __init__(self, cross=False):
        super().__init__()
        self.q = nn.Linear(D, D, bias=False)
        self.kv = nn.Linear(D, 2 * D, bias=False)
        self.out = nn.Linear(D, D, bias=False)
        self.cross = cross

    def forward(self, query, memory, mask):
        b, qn, _ = query.shape
        mn = memory.shape[1]
        q = self.q(query).reshape(b, qn, HEADS, D // HEADS).transpose(1, 2)
        k, v = self.kv(memory).chunk(2, -1)
        k = k.reshape(b, mn, HEADS, D // HEADS).transpose(1, 2)
        v = v.reshape(b, mn, HEADS, D // HEADS).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) * ((D // HEADS) ** -.5)
        scores = scores.masked_fill(~mask[:, None, None, :], -1e4)
        value = torch.matmul(scores.softmax(-1), v).transpose(1, 2).reshape(b, qn, D)
        return self.out(value)


class MemoryBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.RMSNorm(D)
        self.attention = Attention()
        self.norm2 = nn.RMSNorm(D)
        self.ffn = nn.Sequential(nn.Linear(D, 2 * D), nn.GELU(), nn.Linear(2 * D, D))

    def forward(self, memory, mask):
        z = self.norm1(memory)
        memory = memory + self.attention(z, z, mask)
        return memory + self.ffn(self.norm2(memory))


class Model(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.config = Config(spec)
        self.max_seq_len = spec.max_seq_len
        self.vocab_size = spec.vocab_size
        self.digit_embedding = nn.Embedding(10, D)
        self.place_embedding = nn.Embedding(MAX_PLACES, D)
        self.n_role = nn.Parameter(torch.empty(D))
        self.x_role = nn.Parameter(torch.empty(D))
        self.output_role = nn.Parameter(torch.empty(D))
        self.scratch_tokens = nn.Parameter(torch.empty(SCRATCH, D))
        self.n_cell = nn.GRUCell(D, D)
        self.pair_mlp = nn.Sequential(nn.Linear(2 * D, 2 * D), nn.GELU(), nn.Linear(2 * D, D))
        self.pair_film = nn.Linear(D, 2 * D)
        self.memory_block = MemoryBlock()
        self.initial_decoder = nn.Linear(2 * D, D)
        self.cross_attention = Attention(cross=True)
        self.decoder_film = nn.Linear(D, 4 * D)
        self.decoder_cell = nn.GRUCell(D, D)
        self.latent_norm = nn.RMSNorm(D)
        self.readout = nn.Linear(D, 10)
        for value in (self.n_role, self.x_role, self.output_role, self.scratch_tokens):
            nn.init.normal_(value, std=.02)

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

    def encode_n(self, nd, nmask):
        places = nd.shape[1]
        tokens = self.digit_embedding(nd) + self.place_embedding.weight[None, :places] + self.n_role
        hidden = tokens.new_zeros(tokens.shape[0], D)
        for place in range(places):
            update = self.n_cell(tokens[:, place], hidden)
            hidden = torch.where(nmask[:, place, None], update, hidden)
        return tokens, hidden

    def relational_memory(self, x_state, n_tokens, n_hidden, xmask, nmask):
        b, places, _ = x_state.shape
        left = x_state[:, :, None, :].expand(-1, -1, places, -1)
        right = x_state[:, None, :, :].expand(-1, places, -1, -1)
        pairs = self.pair_mlp(torch.cat((left, right), -1)).reshape(b, places * places, D)
        gamma, beta = self.pair_film(n_hidden).chunk(2, -1)
        pairs = (1 + gamma.tanh()[:, None]) * pairs + beta[:, None]
        pair_mask = (xmask[:, :, None] & xmask[:, None, :]).reshape(b, -1)
        scratch = self.scratch_tokens[None].expand(b, -1, -1)
        memory = torch.cat((pairs, n_tokens, scratch), 1)
        memory_mask = torch.cat((pair_mask, nmask, torch.ones(b, SCRATCH, dtype=torch.bool, device=x_state.device)), 1)
        memory = self.memory_block(memory, memory_mask)
        memory = self.memory_block(memory, memory_mask)
        return memory, memory_mask, memory[:, -SCRATCH:]

    def transition(self, x_state, n_tokens, n_hidden, xmask, nmask):
        memory, memory_mask, scratch = self.relational_memory(x_state, n_tokens, n_hidden, xmask, nmask)
        hidden = self.initial_decoder(torch.cat((n_hidden, scratch.mean(1)), -1))
        film = self.decoder_film(n_hidden)
        gi, bi, gh, bh = film.chunk(4, -1)
        outputs = []
        next_state = []
        for place in range(x_state.shape[1]):
            query = hidden + self.place_embedding.weight[place] + self.output_role
            context = self.cross_attention(query[:, None], memory, memory_mask)[:, 0]
            cell_input = (1 + gi.tanh()) * (query + context) + bi
            cell_hidden = (1 + gh.tanh()) * hidden + bh
            update = self.decoder_cell(cell_input, cell_hidden)
            hidden = torch.where(xmask[:, place, None], update, hidden)
            normalized = self.latent_norm(hidden)
            outputs.append(self.readout(normalized))
            latent = normalized + self.place_embedding.weight[place] + self.x_role
            next_state.append(torch.where(xmask[:, place, None], latent, x_state[:, place]))
        return torch.stack(next_state, 1), torch.stack(outputs, 1)

    def forward(self, input_ids, attention_mask=None):
        mask, nd, xd, nmask, xmask, steps = self.prepare(input_ids, attention_mask)
        active_places = int(nmask.sum(1).max().item())
        if not 1 <= active_places <= MAX_PLACES:
            raise ValueError(f"active modulus places must be in 1..{MAX_PLACES}")
        nd = nd[:, :active_places]
        xd = xd[:, :active_places]
        nmask = nmask[:, :active_places]
        # Residues are represented at canonical modulus width.  Missing high X
        # digits are active zero-digit latents rather than masked padding.
        xmask = nmask
        n_tokens, n_hidden = self.encode_n(nd, nmask)
        x_state = self.digit_embedding(xd) + self.place_embedding.weight[None, :active_places] + self.x_role
        endpoint = self.readout(self.latent_norm(x_state))
        macrosteps = 1 if self.training else int(steps.max().item())
        for macro in range(macrosteps):
            candidate, candidate_logits = self.transition(x_state, n_tokens, n_hidden, xmask, nmask)
            active = (steps > macro)[:, None, None]
            x_state = torch.where(active, candidate, x_state)
            endpoint = torch.where(active, candidate_logits, endpoint)
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)[None]
        widths = nmask.sum(1)
        slot = mask.sum(1)[:, None] - 1 - positions
        slot = torch.minimum(slot.clamp_min(0), (widths - 1).clamp_min(0)[:, None])
        selected = endpoint.gather(1, slot[..., None].expand(-1, -1, 10))
        logits = endpoint.new_full((*input_ids.shape, self.vocab_size), -1e4)
        logits[:, :, DIGIT:DIGIT + 10] = selected
        if self.training:
            scale = steps.eq(1).to(logits.dtype)[:, None, None]
            logits = logits.detach() + scale * (logits - logits.detach())
        auxiliary = {"parsed_steps": steps, "widths": widths, "macrosteps": macrosteps,
                     "memory_block_calls": 2 * macrosteps, "scratch_resets": macrosteps,
                     "decoder_calls": active_places * macrosteps,
                     "active_places": active_places, "pair_tokens": active_places * active_places}
        return logits, auxiliary


def build_model(spec):
    model = Model(spec)
    actual = assert_model_state(model, spec)
    if STATE_ELEMENTS and actual != STATE_ELEMENTS:
        raise RuntimeError(f"state element drift: {actual}")
    return model


def build_optimizer(model, spec):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        (decay if parameter.ndim == 2 and "embedding" not in name else no_decay).append(parameter)
    optimizer = torch.optim.AdamW([
        {"params": decay, "weight_decay": .01},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=8e-4, betas=(.9, .98), capturable=spec.device_type == "cuda")
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda update: min((update + 1) / 50, 1.0))
    return OptimizerBundle(optimizer, scheduler)


SUBMISSION = Submission(build_model, build_optimizer, batch_size=256, eval_batch_size=512, max_steps=None)
