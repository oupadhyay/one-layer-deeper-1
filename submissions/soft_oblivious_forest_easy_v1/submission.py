"""Tied soft oblivious forest for the Easy E5 transition."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from benchmark import OptimizerBundle, Submission, assert_model_state

PAD, N, X, T, DIGIT = 0, 2, 3, 4, 7
PLACES, WIDTH, TREES, DEPTH, LEAVES, MAX_T = 4, 64, 16, 5, 32, 64
STATE_ELEMENTS = 49_754


class Config:
    def __init__(self, spec):
        self.vocab_size, self.max_seq_len = spec.vocab_size, spec.max_seq_len


class SoftObliviousForest(nn.Module):
    """The same 16 depth-five trees serve every output place and tick."""
    def __init__(self):
        super().__init__()
        self.digit = nn.Embedding(10, 32)
        self.role = nn.Embedding(2, 32)
        self.place = nn.Embedding(PLACES, 32)
        self.presence = nn.Embedding(2, 32)
        self.phi = nn.Sequential(nn.Linear(32, WIDTH), nn.GELU())
        self.role_mix = nn.ModuleList((nn.Linear(WIDTH, WIDTH, bias=False),
                                       nn.Linear(WIDTH, WIDTH, bias=False)))
        self.query = nn.Parameter(torch.empty(PLACES, WIDTH))
        self.split_weight = nn.Parameter(torch.empty(TREES, DEPTH, WIDTH))
        self.split_bias = nn.Parameter(torch.zeros(TREES, DEPTH))
        self.leaf = nn.Parameter(torch.empty(TREES, LEAVES, WIDTH))
        self.decoder = nn.Linear(WIDTH, 10)
        nn.init.normal_(self.query, std=.02)
        nn.init.normal_(self.split_weight, std=.01)
        nn.init.normal_(self.leaf, std=.02)

    def contexts(self, n, np, x_prob, xp):
        place = self.place.weight[None]
        ne = self.digit(n) + self.role.weight[0] + place + self.presence(np.long())
        xe = x_prob.to(self.digit.weight.dtype) @ self.digit.weight
        xe = xe + self.role.weight[1] + place + self.presence(xp.long())

        def pool(tokens, present):
            values = self.phi(tokens) * present[..., None].to(tokens.dtype)
            return values.sum(1) / present.sum(1, keepdim=True).clamp_min(1).to(tokens.dtype)

        # Masked DeepSets summaries are shared by all four learned queries.
        summary = self.role_mix[0](pool(ne, np)) + self.role_mix[1](pool(xe, xp))
        return summary[:, None, :] + self.query[None]

    def path_probabilities(self, context):
        gates = torch.sigmoid(torch.einsum("bqd,tkd->bqtk", context, self.split_weight)
                              + self.split_bias[None, None])
        paths = gates.new_ones((*gates.shape[:-1], 1))
        for level in range(DEPTH):
            gate = gates[..., level, None]
            paths = torch.cat((paths * (1.0 - gate), paths * gate), -1)
        return paths, gates

    def forward(self, n, np, x_prob, xp):
        context = self.contexts(n, np, x_prob, xp)
        paths, gates = self.path_probabilities(context)
        latent = torch.einsum("bqtl,tld->bqtd", paths, self.leaf).sum(2)
        return self.decoder(latent), {"paths": paths, "gates": gates, "contexts": context}


class Model(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.config = Config(spec)
        self.forest = SoftObliviousForest()

    @staticmethod
    def parse(ids, mask):
        digit = ids.ge(DIGIT) & ids.lt(DIGIT + 10) & mask
        marker = ids.eq(N).long() + 2 * ids.eq(X).long() + 3 * ids.eq(T).long()
        role = torch.cummax(marker, 1).values * digit.long()
        index = torch.arange(ids.shape[1], device=ids.device)
        same = role[:, :, None].eq(role[:, None, :]) & role[:, :, None].gt(0)
        place = (same & (index[None, None, :] > index[None, :, None])
                 & digit[:, None, :]).sum(-1)
        value = (ids - DIGIT).clamp(0, 9)
        slots = torch.arange(PLACES, device=ids.device)

        def field(which):
            selected = role[:, :, None].eq(which) & place[:, :, None].eq(slots)
            return (value[:, :, None] * selected).sum(1).long(), selected.any(1)

        n, np = field(1)
        x, xp = field(2)
        steps = (value * torch.pow(ids.new_tensor(10), place) * role.eq(3)).sum(1).long()
        return n, np, x, xp, steps

    def forward(self, ids, attention_mask=None):
        batch, length = ids.shape
        mask = ids.ne(PAD) if attention_mask is None else attention_mask.bool()
        n, np, x, xp, steps = self.parse(ids, mask)
        nw, xw = np.sum(1), xp.sum(1)
        if length > self.config.max_seq_len or bool(((nw < 1) | (nw > PLACES) |
                (xw < 1) | (xw > PLACES) | (steps > MAX_T)).any()):
            raise ValueError("invalid Easy contract")
        probability = F.one_hot(x, 10).to(self.forest.digit.weight.dtype)
        endpoint = probability.clamp_min(1e-8).log()
        detail = None
        runs = int(steps.max().item())
        for tick in range(runs):
            proposed, detail = self.forest(n, np, probability, xp)
            active = (steps > tick)[:, None, None]
            endpoint = torch.where(active, proposed, endpoint)
            probability = torch.where(active, proposed.softmax(-1), probability)

        slots = torch.arange(PLACES, device=ids.device)
        target = mask.sum(1)[:, None] - 1 - slots[None]
        active = slots[None] < nw[:, None]
        placement = F.one_hot(target.clamp(0, length - 1), length).to(endpoint.dtype)
        placement = placement * active[..., None]
        placed = torch.bmm(placement.transpose(1, 2), endpoint)
        occupied = placement.sum(1).bool()
        digit_logits = F.pad(placed, (DIGIT, self.config.vocab_size - DIGIT - 10), value=-1e4)
        logits = endpoint.new_full((batch, length, self.config.vocab_size), -1e4)
        return torch.where(occupied[..., None], digit_logits, logits), {
            "steps": steps, "macrosteps": runs, "forest": detail}


def build_model(spec):
    model = Model(spec)
    actual = assert_model_state(model, spec)
    if actual != STATE_ELEMENTS:
        raise RuntimeError(f"state drift: {actual}")
    return model


def build_optimizer(model, spec):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        (decay if parameter.ndim >= 2 and "digit" not in name else no_decay).append(parameter)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": .01}, {"params": no_decay, "weight_decay": 0.}],
        lr=8e-4, betas=(.9, .95), eps=1e-8, capturable=spec.device_type == "cuda")
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min((step + 1) / 16, 1.0))
    return OptimizerBundle(optimizer, scheduler)


SUBMISSION = Submission(build_model, build_optimizer, batch_size=256,
                        eval_batch_size=512, max_steps=None)
