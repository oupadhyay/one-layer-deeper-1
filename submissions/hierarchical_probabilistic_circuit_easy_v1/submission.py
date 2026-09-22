"""Generic tied hierarchical probabilistic circuit for Easy E5."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from benchmark import OptimizerBundle, Submission, assert_model_state

PAD, N, X, T, DIGIT = 0, 2, 3, 4, 7
PLACES, STATES, EMBED, HIDDEN, MAX_T = 4, 16, 40, 64, 64
STATE_ELEMENTS = 9_120


class Config:
    def __init__(self, spec):
        self.vocab_size, self.max_seq_len = spec.vocab_size, spec.max_seq_len


class SharedComposer(nn.Module):
    """One learned sum-product rule, tied across every node and level."""
    def __init__(self):
        super().__init__()
        self.log_rule = nn.Parameter(torch.empty(STATES, STATES, STATES))
        nn.init.normal_(self.log_rule, std=.02)

    def forward(self, left, right):
        score = (left[..., None, :, None] + right[..., None, None, :]
                 + self.log_rule)
        value = torch.logsumexp(score.flatten(-2), -1)
        return value - torch.logsumexp(value, -1, keepdim=True)


class Circuit(nn.Module):
    def __init__(self):
        super().__init__()
        self.digit = nn.Embedding(10, EMBED)
        self.place = nn.Embedding(PLACES, EMBED)
        self.role = nn.Embedding(2, EMBED)
        self.presence = nn.Embedding(2, EMBED)
        self.leaf = nn.Sequential(nn.Linear(EMBED, HIDDEN), nn.GELU(),
                                  nn.Linear(HIDDEN, STATES))
        self.composer = SharedComposer()
        self.output_queries = nn.Parameter(torch.empty(PLACES, 10, STATES))
        nn.init.normal_(self.output_queries, std=.02)

    def leaf_potentials(self, n, np, x_prob, xp):
        place = self.place.weight[None]
        ne = self.digit(n) + place + self.role.weight[0] + self.presence(np.long())
        xe = x_prob.to(self.digit.weight.dtype) @ self.digit.weight
        xe = xe + place + self.role.weight[1] + self.presence(xp.long())
        leaves = self.leaf(torch.stack((ne, xe), 2).flatten(1, 2))
        return leaves - torch.logsumexp(leaves, -1, keepdim=True)

    def root(self, leaves):
        nodes = leaves
        for _ in range(3):
            nodes = self.composer(nodes[:, 0::2], nodes[:, 1::2])
        return nodes[:, 0]

    def forward(self, n, np, x_prob, xp):
        leaves = self.leaf_potentials(n, np, x_prob, xp)
        root = self.root(leaves)
        logits = torch.logsumexp(root[:, None, None, :] + self.output_queries[None], -1)
        return logits, (leaves, root)


class Model(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.config = Config(spec)
        self.circuit = Circuit()

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
        # T alone is interpreted as decimal loop control; it never enters the circuit.
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
        prob = F.one_hot(x, 10).to(self.circuit.digit.weight.dtype)
        endpoint = prob.clamp_min(1e-8).log()
        detail = None
        runs = int(steps.max().item())
        for tick in range(runs):
            proposed, detail = self.circuit(n, np, prob, xp)
            active = (steps > tick)[:, None, None]
            endpoint = torch.where(active, proposed, endpoint)
            prob = torch.where(active, proposed.softmax(-1), prob)
        slots = torch.arange(PLACES, device=ids.device)
        target = mask.sum(1)[:, None] - 1 - slots[None]
        active = slots[None] < nw[:, None]
        placement = F.one_hot(target.clamp(0, length - 1), length).to(endpoint.dtype)
        placement = placement * active[..., None]
        placed = torch.bmm(placement.transpose(1, 2), endpoint)
        occupied = placement.sum(1).bool()
        logits = endpoint.new_full((batch, length, self.config.vocab_size), -1e4)
        digit_logits = F.pad(placed, (DIGIT, self.config.vocab_size - DIGIT - 10), value=-1e4)
        logits = torch.where(occupied[..., None], digit_logits, logits)
        return logits, {"steps": steps, "macrosteps": runs, "circuit": detail}


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
