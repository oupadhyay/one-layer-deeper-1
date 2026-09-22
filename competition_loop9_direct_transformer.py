"""Four-block direct Transformer control with learned decimal output queries."""

import torch
from torch import nn
from benchmark import Submission, assert_model_state
import competition_submission_c1a as c1a

PAD, N, X, T, ANS, DIGIT = 0, 2, 3, 4, 5, 7
D_MODEL, HEADS, BLOCKS, FF = 128, 4, 4, 512


class Config:
    def __init__(self, spec):
        self.vocab_size = spec.vocab_size
        self.max_seq_len = spec.max_seq_len


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(D_MODEL)
        self.attention = nn.MultiheadAttention(
            D_MODEL, HEADS, dropout=0.0, batch_first=True
        )
        self.norm2 = nn.LayerNorm(D_MODEL)
        self.ff = nn.Sequential(
            nn.Linear(D_MODEL, FF), nn.GELU(), nn.Linear(FF, D_MODEL)
        )

    def forward(self, tokens, padding_mask):
        normalized = self.norm1(tokens)
        tokens = tokens + self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=padding_mask,
            need_weights=False,
        )[0]
        return tokens + self.ff(self.norm2(tokens))


class Model(nn.Module):
    """Directly map the complete prompt to all decimal output slots once."""

    num_loops = 1

    def __init__(self, spec):
        super().__init__()
        self.config = Config(spec)
        self.token_embedding = nn.Embedding(spec.vocab_size, D_MODEL, padding_idx=PAD)
        self.position_embedding = nn.Embedding(spec.max_seq_len, D_MODEL)
        self.output_queries = nn.Embedding(spec.max_seq_len, D_MODEL)
        self.role_embedding = nn.Embedding(2, D_MODEL)
        self.blocks = nn.ModuleList(Block() for _ in range(BLOCKS))
        self.readout = nn.Linear(D_MODEL, 10)

    @staticmethod
    def parse_steps(input_ids, mask):
        is_digit = (input_ids >= DIGIT) & mask
        markers = ((input_ids == N) | (input_ids == X) |
                   (input_ids == T) | (input_ids == ANS))
        role = torch.cumsum(markers.long(), 1) * is_digit
        index = torch.arange(input_ids.shape[1], device=input_ids.device)
        places = ((role[:, :, None] == role[:, None, :]) &
                  (index[None, None] > index[None, :, None]) &
                  is_digit[:, None]).sum(2)
        t_digits = is_digit & (role == 3)
        return (((input_ids - DIGIT) * input_ids.new_tensor(10).pow(places)) *
                t_digits).sum(1)

    def debug_execution(self, input_ids, attention_mask=None):
        mask = input_ids.ne(PAD) if attention_mask is None else attention_mask.bool()
        steps = self.parse_steps(input_ids, mask)
        return {"parsed_steps": steps,
                "direct_mapping_calls": torch.ones_like(steps)}

    def forward(self, input_ids, attention_mask=None):
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("sequence exceeds max_seq_len")
        mask = input_ids.ne(PAD) if attention_mask is None else attention_mask.bool()
        batch, length = input_ids.shape
        positions = torch.arange(length, device=input_ids.device)
        prompt = (self.token_embedding(input_ids) +
                  self.position_embedding(positions)[None] +
                  self.role_embedding.weight[0])
        slots = torch.arange(self.config.max_seq_len, device=input_ids.device)
        queries = (self.output_queries(slots)[None].expand(batch, -1, -1) +
                   self.role_embedding.weight[1])
        tokens = torch.cat((prompt, queries), 1)
        padding = torch.cat(
            (~mask, torch.zeros(batch, self.config.max_seq_len,
                                dtype=torch.bool, device=input_ids.device)), 1
        )
        for block in self.blocks:
            tokens = block(tokens, padding)
        endpoint = self.readout(tokens[:, length:])
        logits = endpoint.new_full((batch, length, self.config.vocab_size), -1e4)
        output_slot = mask.sum(1)[:, None] - 1 - positions[None]
        selected = endpoint.gather(
            1,
            output_slot.clamp(0, self.config.max_seq_len - 1)[:, :, None]
            .expand(-1, -1, 10),
        )
        logits[:, :, DIGIT:DIGIT + 10] = selected
        return logits, {
            "parsed_steps": self.parse_steps(input_ids, mask),
            "direct_mapping_calls": input_ids.new_ones(batch),
            "output_slot_logits": endpoint,
        }


def build_model(spec):
    model = Model(spec)
    assert_model_state(model, spec)
    if len(model.blocks) != BLOCKS:
        raise RuntimeError("direct Transformer block count drift")
    return model


build_optimizer = c1a.build_optimizer
SUBMISSION = Submission(
    build_model, build_optimizer, batch_size=64, eval_batch_size=128, max_steps=1600
)
