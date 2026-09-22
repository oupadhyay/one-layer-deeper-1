"""Generate the predeclared dynamic conditional T1-weight Easy screen."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "submissions" / "conditional_dynamic_hard_v1" / "submission.py"
VARIANTS = (
    ("weight2", 96, 48, 2, True),
    ("weight4", 96, 48, 4, True),
    ("weight8", 96, 48, 8, True),
    ("weight16", 96, 48, 16, True),
    ("weight4_plain", 96, 48, 4, False),
    ("weight4_d64", 64, 32, 4, True),
)


def loss_source(weight: int, focal: bool) -> str:
    focus = "ce*(1-torch.exp(-ce)).pow(.5)" if focal else "ce"
    return (
        "def token_training_loss(batch:TokenLossBatch):\n"
        f" ce=F.cross_entropy(batch.logits.transpose(1,2),batch.labels,ignore_index=-100,reduction='none');v=batch.valid_mask;seq=({focus}*v).sum(1)/v.sum(1).clamp_min(1);w=torch.where(batch.auxiliary['steps'].eq(1),seq.new_tensor({weight}.),seq.new_tensor(1.));return (seq*w).sum()/w.sum()\n"
    )


def main() -> None:
    base = BASE.read_text(encoding="utf-8")
    old_loss = (
        "def token_training_loss(batch:TokenLossBatch):\n"
        " ce=F.cross_entropy(batch.logits.transpose(1,2),batch.labels,ignore_index=-100,reduction='none');v=batch.valid_mask;return (ce*(1-torch.exp(-ce)).pow(.5)*v).sum()/v.sum()\n"
    )
    for name, d_model, hidden, weight, focal in VARIANTS:
        source = base.replace(
            '"""Conditional global transducer scale batch: dynamic_hard."""',
            f'"""Dynamic conditional T1-weight screen: {name}."""',
        )
        source = source.replace("W,D,H=None,96,48", f"W,D,H=None,{d_model},{hidden}")
        source = source.replace(old_loss, loss_source(weight, focal))
        if source == base or old_loss in source:
            raise RuntimeError(f"generation failed for {name}")
        target = ROOT / "submissions" / f"conditional_t1_{name}_easy_v1" / "submission.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source, encoding="utf-8")


if __name__ == "__main__":
    main()
