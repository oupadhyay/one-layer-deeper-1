"""Generate the predeclared conditional-transducer optimizer screen."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "submissions" / "conditional_global_transducer_easy_v2_batch256" / "submission.py"

VARIANTS = [
    ("lr2", 256, "2e-4", ".01", "32"),
    ("lr3", 256, "3e-4", ".01", "32"),
    ("lr4", 256, "4e-4", ".01", "32"),
    ("lr5", 256, "5e-4", ".01", "32"),
    ("lr7", 256, "7e-4", ".01", "32"),
    ("lr8", 256, "8e-4", ".01", "32"),
    ("lr10", 256, "1e-3", ".01", "32"),
    ("lr12", 256, "1.2e-3", ".01", "32"),
    ("lr15", 256, "1.5e-3", ".01", "32"),
    ("wd0", 256, "6e-4", "0.", "32"),
    ("wd3", 256, "6e-4", ".03", "32"),
    ("wd10", 256, "6e-4", ".1", "32"),
    ("warm8", 256, "6e-4", ".01", "8"),
    ("warm16", 256, "6e-4", ".01", "16"),
    ("warm64", 256, "6e-4", ".01", "64"),
    ("batch128", 128, "6e-4", ".01", "32"),
]

# Phase 2 starts from the best balanced phase-1 setting (LR 1e-3) and
# changes one optimizer or exposure control per package.
PHASE2_VARIANTS = [
    ("lr9", 256, "9e-4", ".01", "32"),
    ("lr11", 256, "1.1e-3", ".01", "32"),
    ("lr10_batch128", 128, "1e-3", ".01", "32"),
    ("lr10_batch512", 512, "1e-3", ".01", "32"),
    ("lr10_wd0", 256, "1e-3", "0.", "32"),
    ("lr10_wd3", 256, "1e-3", ".03", "32"),
    ("lr10_wd10", 256, "1e-3", ".1", "32"),
    ("lr10_warm8", 256, "1e-3", ".01", "8"),
    ("lr10_warm16", 256, "1e-3", ".01", "16"),
    ("lr10_warm64", 256, "1e-3", ".01", "64"),
]

# Phase 3 starts from the phase-2 exact-accuracy winner (LR 1e-3,
# eight warmup steps). Each package changes one control from that parent.
PHASE3_VARIANTS = [
    ("warm8_lr8", 256, "8e-4", ".01", "8", "(.9,.95)"),
    ("warm8_lr9", 256, "9e-4", ".01", "8", "(.9,.95)"),
    ("warm8_lr11", 256, "1.1e-3", ".01", "8", "(.9,.95)"),
    ("warm8_lr12", 256, "1.2e-3", ".01", "8", "(.9,.95)"),
    ("warm8_batch128", 128, "1e-3", ".01", "8", "(.9,.95)"),
    ("warm8_batch512", 512, "1e-3", ".01", "8", "(.9,.95)"),
    ("warm8_wd0", 256, "1e-3", "0.", "8", "(.9,.95)"),
    ("warm8_wd3", 256, "1e-3", ".03", "8", "(.9,.95)"),
    ("warm8_wd10", 256, "1e-3", ".1", "8", "(.9,.95)"),
    ("warm8_beta98", 256, "1e-3", ".01", "8", "(.9,.98)"),
]

PHASE4_LOSSES = {
    "smooth02": "return F.cross_entropy(batch.logits.transpose(1,2),batch.labels,ignore_index=-100,label_smoothing=.02)",
    "smooth05": "return F.cross_entropy(batch.logits.transpose(1,2),batch.labels,ignore_index=-100,label_smoothing=.05)",
    "smooth10": "return F.cross_entropy(batch.logits.transpose(1,2),batch.labels,ignore_index=-100,label_smoothing=.1)",
    "focal05": "ce=F.cross_entropy(batch.logits.transpose(1,2),batch.labels,ignore_index=-100,reduction='none');v=batch.valid_mask;return (ce*(1-torch.exp(-ce)).pow(.5)*v).sum()/v.sum()",
    "focal10": "ce=F.cross_entropy(batch.logits.transpose(1,2),batch.labels,ignore_index=-100,reduction='none');v=batch.valid_mask;return (ce*(1-torch.exp(-ce))*v).sum()/v.sum()",
    "focal20": "ce=F.cross_entropy(batch.logits.transpose(1,2),batch.labels,ignore_index=-100,reduction='none');v=batch.valid_mask;return (ce*(1-torch.exp(-ce)).square()*v).sum()/v.sum()",
    "hard25": "ce=F.cross_entropy(batch.logits.transpose(1,2),batch.labels,ignore_index=-100,reduction='none');v=batch.valid_mask;mean=(ce*v).sum()/v.sum();w=ce.masked_fill(~v,-1e4).softmax(1);return mean+.25*(w*ce).sum(1).mean()",
    "hard50": "ce=F.cross_entropy(batch.logits.transpose(1,2),batch.labels,ignore_index=-100,reduction='none');v=batch.valid_mask;mean=(ce*v).sum()/v.sum();w=ce.masked_fill(~v,-1e4).softmax(1);return mean+.5*(w*ce).sum(1).mean()",
}

PHASE5_VARIANTS = [
    ("focal075", "ce=F.cross_entropy(batch.logits.transpose(1,2),batch.labels,ignore_index=-100,reduction='none');v=batch.valid_mask;return (ce*(1-torch.exp(-ce)).pow(.75)*v).sum()/v.sum()", 256, "1e-3", ".01", "8"),
    ("focal125", "ce=F.cross_entropy(batch.logits.transpose(1,2),batch.labels,ignore_index=-100,reduction='none');v=batch.valid_mask;return (ce*(1-torch.exp(-ce)).pow(1.25)*v).sum()/v.sum()", 256, "1e-3", ".01", "8"),
    ("focal150", "ce=F.cross_entropy(batch.logits.transpose(1,2),batch.labels,ignore_index=-100,reduction='none');v=batch.valid_mask;return (ce*(1-torch.exp(-ce)).pow(1.5)*v).sum()/v.sum()", 256, "1e-3", ".01", "8"),
    ("hard35", "ce=F.cross_entropy(batch.logits.transpose(1,2),batch.labels,ignore_index=-100,reduction='none');v=batch.valid_mask;mean=(ce*v).sum()/v.sum();w=ce.masked_fill(~v,-1e4).softmax(1);return mean+.35*(w*ce).sum(1).mean()", 256, "1e-3", ".01", "8"),
    ("hard75", "ce=F.cross_entropy(batch.logits.transpose(1,2),batch.labels,ignore_index=-100,reduction='none');v=batch.valid_mask;mean=(ce*v).sum()/v.sum();w=ce.masked_fill(~v,-1e4).softmax(1);return mean+.75*(w*ce).sum(1).mean()", 256, "1e-3", ".01", "8"),
    ("hard100", "ce=F.cross_entropy(batch.logits.transpose(1,2),batch.labels,ignore_index=-100,reduction='none');v=batch.valid_mask;mean=(ce*v).sum()/v.sum();w=ce.masked_fill(~v,-1e4).softmax(1);return mean+(w*ce).sum(1).mean()", 256, "1e-3", ".01", "8"),
    ("focal05_lr9", PHASE4_LOSSES["focal05"], 256, "9e-4", ".01", "8"),
    ("focal05_lr11", PHASE4_LOSSES["focal05"], 256, "1.1e-3", ".01", "8"),
    ("focal05_batch128", PHASE4_LOSSES["focal05"], 128, "1e-3", ".01", "8"),
    ("focal05_batch512", PHASE4_LOSSES["focal05"], 512, "1e-3", ".01", "8"),
    ("focal05_wd0", PHASE4_LOSSES["focal05"], 256, "1e-3", "0.", "8"),
    ("focal05_warm16", PHASE4_LOSSES["focal05"], 256, "1e-3", ".01", "16"),
]


def main() -> None:
    base = BASE.read_text(encoding="utf-8")
    generated = []
    for name, batch, lr, decay, warmup in VARIANTS:
        source = base.replace(
            '"""Global contextual N-conditioned transducer with more optimizer updates."""',
            f'"""Conditional optimizer screen: {name}."""',
        )
        source = source.replace("'weight_decay':.01", f"'weight_decay':{decay}")
        source = source.replace("lr=6e-4", f"lr={lr}")
        source = source.replace("(u+1)/32", f"(u+1)/{warmup}")
        source = source.replace("batch_size=256", f"batch_size={batch}")
        target = ROOT / "submissions" / f"conditional_opt_{name}_easy_v1" / "submission.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source, encoding="utf-8")
        generated.append(str(target.relative_to(ROOT)))
    (ROOT / "artifacts" / "conditional_optimizer_batch_manifest.json").write_text(
        json.dumps({"variants": generated}, indent=2), encoding="utf-8"
    )

    phase2_generated = []
    for name, batch, lr, decay, warmup in PHASE2_VARIANTS:
        source = base.replace(
            '"""Global contextual N-conditioned transducer with more optimizer updates."""',
            f'"""Conditional optimizer phase 2: {name}."""',
        )
        source = source.replace("'weight_decay':.01", f"'weight_decay':{decay}")
        source = source.replace("lr=6e-4", f"lr={lr}")
        source = source.replace("(u+1)/32", f"(u+1)/{warmup}")
        source = source.replace("batch_size=256", f"batch_size={batch}")
        target = ROOT / "submissions" / f"conditional_opt2_{name}_easy_v1" / "submission.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source, encoding="utf-8")
        phase2_generated.append(str(target.relative_to(ROOT)))
    (ROOT / "artifacts" / "conditional_optimizer_phase2_manifest.json").write_text(
        json.dumps({"variants": phase2_generated}, indent=2), encoding="utf-8"
    )

    phase3_generated = []
    for name, batch, lr, decay, warmup, betas in PHASE3_VARIANTS:
        source = base.replace(
            '"""Global contextual N-conditioned transducer with more optimizer updates."""',
            f'"""Conditional optimizer phase 3: {name}."""',
        )
        source = source.replace("'weight_decay':.01", f"'weight_decay':{decay}")
        source = source.replace("lr=6e-4", f"lr={lr}")
        source = source.replace("betas=(.9,.95)", f"betas={betas}")
        source = source.replace("(u+1)/32", f"(u+1)/{warmup}")
        source = source.replace("batch_size=256", f"batch_size={batch}")
        target = ROOT / "submissions" / f"conditional_opt3_{name}_easy_v1" / "submission.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source, encoding="utf-8")
        phase3_generated.append(str(target.relative_to(ROOT)))
    (ROOT / "artifacts" / "conditional_optimizer_phase3_manifest.json").write_text(
        json.dumps({"variants": phase3_generated}, indent=2), encoding="utf-8"
    )

    phase4_generated = []
    for name, loss_body in PHASE4_LOSSES.items():
        source = base.replace(
            '"""Global contextual N-conditioned transducer with more optimizer updates."""',
            f'"""Conditional structured-loss screen: {name}."""',
        )
        source = source.replace(
            "OptimizerBundle,Submission,assert_model_state",
            "OptimizerBundle,Submission,TokenLossBatch,assert_model_state",
        )
        source = source.replace("lr=6e-4", "lr=1e-3")
        source = source.replace("(u+1)/32", "(u+1)/8")
        source = source.replace(
            "SUBMISSION=Submission(build_model,build_optimizer,batch_size=256,eval_batch_size=512,max_steps=None)",
            f"def token_training_loss(batch:TokenLossBatch):\n {loss_body}\n"
            "SUBMISSION=Submission(build_model,build_optimizer,batch_size=256,eval_batch_size=512,max_steps=None,token_training_loss=token_training_loss)",
        )
        target = ROOT / "submissions" / f"conditional_loss_{name}_easy_v1" / "submission.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source, encoding="utf-8")
        phase4_generated.append(str(target.relative_to(ROOT)))
    (ROOT / "artifacts" / "conditional_loss_phase4_manifest.json").write_text(
        json.dumps({"variants": phase4_generated}, indent=2), encoding="utf-8"
    )

    phase5_generated = []
    for name, loss_body, batch, lr, decay, warmup in PHASE5_VARIANTS:
        source = base.replace(
            '"""Global contextual N-conditioned transducer with more optimizer updates."""',
            f'"""Conditional structured-loss phase 5: {name}."""',
        )
        source = source.replace(
            "OptimizerBundle,Submission,assert_model_state",
            "OptimizerBundle,Submission,TokenLossBatch,assert_model_state",
        )
        source = source.replace("'weight_decay':.01", f"'weight_decay':{decay}")
        source = source.replace("lr=6e-4", f"lr={lr}")
        source = source.replace("(u+1)/32", f"(u+1)/{warmup}")
        source = source.replace("batch_size=256", f"batch_size={batch}")
        source = source.replace(
            "SUBMISSION=Submission(build_model,build_optimizer,batch_size="
            f"{batch},eval_batch_size=512,max_steps=None)",
            f"def token_training_loss(batch:TokenLossBatch):\n {loss_body}\n"
            f"SUBMISSION=Submission(build_model,build_optimizer,batch_size={batch},eval_batch_size=512,max_steps=None,token_training_loss=token_training_loss)",
        )
        target = ROOT / "submissions" / f"conditional_loss2_{name}_easy_v1" / "submission.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source, encoding="utf-8")
        phase5_generated.append(str(target.relative_to(ROOT)))
    (ROOT / "artifacts" / "conditional_loss_phase5_manifest.json").write_text(
        json.dumps({"variants": phase5_generated}, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
