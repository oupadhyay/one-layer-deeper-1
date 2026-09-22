"""C1A local diagnostic: frozen C1 model with scheduler warmup 200 -> 20 only."""

import torch
from benchmark import OptimizerBundle, Submission
import competition_submission as c1


def build_optimizer(model, spec):
    decay, no_decay = [], []
    for parameter in model.parameters():
        (decay if parameter.ndim > 1 else no_decay).append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 0.02},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=8e-4,
        betas=(0.9, 0.98),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(1.0, (step + 1) / 20),
    )
    return OptimizerBundle(optimizer, scheduler)


SUBMISSION = Submission(
    build_model=c1.build_model,
    build_optimizer=build_optimizer,
    batch_size=64,
    eval_batch_size=128,
    max_steps=9500,
)
