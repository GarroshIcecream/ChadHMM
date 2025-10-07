from dataclasses import dataclass, field

import torch


@dataclass
class Observations:
    sequence: list[torch.Tensor]
    log_probs: list[torch.Tensor]
    lengths: list[int]


@dataclass
class ContextualVariables:
    n_context: int
    X: tuple[torch.Tensor, ...]
    time_dependent: bool = field(default=False)
