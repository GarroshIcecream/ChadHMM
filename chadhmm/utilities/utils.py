from typing import Tuple, List
from dataclasses import dataclass, field
import torch


@dataclass
class Observations:
    sequence: List[torch.Tensor]
    log_probs: List[torch.Tensor]
    lengths: List[int]

@dataclass
class ContextualVariables:
    n_context: int
    X: Tuple[torch.Tensor,...]
    time_dependent: bool = field(default=False)
