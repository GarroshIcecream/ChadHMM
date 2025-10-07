import torch
from typing import Optional


class SeedGenerator:
    def __init__(self, seed: Optional[int] = None):
        if seed:
            self.seed_gen = torch.random.manual_seed(seed)
        else:
            initial_seed = torch.random.seed()
            self.seed_gen = torch.random.manual_seed(initial_seed)

    def __repr__(self) -> str:
        return f"SeedGenerator(seed={self()})"

    def __call__(self) -> int:
        return self.seed_gen.seed()

    @property
    def seed(self) -> int:
        return self.seed_gen.initial_seed()

    @seed.setter
    def seed(self, seed: int) -> None:
        self.seed_gen = torch.random.manual_seed(seed)
