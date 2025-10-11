from abc import ABC, abstractmethod

import torch
from torch.distributions import Distribution

from chadhmm.schemas import ContextualVariables

class BaseDistribution(ABC, Distribution):
    """
    Base class for all emission distributions in HMM/HSMM models.

    All distribution classes must implement:
    - sample_distribution: Initialize distribution parameters
    - from_posterior_distribution: Update distribution from posterior probabilities
    - dof: Return degrees of freedom for this distribution
    """

    @property
    @abstractmethod
    def dof(self) -> int:
        """Return the degrees of freedom for this distribution."""
        pass

    @classmethod
    @abstractmethod
    def sample_distribution(
        cls, n_components: int, n_features: int, X: torch.Tensor | None = None, **kwargs
    ) -> "BaseDistribution":
        """Sample/initialize distribution parameters."""
        pass

    @abstractmethod
    def _update_posterior(self, X: torch.Tensor, posterior: torch.Tensor, theta: ContextualVariables | None = None) -> None:
        """Update distribution parameters from posterior probabilities."""
        pass

    def map_emission(self, x: torch.Tensor) -> torch.Tensor:
        """Get emission probabilities for a given sequence of observations."""
        pdf_shape = self.batch_shape + self.event_shape
        b_size = torch.Size([x.shape[0]]) + pdf_shape
        x_batched = x.unsqueeze(-len(pdf_shape)).expand(b_size)
        return self.log_prob(x_batched).squeeze()
