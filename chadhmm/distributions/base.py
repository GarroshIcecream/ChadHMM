from abc import ABC, abstractmethod

import torch


class BaseDistribution(ABC):
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
    def _update_posterior(self, X: torch.Tensor, posterior: torch.Tensor) -> None:
        """Update distribution parameters from posterior probabilities."""
        pass
