from abc import ABC, abstractmethod

import torch

from chadhmm.schemas import ContextualVariables


class BaseDistribution(ABC):
    """
    Base class for all emission distributions in HMM/HSMM models.

    All distribution classes must implement:
    - sample_distribution: Initialize distribution parameters
    - from_posterior_distribution: Update distribution from posterior probabilities
    - dof: Return degrees of freedom for this distribution
    - log_prob: Compute log probability
    - sample: Sample from the distribution
    - batch_shape: Shape of the batch dimensions
    - event_shape: Shape of the event dimensions
    """

    @property
    @abstractmethod
    def dof(self) -> int:
        """Return the degrees of freedom for this distribution."""
        pass

    @property
    @abstractmethod
    def n_components(self) -> int:
        """Number of components."""
        pass

    @property
    @abstractmethod
    def n_features(self) -> int:
        """Number of features."""
        pass

    @property
    @abstractmethod
    def batch_shape(self) -> torch.Size:
        """Shape of the batch dimensions (typically (n_components,))."""
        pass

    @property
    @abstractmethod
    def event_shape(self) -> torch.Size:
        """Shape of the event dimensions (typically (n_features,))."""
        pass

    @classmethod
    @abstractmethod
    def sample_distribution(
        cls, n_components: int, n_features: int, X: torch.Tensor | None = None, **kwargs
    ) -> "BaseDistribution":
        """Sample/initialize distribution parameters."""
        pass

    @abstractmethod
    def _update_posterior(
        self,
        X: torch.Tensor,
        posterior: torch.Tensor,
        theta: ContextualVariables | None = None,
    ) -> None:
        """Update distribution parameters from posterior probabilities."""
        pass

    @abstractmethod
    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Compute log probability."""
        pass

    @abstractmethod
    def sample(self, sample_shape: torch.Size) -> torch.Tensor:
        """Sample from the distribution."""
        pass

    def map_emission(self, x: torch.Tensor) -> torch.Tensor:
        """Get emission probabilities for a given sequence of observations."""
        pdf_shape = self.batch_shape + self.event_shape
        b_size = torch.Size([x.shape[0]]) + pdf_shape
        x_batched = x.unsqueeze(-len(pdf_shape)).expand(b_size)
        return self.log_prob(x_batched).squeeze()
