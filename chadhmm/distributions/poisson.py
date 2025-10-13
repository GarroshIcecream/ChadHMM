import torch
import torch.nn as nn
from torch.distributions import Independent, Poisson

from chadhmm.distributions.base import BaseDistribution
from chadhmm.schemas import ContextualVariables


class PoissonDistribution(nn.Module, BaseDistribution):
    """
    Poisson distribution for HMM/HSMM models.

    Each state has independent Poisson distributions for each feature.

    Parameters:
    ----------
    rates : torch.Tensor
        Rate parameters of shape (n_states, n_features).
    """

    def __init__(self, rates: torch.Tensor):
        super().__init__()
        self._rates = nn.Parameter(rates.clone(), requires_grad=False)

    @property
    def rate(self) -> torch.Tensor:
        """Rate parameter."""
        return self._rates

    @property
    def mean(self) -> torch.Tensor:
        """Mean of the distribution (equal to rate for Poisson)."""
        return self._rates

    @property
    def n_components(self) -> int:
        """Number of components."""
        return self._rates.shape[0]

    @property
    def n_features(self) -> int:
        """Number of features."""
        return self._rates.shape[1]

    @property
    def dof(self) -> int:
        return self._rates.numel()

    @property
    def batch_shape(self) -> torch.Size:
        """Batch shape: (n_components,)"""
        return torch.Size([self.n_components])

    @property
    def event_shape(self) -> torch.Size:
        """Event shape: (n_features,)"""
        return torch.Size([self.n_features])

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Compute log probability."""
        return Independent(Poisson(self._rates), 1).log_prob(value)

    def sample(self, sample_shape: torch.Size) -> torch.Tensor:
        """Sample from the distribution."""
        return Independent(Poisson(self._rates), 1).sample(sample_shape)

    @classmethod
    def sample_distribution(
        cls,
        n_components: int,
        n_features: int,
        X: torch.Tensor | None = None,
        **kwargs,
    ) -> "PoissonDistribution":
        """
        Sample a Poisson distribution.

        If X is provided, initialize from empirical means.
        Otherwise, initialize with rates of 1.0.
        """
        if X is not None:
            rates = X.mean(dim=0).expand(n_components, -1).clone()
        else:
            rates = torch.ones(size=(n_components, n_features))

        return cls(rates=rates)

    @classmethod
    def from_posterior_distribution(
        cls,
        X: torch.Tensor,
        posterior: torch.Tensor,
        theta: ContextualVariables | None = None,
    ) -> "PoissonDistribution":
        """
        Estimate Poisson rates from data and posterior.

        Parameters:
        -----------
        X : torch.Tensor
            Observed data
        posterior : torch.Tensor
            Posterior probabilities of shape (n_samples, n_states)
        theta : ContextualVariables | None
            Contextual variables (not yet supported)

        Returns:
        --------
        PoissonDistribution
            Estimated Poisson distribution
        """
        if theta is not None:
            raise NotImplementedError(
                "Contextualized emissions not implemented for PoissonDistribution"
            )

        new_rates = posterior.T @ X
        new_rates /= posterior.T.sum(1, keepdim=True)

        return cls(rates=new_rates)

    def _update_posterior(
        self,
        X: torch.Tensor,
        posterior: torch.Tensor,
        theta: ContextualVariables | None = None,
    ) -> None:
        """Update the posterior distribution."""
        new_rates = posterior.T @ X
        new_rates /= posterior.T.sum(1, keepdim=True)

        self._rates.data = new_rates
