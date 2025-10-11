import torch
from torch.distributions import Independent, Poisson

from chadhmm.distributions.base import BaseDistribution
from chadhmm.schemas import ContextualVariables


class PoissonDistribution(Independent, BaseDistribution):
    """
    Poisson distribution for HMM/HSMM models.

    Each state has independent Poisson distributions for each feature.

    Parameters:
    ----------
    rates : torch.Tensor
        Rate parameters of shape (n_states, n_features).
    """

    def __init__(self, rates: torch.Tensor):
        self.rates = rates
        super().__init__(Poisson(rates), 1)

    @property
    def rates(self) -> torch.Tensor:
        """Access the rate parameters."""
        return self.base_dist.rate

    @property
    def dof(self) -> int:
        """
        Degrees of freedom: total number of rate parameters.
        Each state has n_features independent rate parameters.
        """
        return self.rates.numel()

    @classmethod
    def sample_distribution(
        cls,
        n_components: int,
        n_features: int,
        X: torch.Tensor | None = None,
    ) -> "PoissonDistribution":
        """
        Sample a Poisson distribution.

        If X is provided, initialize from empirical means.
        Otherwise, initialize with rates of 1.0.
        """
        if X is not None:
            # Initialize from data mean
            rates = X.mean(dim=0).expand(n_components, -1).clone()
        else:
            # Default initialization
            rates = torch.ones(size=(n_components, n_features), dtype=torch.float64)

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

        # Compute rates: E[X | state]
        new_rates = posterior.T @ X  # (n_states, n_features)
        new_rates /= posterior.T.sum(1, keepdim=True)

        return cls(rates=new_rates)
