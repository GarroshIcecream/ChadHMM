import torch
import torch.nn as nn
from torch.distributions import Multinomial

from chadhmm.distributions.base import BaseDistribution
from chadhmm.schemas import ContextualVariables
from chadhmm.utils import constraints


class MultinomialDistribution(nn.Module, BaseDistribution):
    """
    Multinomial distribution for HMM/HSMM models.

    Handles categorical, binomial, and multinomial emissions.
    - If n_trials=1 and n_features=2: Bernoulli
    - If n_trials=1 and n_features>2: Categorical
    - If n_trials>1 and n_features=2: Binomial
    - If n_trials>1 and n_features>2: Multinomial

    Parameters:
    ----------
    logits : torch.Tensor
        Log probabilities of shape (n_states, n_features).
    n_trials : int
        Number of trials for the multinomial distribution.
    """

    def __init__(self, logits: torch.Tensor, n_trials: int = 1):
        super().__init__()
        self._logits = nn.Parameter(logits.clone(), requires_grad=False)
        self.n_trials = n_trials

    @property
    def logits(self) -> torch.Tensor:
        """Log probabilities."""
        return self._logits

    @property
    def probs(self) -> torch.Tensor:
        """Probabilities (normalized)."""
        return torch.softmax(self._logits, dim=-1)

    @property
    def n_components(self) -> int:
        """Number of components."""
        return self._logits.shape[0]

    @property
    def n_features(self) -> int:
        """Number of features."""
        return self._logits.shape[1]

    @property
    def dof(self) -> int:
        """
        Degrees of freedom: (n_states * n_features) - n_states
        We subtract n_states because probabilities in each row must sum to 1.
        """
        n_states, n_features = self._logits.shape
        return n_states * n_features - n_states

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
        return Multinomial(total_count=self.n_trials, logits=self._logits).log_prob(
            value
        )

    def sample(self, sample_shape: torch.Size) -> torch.Tensor:
        """Sample from the distribution."""
        return Multinomial(total_count=self.n_trials, logits=self._logits).sample(
            sample_shape
        )

    @classmethod
    def sample_distribution(
        cls,
        n_components: int,
        n_features: int,
        X: torch.Tensor | None = None,
        n_trials: int = 1,
        alpha: float = 1.0,
        **kwargs,
    ) -> "MultinomialDistribution":
        """
        Sample a multinomial distribution.

        If X is provided, initialize from empirical frequencies.
        Otherwise, sample from Dirichlet prior.
        """
        if X is not None:
            emission_freqs = torch.bincount(X.long(), minlength=n_features) / X.shape[0]
            emission_matrix = torch.log(emission_freqs.expand(n_components, -1).clone())
        else:
            emission_matrix = torch.log(
                constraints.sample_probs(alpha, (n_components, n_features))
            )

        return cls(logits=emission_matrix, n_trials=n_trials)

    @classmethod
    def from_posterior_distribution(
        cls,
        X: torch.Tensor,
        posterior: torch.Tensor,
        n_trials: int = 1,
        theta: ContextualVariables | None = None,
    ) -> "MultinomialDistribution":
        """
        Estimate emission probabilities from data and posterior.

        Parameters:
        -----------
        X : torch.Tensor
            Observed data
        posterior : torch.Tensor
            Posterior probabilities of shape (n_samples, n_states)
        n_trials : int
            Number of trials
        theta : ContextualVariables | None
            Contextual variables (not yet supported)

        Returns:
        --------
        MultinomialDistribution
            Estimated multinomial distribution
        """
        if theta is not None:
            raise NotImplementedError(
                "Contextualized emissions not implemented for MultinomialDistribution"
            )

        new_B = posterior.T @ X
        new_B /= posterior.T.sum(1, keepdim=True)

        return cls(logits=torch.log(new_B), n_trials=n_trials)

    def _update_posterior(
        self,
        X: torch.Tensor,
        posterior: torch.Tensor,
        theta: ContextualVariables | None = None,
    ) -> None:
        if theta is not None:
            raise NotImplementedError(
                "Contextualized emissions not implemented for MultinomialDistribution"
            )

        self._logits.data = torch.log(
            posterior.T @ X.to(posterior.dtype) / posterior.T.sum(1, keepdim=True)
        )
