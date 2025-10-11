import torch
from torch.distributions import Multinomial

from chadhmm.distributions.base import BaseDistribution
from chadhmm.schemas import ContextualVariables
from chadhmm.utils import constraints


class MultinomialDistribution(Multinomial, BaseDistribution):
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
        self.n_trials = n_trials
        super().__init__(total_count=n_trials, logits=logits)

    @property
    def dof(self) -> int:
        """
        Degrees of freedom: (n_states * n_features) - n_states
        We subtract n_states because probabilities in each row must sum to 1.
        """
        n_states, n_features = self.logits.shape
        return n_states * n_features - n_states

    @classmethod
    def sample_distribution(
        cls,
        n_components: int,
        n_features: int,
        n_trials: int = 1,
        X: torch.Tensor | None = None,
        alpha: float = 1.0,
    ) -> "MultinomialDistribution":
        """
        Sample a multinomial distribution.

        If X is provided, initialize from empirical frequencies.
        Otherwise, sample from Dirichlet prior.
        """
        if X is not None:
            # Initialize from data frequencies
            emission_freqs = torch.bincount(X.long(), minlength=n_features) / X.shape[0]
            emission_matrix = torch.log(emission_freqs.expand(n_components, -1).clone())
        else:
            # Sample from Dirichlet
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

        new_logits = torch.log(posterior.T @ X.to(posterior.dtype) / posterior.T.sum(1, keepdim=True))
        Multinomial.__init__(self, total_count=self.n_trials, logits=new_logits)
