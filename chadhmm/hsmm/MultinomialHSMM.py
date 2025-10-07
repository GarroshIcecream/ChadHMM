import torch
from torch.distributions import Multinomial
from typing import Optional

from chadhmm.hsmm.BaseHSMM import BaseHSMM
from chadhmm.utils import constraints
from chadhmm.schemas import ContextualVariables


class MultinomialHSMM(BaseHSMM):
    """
    Categorical Hidden semi-Markov Model (HSMM)
    ----------
    Hidden semi-Markov model with categorical (discrete) emissions. This model
    is an extension of classical HMMs where the duration of each state is
    modeled by a geometric distribution.
    Duration in each state is modeled by a Categorical distribution with a
    fixed maximum duration.

    If n_trials = 1 and and n_features = 2
        Bernoulli distribution
    If n_trials = 1 and and n_features > 2
        Categorical distribution
    If n_trials > 1 and and n_features = 2
        Binomial distribution
    If n_trials > 1 and and n_features > 2
        Multionomial distribution

    Parameters:
    ----------
    n_states (int):
        Number of hidden states in the model.
    n_features (int):
        Number of emissions in the model.
    max_duration (int):
        Maximum duration of each state.
    n_trials (int):
        Number of trials to estimate the emission distribution.
    alpha (float):
        Dirichlet concentration parameter for the prior over initial state
        probabilities and transition probabilities.
    seed (int):
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_states: int,
        n_features: int,
        max_duration: int,
        n_trials: int = 1,
        alpha: float = 1.0,
        seed: Optional[int] = None,
    ):
        self.n_features = n_features
        self.n_trials = n_trials
        super().__init__(n_states, max_duration, alpha, seed)

    @property
    def dof(self):
        return self.n_states**2 + self.n_states * self.n_features - self.n_states - 1

    def sample_emission_pdf(self, X=None):
        if X is not None:
            emission_freqs = torch.bincount(X) / X.shape[0]
            emission_matrix = torch.log(emission_freqs.expand(self.n_states, -1))
        else:
            emission_matrix = torch.log(
                constraints.sample_probs(self.alpha, (self.n_states, self.n_features))
            )

        return Multinomial(total_count=self.n_trials, logits=emission_matrix)

    def _estimate_emission_pdf(self, X, posterior, theta=None):
        new_B = torch.log(self._compute_B(X, posterior, theta))
        return Multinomial(total_count=self.n_trials, logits=new_B)

    def _compute_B(
        self,
        X: torch.Tensor,
        posterior: torch.Tensor,
        theta: Optional[ContextualVariables] = None,
    ) -> torch.Tensor:
        """Compute the emission probabilities for each hidden state."""
        if theta is not None:
            # TODO: Implement contextualized emissions
            raise NotImplementedError(
                "Contextualized emissions not implemented for CategoricalEmissions"
            )
        else:
            new_B = posterior.T @ X
            new_B /= posterior.T.sum(1, keepdim=True)

        return new_B
