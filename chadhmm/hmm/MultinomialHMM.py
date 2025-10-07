import torch
from torch.distributions import Multinomial

from typing import Optional
from chadhmm.hmm.BaseHMM import BaseHMM
from chadhmm.utils import constraints
from chadhmm.schemas import Transitions, ContextualVariables


class MultinomialHMM(BaseHMM):
    """
    Multinomial Hidden Markov Model (HMM)
    ----------
    Hidden Markov model with multinomial (discrete) emissions. This model is a
    special case of the HSMM model with a geometric duration distribution.

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
        Number of hidden states in the model. Defines shape of Transition
        Matrix (n_states,n_states)
    n_features (int):
        Number of emissions in the model. Defines the shape of Emisison
        Matrix (n_states,n_features)
    n_trials (int):
        Number of trials in the model
    transitions (Transitions)
        Represents the type of the transition matrix in HMM
            If 'ergodic'
                all states must be reachable from any state
            If 'left-to-right'
                may only transition to current or next state - remains in last
                state if reached
    alpha (float):
        Dirichlet concentration parameter for the prior over initial
        distribution, transition amd emission probabilities.
            Default to 1 thus samples from Uniform distribution
    seed (int):
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_states: int,
        n_features: int,
        transitions: Transitions,
        n_trials: int = 1,
        alpha: float = 1.0,
        seed: Optional[int] = None,
    ):
        self.n_features = n_features
        self.n_trials = n_trials
        super().__init__(n_states, transitions, alpha, seed)

    @property
    def pdf(self) -> Multinomial:
        return self._params.emission_pdf

    @property
    def dof(self):
        return self.n_states**2 + self.n_states * self.n_features - self.n_states - 1

    def sample_emission_pdf(self, X: Optional[torch.Tensor] = None) -> Multinomial:
        if X is not None:
            emission_freqs = torch.bincount(X) / X.shape[0]
            emission_matrix = torch.log(emission_freqs.expand(self.n_states, -1))
        else:
            emission_matrix = torch.log(
                constraints.sample_probs(self.alpha, (self.n_states, self.n_features))
            )

        return Multinomial(total_count=self.n_trials, logits=emission_matrix)

    def _estimate_emission_pdf(self, X: torch.Tensor, posterior: torch.Tensor, theta: Optional[ContextualVariables] = None):
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
                "Contextualized emissions not implemented for MultinomialEmissions"
            )
        else:
            new_B = posterior.T @ X
            new_B /= posterior.T.sum(1, keepdim=True)

        return new_B
