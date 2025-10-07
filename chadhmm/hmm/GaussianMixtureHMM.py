import torch
from sklearn.cluster import KMeans
from torch.distributions import Categorical, MixtureSameFamily, MultivariateNormal
from typing import Optional

from chadhmm.hmm.BaseHMM import BaseHMM
from chadhmm.utils import constraints
from chadhmm.schemas import (
    Transitions, 
    ContextualVariables, 
    CovarianceType
)


class GaussianMixtureHMM(BaseHMM):
    """
    Gaussian Hidden Markov Model (Gaussian HMM)
    ----------
    This model assumes that the data follows a multivariate Gaussian distribution.
    The model parameters (initial state probabilities, transition probabilities,
    emission means, and emission covariances) are learned using the
    Baum-Welch algorithm.

    Parameters:
    ----------
    n_states (int):
        Number of hidden states in the model.
    n_features (int):
        Number of features in the emission data.
    n_components (int):
        Number of components in the Gaussian mixture model.
    transitions (Transitions):
        Type of transitions to use for the model.
            If 'ergodic'
                The transition probabilities are uniform.
            If 'left-to-right'
                The transition probabilities are left-to-right
                (i.e. each state can only transition to the next state).
    covariance_type (CovarianceType):
        Type of covariance parameters to use for the emission distributions.
    k_means (bool):
        Whether to use k-means clustering to initialize the emission means.
    alpha (float):
        Dirichlet concentration parameter for the prior over initial state
        probabilities and transition probabilities.
    min_covar (float):
        Floor value for covariance matrices.
    seed (Optional[int]):
        Random seed to use for reproducible results.
    """

    __slots__ = "n_components"

    def __init__(
        self,
        n_states: int,
        n_features: int,
        transitions: Transitions,
        covariance_type: CovarianceType,
        n_components: int = 1,
        k_means: bool = False,
        alpha: float = 1.0,
        min_covar: float = 1e-3,
        seed: Optional[int] = None,
    ):
        self.n_features = n_features
        self.n_components = n_components
        self.min_covar = min_covar
        self.covariance_type = covariance_type
        self.k_means = k_means
        super().__init__(n_states, transitions, alpha, seed)

    @property
    def pdf(self) -> MixtureSameFamily:
        return self._params.emission_pdf

    @property
    def dof(self):
        return (
            self.n_states**2
            - 1
            + self.n_states * self.n_components
            - self.n_states
            + self.pdf.component_distribution.loc.numel()
            + self.pdf.covariance_matrix.numel()
        )

    def sample_emission_pdf(self, X=None):
        weights = torch.log(
            constraints.sample_probs(self.alpha, (self.n_states, self.n_components))
        )
        if X is not None:
            means = (
                self._sample_kmeans(X)
                if self.k_means
                else X.mean(dim=0, keepdim=True)
                .expand(self.n_states, self.n_components, -1)
                .clone()
            )
            centered_data = X - X.mean(dim=0)
            covs = (
                (torch.mm(centered_data.T, centered_data) / (X.shape[0] - 1))
                .expand(self.n_states, self.n_components, -1, -1)
                .clone()
            )
        else:
            means = torch.zeros(
                size=(self.n_states, self.n_components, self.n_features),
                dtype=torch.float64,
            )

            covs = (
                self.min_covar
                + torch.eye(n=self.n_features, dtype=torch.float64)
                .expand(
                    (self.n_states, self.n_components, self.n_features, self.n_features)
                )
                .clone()
            )

        return MixtureSameFamily(
            Categorical(logits=weights),
            MultivariateNormal(loc=means, covariance_matrix=covs),
        )

    def _estimate_emission_pdf(self, X, posterior, theta):
        responsibilities = self._compute_log_responsibilities(X).exp()
        posterior_resp = responsibilities.permute(1, 2, 0) * posterior.T.unsqueeze(-2)

        new_weights = constraints.log_normalize(
            matrix=torch.log(posterior_resp.sum(-1)), 
            dim=1
        )
        new_means = self._compute_means(X, posterior_resp, theta)
        new_covs = self._compute_covs(X, posterior_resp, new_means, theta)

        return MixtureSameFamily(
            Categorical(logits=new_weights),
            MultivariateNormal(loc=new_means, covariance_matrix=new_covs),
        )

    def _sample_kmeans(self, X: torch.Tensor, seed: Optional[int] = None) -> torch.Tensor:
        """Sample cluster means from K Means algorithm"""
        k_means_alg = KMeans(
            n_clusters=self.n_states, random_state=seed, n_init="auto"
        ).fit(X)

        return torch.from_numpy(k_means_alg.cluster_centers_).reshape(
            self.n_states, self.n_components, self.n_features
        )

    def _compute_log_responsibilities(self, X: torch.Tensor) -> torch.Tensor:
        """Compute the responsibilities for each component."""
        X_expanded = X.unsqueeze(-2).unsqueeze(-2)
        component_log_probs = self.pdf.component_distribution.log_prob(X_expanded)
        log_responsibilities = constraints.log_normalize(
            matrix=self.pdf.mixture_distribution.logits.unsqueeze(0) + component_log_probs,
            dim=1
        )
        return log_responsibilities

    def _compute_means(
        self,
        X: torch.Tensor,
        posterior: torch.Tensor,
        theta: Optional[ContextualVariables] = None,
    ) -> torch.Tensor:
        """Compute the means for each hidden state"""
        if theta is not None:
            # TODO: matmul shapes are inconsistent
            raise NotImplementedError(
                "Contextualized emissions not implemented for GaussianMixtureHMM"
            )
        else:
            new_mean = posterior @ X
            new_mean /= posterior.sum(-1, keepdim=True)

        return new_mean

    def _compute_covs(
        self,
        X: torch.Tensor,
        posterior: torch.Tensor,
        new_means: torch.Tensor,
        theta: Optional[ContextualVariables] = None,
    ) -> torch.Tensor:
        """Compute the covariances for each component."""
        if theta is not None:
            # TODO: matmul shapes are inconsistent
            raise NotImplementedError(
                "Contextualized emissions not implemented for GaussianMixtureHMM"
            )
        else:
            posterior_adj = posterior.unsqueeze(-1)
            diff = X.unsqueeze(0).expand(
                self.n_states, self.n_components, -1, -1
            ) - new_means.unsqueeze(-2)
            new_covs = torch.transpose(diff * posterior_adj, -1, -2) @ diff
            new_covs /= posterior_adj.sum(-2, keepdim=True)

        new_covs += self.min_covar * torch.eye(self.n_features)
        return new_covs
