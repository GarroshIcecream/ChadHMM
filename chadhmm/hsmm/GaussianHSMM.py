import torch
from sklearn.cluster import KMeans
from torch.distributions import MultivariateNormal
from typing import Optional

from chadhmm.hsmm.BaseHSMM import BaseHSMM
from chadhmm.schemas import ContextualVariables, CovarianceType


class GaussianHSMM(BaseHSMM):
    """
    Gaussian Hidden Semi-Markov Model (Gaussian HSMM)
    ----------
    This model assumes that the data follows a multivariate Gaussian distribution.
    The model parameters are learned using the Baum-Welch algorithm.

    Parameters:
    ----------
    n_states (int):
        Number of hidden states in the model.
    max_duration (int):
        Maximum duration of the states.
    n_features (int):
        Number of features in the emission data.
    n_components (int):
        Number of components in the Gaussian mixture model.
    alpha (float):
        Dirichlet concentration parameter for the prior over initial state
        probabilities and transition probabilities.
    covariance_type (COVAR_TYPES):
        Type of covariance parameters to use for the emission distributions.
    min_covar (float):
        Floor value for covariance matrices.
    seed (Optional[int]):
        Random seed to use for reproducible results.
    """

    def __init__(
        self,
        n_states: int,
        n_features: int,
        max_duration: int,
        covariance_type: CovarianceType,
        k_means: bool = False,
        alpha: float = 1.0,
        min_covar: float = 1e-3,
        seed: Optional[int] = None,
    ):
        self.n_features = n_features
        self.min_covar = min_covar
        self.k_means = k_means
        self.covariance_type = covariance_type
        BaseHSMM.__init__(self, n_states, max_duration, alpha, seed)

    @property
    def dof(self):
        return (
            self.n_states**2
            - 1
            + self.pdf.loc.numel()
            + self.pdf.covariance_matrix.numel()
        )

    def sample_emission_pdf(self, X=None):
        if X is not None:
            means = (
                self._sample_kmeans(X)
                if self.k_means
                else X.mean(dim=0).expand(self.n_states, -1).clone()
            )
            centered_data = X - X.mean(dim=0)
            covs = (
                (torch.mm(centered_data.T, centered_data) / (X.shape[0] - 1))
                .expand(self.n_states, -1, -1)
                .clone()
            )
        else:
            means = torch.zeros(
                size=(self.n_states, self.n_features), dtype=torch.float64
            )

            covs = (
                self.min_covar
                + torch.eye(n=self.n_features, dtype=torch.float64)
                .expand((self.n_states, self.n_features, self.n_features))
                .clone()
            )

        return MultivariateNormal(means, covs)

    def _estimate_emission_pdf(self, X, posterior, theta=None):
        new_means = self._compute_means(X, posterior, theta)
        new_covs = self._compute_covs(X, posterior, new_means, theta)
        return MultivariateNormal(new_means, new_covs)

    def _sample_kmeans(self, X: torch.Tensor, seed: Optional[int] = None) -> torch.Tensor:
        """Sample cluster means from K Means algorithm"""
        k_means_alg = KMeans(
            n_clusters=self.n_states, random_state=seed, n_init="auto"
        ).fit(X)

        return torch.from_numpy(k_means_alg.cluster_centers_).reshape(
            self.n_states, self.n_features
        )

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
                "Contextualized emissions not implemented for GaussianHMM"
            )
        else:
            new_mean = posterior.T @ X
            new_mean /= posterior.T.sum(-1, keepdim=True)

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
                "Contextualized emissions not implemented for GaussianHMM"
            )
        else:
            # TODO: Uses old mean value of normal distribution, correct?
            posterior_adj = posterior.T.unsqueeze(-1)
            diff = X.expand(self.n_states, -1, -1) - new_means.unsqueeze(-2)
            new_covs = torch.transpose(posterior_adj * diff, -1, -2) @ diff
            new_covs /= posterior_adj.sum(-2, keepdim=True)

        new_covs += self.min_covar * torch.eye(self.n_features)
        return new_covs
