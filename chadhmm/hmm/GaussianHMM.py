import torch
from sklearn.cluster import KMeans
from torch.distributions import MultivariateNormal
from typing import Optional

from chadhmm.hmm.BaseHMM import BaseHMM
from chadhmm.utils import constraints
from chadhmm.schemas import Transitions, ContextualVariables, CovarianceType


class GaussianHMM(BaseHMM):
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

    def __init__(
        self,
        n_states: int,
        n_features: int,
        transitions: Transitions,
        covariance_type: CovarianceType,
        alpha: float = 1.0,
        k_means: bool = False,
        min_covar: float = 1e-3,
        seed: Optional[int] = None,
    ):
        self.n_features = n_features
        self.k_means = k_means
        self.min_covar = min_covar
        self.covariance_type = covariance_type
        super().__init__(n_states, transitions, alpha, seed)

    @property
    def pdf(self) -> MultivariateNormal:
        return self._params.emission_pdf

    @property
    def dof(self):
        return (
            self.n_states**2
            - 1
            + self.pdf.loc.numel()
            + self.pdf.covariance_matrix.numel()
        )

    def summary(self) -> dict:
        """
        Get a summary of the model's parameters and statistics.

        Returns:
            dict: Dictionary containing model information including:
                - Model type
                - Number of states
                - Number of features
                - Transition type
                - Covariance type
                - Degrees of freedom
                - Parameters per component
                - Memory usage estimate
                - Model configuration
        """
        # Calculate memory usage (approximate)
        param_bytes = (
            self.pdf.loc.numel() * 8
            + self.pdf.covariance_matrix.numel() * 8
            + self.initial_probs.numel() * 8
            + self.transition_probs.numel() * 8
        )
        memory_mb = param_bytes / (1024 * 1024)

        # Parameters per component
        params_per_component = {
            "means": self.n_features,
            "covariances": {
                constraints.CovarianceType.FULL: self.n_features * self.n_features,
                constraints.CovarianceType.DIAG: self.n_features,
                constraints.CovarianceType.SPHERICAL: 1,
            }[self.covariance_type],
        }

        return {
            "model_type": "Gaussian HMM",
            "n_states": self.n_states,
            "n_features": self.n_features,
            "transition_type": self._A_type,
            "covariance_type": self.covariance_type.name,
            "degrees_of_freedom": self.dof,
            "parameters_per_component": params_per_component,
            "memory_usage_mb": f"{memory_mb:.2f}",
            "configuration": {
                "k_means_init": self.k_means,
                "min_covar": self.min_covar,
                "alpha": self.alpha,
            },
            "current_parameters": {
                "mean_range": {
                    "min": float(self.pdf.loc.min()),
                    "max": float(self.pdf.loc.max()),
                },
                "covariance_range": {
                    "min": float(self.pdf.covariance_matrix.min()),
                    "max": float(self.pdf.covariance_matrix.max()),
                },
            },
        }

    def __str__(self) -> str:
        """
        Returns a string representation of the model.
        """
        summary_dict = self.summary()

        return (
            f"GaussianHMM(n_states={summary_dict['n_states']}, "
            f"n_features={summary_dict['n_features']}, "
            f"transitions={summary_dict['transition_type']}, "
            f"covariance_type={summary_dict['covariance_type']})"
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
        """Improved K-means initialization with multiple attempts."""
        best_inertia = float("inf")
        best_centers = None

        n_attempts = 3
        for _ in range(n_attempts):
            k_means = KMeans(
                n_clusters=self.n_states, random_state=seed, n_init="auto"
            ).fit(X.cpu().numpy())

            if k_means.inertia_ < best_inertia:
                best_inertia = k_means.inertia_
                best_centers = k_means.cluster_centers_

        centers_reshaped = (
            torch.from_numpy(best_centers)
            .to(X.device)
            .reshape(self.n_states, self.n_features)
        )
        return centers_reshaped

    @staticmethod
    @torch.jit.script
    def _compute_means_jit(X: torch.Tensor, posterior: torch.Tensor) -> torch.Tensor:
        """JIT-compiled mean computation."""
        new_mean = torch.matmul(posterior.T, X)
        new_mean /= posterior.sum(dim=0).unsqueeze(-1)
        return new_mean

    @staticmethod
    @torch.jit.script
    def _compute_covs_jit(
        X: torch.Tensor,
        posterior: torch.Tensor,
        new_means: torch.Tensor,
        min_covar: float,
        n_features: int,
    ) -> torch.Tensor:
        """JIT-compiled covariance computation."""
        posterior_adj = posterior.T.unsqueeze(-1)
        diff = X.unsqueeze(0) - new_means.unsqueeze(1)
        new_covs = torch.matmul(posterior_adj * diff.transpose(-1, -2), diff)
        new_covs /= posterior_adj.sum(-2, keepdim=True)

        # Add minimum covariance
        new_covs += min_covar * torch.eye(n_features, device=X.device, dtype=X.dtype)
        return new_covs

    def _compute_means(
        self,
        X: torch.Tensor,
        posterior: torch.Tensor,
        theta: Optional[ContextualVariables] = None,
    ) -> torch.Tensor:
        """Compute the means for each hidden state."""
        if theta is not None:
            raise NotImplementedError(
                "Contextualized emissions not implemented for GaussianHMM"
            )

        return self._compute_means_jit(X, posterior)

    def _compute_covs(
        self,
        X: torch.Tensor,
        posterior: torch.Tensor,
        new_means: torch.Tensor,
        theta: Optional[ContextualVariables] = None,
    ) -> torch.Tensor:
        """Compute the covariances for each component."""
        if theta is not None:
            raise NotImplementedError(
                "Contextualized emissions not implemented for GaussianHMM"
            )

        return self._compute_covs_jit(
            X, posterior, new_means, self.min_covar, self.n_features
        )
