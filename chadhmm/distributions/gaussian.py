import torch
from sklearn.cluster import KMeans
from torch.distributions import MultivariateNormal

from chadhmm.distributions.base import BaseDistribution
from chadhmm.schemas import ContextualVariables


class GaussianDistribution(MultivariateNormal, BaseDistribution):
    """
    Gaussian distribution for HMM/HSMM models.

    Parameters:
    ----------
    mean (torch.Tensor):
        Mean of the Gaussian distribution of shape (n_states, n_features).
    covariance (torch.Tensor):
        Covariance of the Gaussian distribution of shape
        (n_states, n_features, n_features).
    min_covar (float):
        Minimum covariance value for numerical stability.
    """

    def __init__(
        self, mean: torch.Tensor, covariance: torch.Tensor, min_covar: float = 1e-3
    ):
        self.min_covar = min_covar
        super().__init__(mean, covariance)

    @property
    def dof(self) -> int:
        """Degrees of freedom: number of mean parameters + covariance parameters."""
        return self.loc.numel() + self.covariance_matrix.numel()

    @classmethod
    def sample_distribution(
        cls,
        n_components: int,
        n_features: int,
        X: torch.Tensor | None = None,
        k_means: bool = False,
        min_covar: float = 1e-3,
    ) -> "GaussianDistribution":
        """
        Sample a Gaussian distribution. If X is provided and k_means is True,
        use k-means to initialize the means, otherwise use the mean of X.
        """
        if X is not None:
            means = (
                cls.sample_kmeans(n_components, n_features, X)
                if k_means
                else X.mean(dim=0).expand(n_components, -1).clone()
            )
            centered_data = X - X.mean(dim=0)
            covs = (
                (torch.mm(centered_data.T, centered_data) / (X.shape[0] - 1))
                .expand(n_components, -1, -1)
                .clone()
            )
        else:
            means = torch.zeros(size=(n_components, n_features), dtype=torch.float64)

            covs = (
                min_covar
                + torch.eye(n=n_features, dtype=torch.float64)
                .expand((n_components, n_features, n_features))
                .clone()
            )

        return cls(means, covs, min_covar)

    @classmethod
    def from_posterior_distribution(
        cls,
        X: torch.Tensor,
        posterior: torch.Tensor,
        min_covar: float,
        theta: ContextualVariables | None = None,
    ) -> "GaussianDistribution":
        """
        Estimate emission probability distribution from data and posterior
        probabilities.

        Parameters:
        -----------
        X : torch.Tensor
            Observed data
        posterior : torch.Tensor
            Posterior probabilities
        min_covar : float
            Minimum covariance value for numerical stability
        theta : ContextualVariables | None
            Contextual variables (not yet supported)

        Returns:
        --------
        GaussianDistribution
            Estimated multivariate normal distribution
        """
        if theta is not None:
            raise NotImplementedError(
                "Contextualized emissions not implemented for GaussianHMM"
            )

        means = cls._compute_means_jit(X, posterior)
        covs = cls._compute_covs_jit(X, posterior, means, min_covar)
        return cls(means, covs, min_covar)

    @staticmethod
    def sample_kmeans(
        n_components: int, n_features: int, X: torch.Tensor, seed: int | None = None
    ) -> torch.Tensor:
        """Improved K-means initialization with multiple attempts."""
        best_inertia = float("inf")
        best_centers = None

        n_attempts = 3
        for _ in range(n_attempts):
            k_means = KMeans(
                n_clusters=n_components, random_state=seed, n_init="auto"
            ).fit(X.cpu().numpy())

            if k_means.inertia_ < best_inertia:
                best_inertia = k_means.inertia_
                best_centers = k_means.cluster_centers_

        centers_reshaped = (
            torch.from_numpy(best_centers)
            .to(X.device)
            .reshape(n_components, n_features)
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
    ) -> torch.Tensor:
        """JIT-compiled covariance computation."""
        posterior_adj = posterior.T.unsqueeze(-1)
        diff = X.unsqueeze(0) - new_means.unsqueeze(1)
        new_covs = torch.matmul(posterior_adj * diff.transpose(-1, -2), diff)
        new_covs /= posterior_adj.sum(-2, keepdim=True)

        # Add minimum covariance
        n_features = new_means.shape[-1]
        new_covs += min_covar * torch.eye(n_features, device=X.device, dtype=X.dtype)
        return new_covs
