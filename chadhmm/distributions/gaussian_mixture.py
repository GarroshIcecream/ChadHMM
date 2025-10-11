import torch
from sklearn.cluster import KMeans
from torch.distributions import Categorical, MixtureSameFamily, MultivariateNormal

from chadhmm.distributions.base import BaseDistribution
from chadhmm.schemas import ContextualVariables
from chadhmm.utils import constraints


class GaussianMixtureDistribution(MixtureSameFamily, BaseDistribution):
    """
    Gaussian Mixture distribution for HMM/HSMM models.

    This distribution represents a mixture of multivariate Gaussian distributions,
    where each state has multiple Gaussian components.

    Parameters:
    ----------
    weights (torch.Tensor):
        Mixture weights (logits) of shape (n_states, n_components).
    means (torch.Tensor):
        Means of the Gaussian components of shape (n_states, n_components, n_features).
    covariances (torch.Tensor):
        Covariances of the Gaussian components of shape 
        (n_states, n_components, n_features, n_features).
    min_covar (float):
        Minimum covariance value for numerical stability.
    """

    def __init__(
        self,
        weights: torch.Tensor,
        means: torch.Tensor,
        covariances: torch.Tensor,
        min_covar: float = 1e-3,
    ):
        self.min_covar = min_covar
        mixture_dist = Categorical(logits=weights)
        component_dist = MultivariateNormal(loc=means, covariance_matrix=covariances)
        super().__init__(mixture_dist, component_dist, validate_args=True)

    @property
    def n_states(self) -> int:
        return self.mixture_distribution.logits.shape[0]
    
    @property
    def n_components(self) -> int:
        return self.mixture_distribution.logits.shape[1]
    
    @property
    def n_features(self) -> int:
        return self.component_distribution.loc.shape[-1]

    @property
    def mixture_distribution(self) -> Categorical:
        return self.mixture_distribution
    
    @property
    def component_distribution(self) -> MultivariateNormal:
        return self.component_distribution

    @property
    def dof(self) -> int:
        """
        Degrees of freedom: number of mixture weights + mean parameters + covariance parameters.
        """
        n_states, n_components = self.mixture_distribution.logits.shape
        weights_dof = n_states * (n_components - 1)
        means_dof = self.component_distribution.loc.numel()
        covs_dof = self.component_distribution.covariance_matrix.numel()
        return weights_dof + means_dof + covs_dof

    @classmethod
    def sample_distribution(
        cls,
        n_components: int,
        n_features: int,
        n_mixture_components: int = 1,
        X: torch.Tensor | None = None,
        k_means: bool = False,
        min_covar: float = 1e-3,
        alpha: float = 1.0,
    ) -> "GaussianMixtureDistribution":
        """
        Sample/initialize a Gaussian Mixture distribution.

        Parameters:
        ----------
        n_components : int
            Number of states (components in the HMM sense)
        n_features : int
            Number of features in the emission data
        X : torch.Tensor | None
            Optional training data for initialization
        n_mixture_components : int
            Number of Gaussian components per state
        k_means : bool
            Whether to use k-means clustering to initialize the means
        min_covar : float
            Minimum covariance value for numerical stability
        alpha : float
            Dirichlet concentration parameter for mixture weights

        Returns:
        --------
        GaussianMixtureDistribution
            Initialized Gaussian mixture distribution
        """
        weights = torch.log(
            constraints.sample_probs(alpha, (n_components, n_mixture_components))
        )

        if X is not None:
            means = (
                cls._sample_kmeans(n_components, n_mixture_components, n_features, X)
                if k_means
                else X.mean(dim=0, keepdim=True)
                .expand(n_components, n_mixture_components, -1)
                .clone()
            )
            centered_data = X - X.mean(dim=0)
            covs = (
                (torch.mm(centered_data.T, centered_data) / (X.shape[0] - 1))
                .expand(n_components, n_mixture_components, -1, -1)
                .clone()
            )
        else:
            means = torch.zeros(
                size=(n_components, n_mixture_components, n_features),
                dtype=torch.float64,
            )

            covs = (
                min_covar
                + torch.eye(n=n_features, dtype=torch.float64)
                .expand((n_components, n_mixture_components, n_features, n_features))
                .clone()
            )

        return cls(weights, means, covs, min_covar)

    @classmethod
    def from_posterior_distribution(
        cls,
        X: torch.Tensor,
        posterior: torch.Tensor,
        min_covar: float,
        n_mixture_components: int,
        theta: ContextualVariables | None = None,
    ) -> "GaussianMixtureDistribution":
        """
        Estimate Gaussian Mixture distribution from data and posterior probabilities.

        Parameters:
        -----------
        X : torch.Tensor
            Observed data of shape (n_samples, n_features)
        posterior : torch.Tensor
            Posterior probabilities of shape (n_samples, n_states)
        min_covar : float
            Minimum covariance value for numerical stability
        n_mixture_components : int
            Number of Gaussian components per state
        theta : ContextualVariables | None
            Contextual variables (not yet supported)

        Returns:
        --------
        GaussianMixtureDistribution
            Estimated Gaussian mixture distribution
        """
        if theta is not None:
            raise NotImplementedError(
                "Contextualized emissions not implemented for GaussianMixtureDistribution"
            )

        # This will be called during EM, need a reference to current distribution
        # For now, this is a placeholder - actual implementation requires access to current weights
        raise NotImplementedError(
            "from_posterior_distribution should use _update_posterior instead"
        )

    @staticmethod
    def _sample_kmeans(
        n_states: int,
        n_mixture_components: int,
        n_features: int,
        X: torch.Tensor,
        seed: int | None = None,
    ) -> torch.Tensor:
        """Sample cluster means from K-Means algorithm."""
        k_means_alg = KMeans(
            n_clusters=n_states, random_state=seed, n_init="auto"
        ).fit(X.cpu().numpy())

        return torch.from_numpy(k_means_alg.cluster_centers_).reshape(
            n_states, n_mixture_components, n_features
        ).to(X.device)

    def _compute_log_responsibilities(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute the responsibilities for each mixture component.

        Parameters:
        -----------
        X : torch.Tensor
            Observed data of shape (n_samples, n_features)

        Returns:
        --------
        torch.Tensor
            Log responsibilities of shape (n_samples, n_states, n_components)
        """
        X_expanded = X.unsqueeze(-2).unsqueeze(-2)
        component_log_probs = self.component_distribution.log_prob(X_expanded)
        log_responsibilities = constraints.log_normalize(
            matrix=self.mixture_distribution.logits.unsqueeze(0) + component_log_probs,
            dim=1,
        )
        return log_responsibilities

    def _update_posterior(
        self,
        X: torch.Tensor,
        posterior: torch.Tensor,
        theta: ContextualVariables | None = None,
    ) -> None:
        """
        Update distribution parameters from posterior probabilities.

        Parameters:
        -----------
        X : torch.Tensor
            Observed data of shape (n_samples, n_features)
        posterior : torch.Tensor
            Posterior probabilities of shape (n_samples, n_states)
        theta : ContextualVariables | None
            Contextual variables (not yet supported)
        """
        if theta is not None:
            raise NotImplementedError(
                "Contextualized emissions not implemented for GaussianMixtureDistribution"
            )

        responsibilities = self._compute_log_responsibilities(X).exp()
        posterior_resp = responsibilities.permute(1, 2, 0) * posterior.T.unsqueeze(-2)
        new_weights = constraints.log_normalize(
            matrix=torch.log(posterior_resp.sum(-1)), dim=1
        )

        new_means = self._compute_means(X, posterior_resp)
        new_covs = self._compute_covs(X, posterior_resp, new_means)

        self.mixture_distribution.logits = new_weights
        self.component_distribution.loc = new_means
        self.component_distribution.covariance_matrix = new_covs

    def _compute_means(
        self,
        X: torch.Tensor,
        posterior: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the means for each mixture component.

        Parameters:
        -----------
        X : torch.Tensor
            Observed data of shape (n_samples, n_features)
        posterior_resp : torch.Tensor
            Combined posterior responsibilities of shape (n_states, n_components, n_samples)

        Returns:
        --------
        torch.Tensor
            Updated means of shape (n_states, n_components, n_features)
        """
        new_mean = posterior @ X
        new_mean /= posterior.sum(-1, keepdim=True)
        return new_mean

    def _compute_covs(
        self,
        X: torch.Tensor,
        posterior: torch.Tensor,
        new_means: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the covariances for each mixture component.

        Parameters:
        -----------
        X : torch.Tensor
            Observed data of shape (n_samples, n_features)
        posterior_resp : torch.Tensor
            Combined posterior responsibilities of shape (n_states, n_components, n_samples)
        new_means : torch.Tensor
            Updated means of shape (n_states, n_components, n_features)

        Returns:
        --------
        torch.Tensor
            Updated covariances of shape (n_states, n_components, n_features, n_features)
        """
        posterior_adj = posterior.unsqueeze(-1)
        diff = X.unsqueeze(0).expand(
            self.n_states, self.n_components, -1, -1
        ) - new_means.unsqueeze(-2)
        new_covs = torch.transpose(diff * posterior_adj, -1, -2) @ diff
        new_covs /= posterior_adj.sum(-2, keepdim=True)
        new_covs += self.min_covar * torch.eye(self.n_features)
        
        return new_covs

