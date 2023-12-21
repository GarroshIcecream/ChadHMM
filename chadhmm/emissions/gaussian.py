from typing import Optional, Literal, List
from sklearn.cluster import KMeans # type: ignore
import torch
from torch.distributions import MultivariateNormal

from .base_emiss import BaseEmission # type: ignore
from ..utils import ContextualVariables, validate_covars, fill_covars # type: ignore


class GaussianEmissions(BaseEmission):
    """
    Gaussian Distribution for HMM emissions.    
    
    Parameters:
    ----------
    n_dims (int):
        Number of mixtures in the model. This is equal to the number of hidden states in the HMM.
    n_components (int):
        Number of components in the mixture model.
    n_features (int):
        Number of features in the data.
    init_weights (bool):
        Whether to initialize the mixture weights prior to fitting.
    init_dist ():
        Distribution to use for initializing the mixture weights.
    alpha (float):
        Dirichlet concentration parameter for the prior over mixture weights.
    k_means (bool):
        Whether to initialize the mixture means using K-Means clustering.
    min_covar (float):
        Minimum covariance for the mixture components.
    covariance_type (COVAR_TYPES):
        Type of covariance matrix to use for the mixture components. One of 'spherical', 'tied', 'diag', 'full'.
    dims (Sequence[str]):
        Names of the mixtures in the model.
    seed (int):
        Random seed for reproducibility.
    device (torch.device):
        Device to use for computations.
    """

    COVAR_TYPES_HINT = Literal['spherical', 'tied', 'diag', 'full']

    def __init__(self, 
                 n_dims: int,
                 n_features: int,
                 k_means: bool = False,
                 covariance_type: COVAR_TYPES_HINT = 'full',
                 min_covar: float = 1e-3,
                 device: Optional[torch.device] = None):
        
        BaseEmission.__init__(self,n_dims,n_features,False,device)

        self.min_covar = min_covar
        self.covariance_type = covariance_type
        self.k_means = k_means
        self._means, self._covs = self.sample_emission_params()

    @property
    def means(self) -> torch.Tensor:
        return self._means
    
    @means.setter
    def means(self, means: torch.Tensor):
        target_size = (self.n_dims, self.n_features)
        assert means.shape == target_size, ValueError(f'Means shape must be {target_size}, got {means.shape}')
        self._means = means.to(self.device)

    @property
    def covs(self) -> torch.Tensor:
        return self._covs

    @covs.setter
    def covs(self, new_covars: torch.Tensor):
        """Setter function for the covariance matrices."""
        valid_covars = validate_covars(new_covars, self.covariance_type, self.n_dims, self.n_features)
        self._covs = fill_covars(valid_covars, self.covariance_type, self.n_dims, self.n_features).to(self.device)

    @property
    def pdf(self) -> MultivariateNormal:
        return MultivariateNormal(self.means,self.covs)
    
    def map_emission(self,x):
        b_size = (-1,self.n_dims,-1) if x.ndim == 2 else (self.n_dims,-1)
        x_batched = x.unsqueeze(-2).expand(b_size)
        return self.pdf.log_prob(x_batched)
    
    def sample_emission_params(self,X=None):
        if X is not None:
            means = self._sample_kmeans(X) if self.k_means else X.mean(dim=0).expand(self.n_dims,-1).clone()
            centered_data = X - X.mean(dim=0)
            covs = (torch.mm(centered_data.T, centered_data) / (X.shape[0] - 1)).expand(self.n_dims,-1,-1).clone()
        else:
            means = torch.zeros(size=(self.n_dims, self.n_features), 
                                dtype=torch.float64, 
                                device=self.device) 
            covs = self.min_covar + torch.eye(n=self.n_features, 
                                              dtype=torch.float64,
                                              device=self.device).expand((self.n_dims, self.n_features, self.n_features)).clone()

        return means, covs
    
    def update_emission_params(self,X,posterior,theta=None):
        self._means.copy_(self._compute_means(X,posterior,theta))
        self._covs.copy_(self._compute_covs(X,posterior,theta))
    
    def _sample_kmeans(self, X:torch.Tensor, seed:Optional[int]=None) -> torch.Tensor:
        """Sample cluster means from K Means algorithm"""
        k_means_alg = KMeans(n_clusters=self.n_dims, 
                             random_state=seed, 
                             n_init="auto").fit(X)
        return torch.from_numpy(k_means_alg.cluster_centers_).reshape(self.n_dims,self.n_features)

    def _compute_means(self,
                       X:List[torch.Tensor],
                       posterior:List[torch.Tensor],
                       theta:Optional[ContextualVariables]) -> torch.Tensor:
        """Compute the means for each hidden state"""
        new_mean = torch.zeros(size=(self.n_dims, self.n_features), 
                               dtype=torch.float64, 
                               device=self.device)
        
        denom = torch.zeros(size=(self.n_dims,1), 
                            dtype=torch.float64, 
                            device=self.device)
        
        for seq,gamma_val in zip(X,posterior):
            if theta is not None:
                # TODO: matmul shapes are inconsistent 
                raise NotImplementedError('Contextualized emissions not implemented for GaussianEmissions')
            else:
                new_mean += gamma_val.T @ seq
                denom += gamma_val.T.sum(dim=-1,keepdim=True)

        return new_mean / denom
    
    def _compute_covs(self, 
                      X:List[torch.Tensor],
                      posterior:List[torch.Tensor],
                      theta:Optional[ContextualVariables]) -> torch.Tensor:
        """Compute the covariances for each component."""
        new_covs = torch.zeros(size=(self.n_dims,self.n_features, self.n_features), 
                               dtype=torch.float64, 
                               device=self.device)
        
        denom = torch.zeros(size=(self.n_dims,1,1), 
                            dtype=torch.float64, 
                            device=self.device)

        for seq,gamma_val in zip(X,posterior):
            if theta is not None:
                # TODO: matmul shapes are inconsistent 
                raise NotImplementedError('Contextualized emissions not implemented for GaussianEmissions')
            else:
                gamma_expanded = gamma_val.T.unsqueeze(-1)
                diff = seq.expand(self.n_dims,-1,-1) - self.means.unsqueeze(1)
                new_covs += torch.transpose(gamma_expanded * diff,1,2) @ diff
                denom += torch.sum(gamma_expanded,dim=-2,keepdim=True)

        new_covs /= denom
        new_covs += self.min_covar * torch.eye(self.n_features, device=self.device)

        return new_covs