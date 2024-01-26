from typing import Optional
import torch
import torch.nn as nn
from torch.distributions import Poisson, Independent

from .BaseHSMM import BaseHSMM # type: ignore
from ..utils import ContextualVariables


class PoissonHSMM(BaseHSMM):
    """
    Poisson Hidden Semi-Markov Model (HSMM)
    ----------
    Hidden Markov model with emissions. This model is a special case of the HSMM model with a geometric duration distribution.

    Parameters:
    ----------
    n_states (int):
        Number of hidden states in the model.
    n_features (int): 
        Number of emissions in the model.
    alpha (float):
        Dirichlet concentration parameter for the prior over initial distribution, transition amd emission probabilities.
    seed (int):
        Random seed for reproducibility.
    """

    def __init__(self,
                 n_states: int,
                 n_features: int,
                 max_duration:int,
                 alpha: float = 1.0,
                 seed: Optional[int] = None):
        
        self.n_features = n_features
        super().__init__(n_states,max_duration,alpha,seed)

    @property
    def dof(self):
        return self.n_states ** 2 + self.n_states * self.n_features - self.n_states - 1 + self.rates.numel()
    
    @property
    def rates(self) -> torch.Tensor:
        return self._params.rates.data
    
    @rates.setter
    def rates(self,lambdas:torch.Tensor):
        assert (o:=self.rates.shape) == (f:=lambdas.shape), ValueError(f'Expected shape {o} but got {f}') 
        assert torch.all(lambdas > 0.0), ValueError(f'Rates must be nonnegative.')
        self._params.rates.data = lambdas

    @property
    def pdf(self) -> Independent:
        return Independent(Poisson(self.rates),1)

    def sample_emission_params(self,X=None):
        if X is not None:
            rates = X.mean(dim=0).expand(self.n_states,-1).clone()
        else:
            rates = torch.ones(size=(self.n_states, self.n_features), 
                               dtype=torch.float64)

        return nn.ParameterDict({
            'rates':nn.Parameter(rates,requires_grad=False)
        })

    def estimate_emission_params(self,X,posterior,theta):
        return nn.ParameterDict({
            'rates':nn.Parameter(self._compute_rates(X,posterior,theta),requires_grad=False)
        })

    def _compute_rates(self,
                       X:torch.Tensor,
                       posterior:torch.Tensor,
                       theta:Optional[ContextualVariables]) -> torch.Tensor:
        """Compute the rates for each hidden state"""
        if theta is not None:
            # TODO: matmul shapes are inconsistent 
            raise NotImplementedError('Contextualized emissions not implemented for PoissonHMM')
        else:
            new_rates = posterior @ X
            new_rates /= posterior.sum(1,keepdim=True)

        return new_rates