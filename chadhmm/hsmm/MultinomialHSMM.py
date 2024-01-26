from typing import Optional
import torch
import torch.nn as nn
from torch.distributions import Multinomial

from .BaseHSMM import BaseHSMM # type: ignore
from ..utils import ContextualVariables, sample_probs


class MultinomialHSMM(BaseHSMM):
    """
    Categorical Hidden semi-Markov Model (HSMM)
    ----------
    Hidden semi-Markov model with categorical (discrete) emissions. This model is an extension of classical HMMs where the duration of each state is modeled by a geometric distribution.
    Duration in each state is modeled by a Categorical distribution with a fixed maximum duration.

    Parameters:
    ----------
    n_states (int):
        Number of hidden states in the model.
    n_features (int): 
        Number of emissions in the model.
    seed (int):
        Random seed for reproducibility.
    params_init (bool):
        Whether to initialize the model parameters prior to fitting.
    init_dist (SAMPLING_DISTRIBUTIONS):
        Distribution to use for initializing the model parameters.
    alpha (float):
        Dirichlet concentration parameter for the prior over initial state probabilities and transition probabilities.
    verbose (bool):
        Whether to print progress logs during fitting.
    """
    def __init__(self,
                 n_states:int,
                 n_features:int,
                 max_duration:int,
                 n_trials:int = 1,
                 alpha:float = 1.0,
                 seed:Optional[int] = None):
        
        self.n_features = n_features
        self.n_trials = n_trials
        super().__init__(n_states,max_duration,alpha,seed)

    @property
    def dof(self):
        return self.n_states ** 2 + self.n_states * self.n_features - self.n_states - 1

    @property
    def B(self) -> torch.Tensor:
        return self._params.B.data
    
    @B.setter
    def B(self, emission_matrix:torch.Tensor):
        assert (o:=self.A.shape) == (f:=emission_matrix.shape), ValueError(f'Expected shape {o} but got {f}') 
        assert torch.allclose(emission_matrix.logsumexp(1),torch.ones(o)), ValueError(f'Probs do not sum to 1')
        self._params.B.data = emission_matrix  
    
    @property
    def pdf(self) -> Multinomial:
        return Multinomial(total_count=self.n_trials,logits=self._params.B)

    def estimate_emission_params(self,X,posterior,theta=None):
        return nn.ParameterDict({
            'B':nn.Parameter(
                torch.log(self._compute_B(X,posterior,theta)),
                requires_grad=False
            )
        })

    def sample_emission_params(self,X=None):
        if X is not None:
            emission_freqs = torch.bincount(X) / X.shape[0]
            emission_matrix = torch.log(emission_freqs.expand(self.n_states,-1))
        else:
            emission_matrix = torch.log(sample_probs(self.alpha,(self.n_states,self.n_features)))

        return nn.ParameterDict({
            'B':nn.Parameter(
                emission_matrix,
                requires_grad=False
            )
        })

    def _compute_B(self,
                   X:torch.Tensor,
                   posterior:torch.Tensor,
                   theta:Optional[ContextualVariables]=None) -> torch.Tensor: 
        """Compute the emission probabilities for each hidden state."""
        if theta is not None:
            #TODO: Implement contextualized emissions
            raise NotImplementedError('Contextualized emissions not implemented for CategoricalEmissions')
        else:  
            new_B = posterior @ X
            new_B /= posterior.sum(1,keepdim=True)

        return new_B

