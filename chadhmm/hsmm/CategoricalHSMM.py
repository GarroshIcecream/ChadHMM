from typing import Optional, List
import torch
import torch.nn as nn
from torch.distributions import Categorical

from .BaseHSMM import BaseHSMM # type: ignore
from ..utils import ContextualVariables, log_normalize, sample_logits


class CategoricalHSMM(BaseHSMM):
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
                 alpha:float = 1.0,
                 seed:Optional[int] = None):
        
        BaseHSMM.__init__(self,n_states,n_features,max_duration,alpha,seed)

    @property
    def dof(self):
        return self.n_states ** 2 + self.n_states * self.n_features - self.n_states - 1

    def map_emission(self,x):
        batch_shaped = x.repeat(self.n_states,1).T
        return self.pdf.log_prob(batch_shaped)
    
    @property
    def pdf(self) -> Categorical:
        return Categorical(logits=self.params.B)

    def estimate_emission_params(self,X,posterior,theta=None):
        return nn.ParameterDict({
            'B':nn.Parameter(self._compute_emprobs(X,posterior,theta),requires_grad=False)
        })

    def sample_emission_params(self,X=None):
        if X is not None:
            emission_freqs = torch.bincount(X) / X.shape[0]
            emission_matrix = torch.log(emission_freqs.expand(self.n_states,-1))
        else:
            emission_matrix = sample_logits(self.alpha,(self.n_states,self.n_features),False)
            
        return nn.ParameterDict({
            'B':nn.Parameter(emission_matrix,requires_grad=False)
        })

    def _compute_emprobs(self,
                        X:List[torch.Tensor],
                        posterior:List[torch.Tensor],
                        theta:Optional[ContextualVariables]=None) -> torch.Tensor: 
        """Compute the emission probabilities for each hidden state."""
        emission_mat = torch.zeros(size=(self.n_states,self.n_features),
                                   dtype=torch.float64)

        for seq,gamma_val in zip(X,posterior):
            if theta is not None:
                #TODO: Implement contextualized emissions
                raise NotImplementedError('Contextualized emissions not implemented for CategoricalEmissions')
            else:
                masks = seq.view(1,-1) == self.pdf.enumerate_support(expand=False)
                for i,mask in enumerate(masks):
                    masked_gamma = gamma_val[mask]
                    emission_mat[:,i] += masked_gamma.sum(dim=0)

        return log_normalize(emission_mat.log(),1)

