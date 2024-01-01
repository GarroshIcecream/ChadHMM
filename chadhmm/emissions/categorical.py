import torch
import torch.nn as nn
from torch.distributions import Categorical # type: ignore
from typing import Optional, List

from .base_emiss import BaseEmission # type: ignore
from ..stochastic_matrix import StochasticTensor # type: ignore
from ..utils import ContextualVariables, log_normalize # type: ignore


class CategoricalEmissions(BaseEmission):
    """
    Categorical emission distribution for HMMs.

    Parameters:
    ----------
    n_dims (int):
        Number of hidden states in the model.
    n_features (int):
        Number of emissions in the model.
    alpha (float):
        Dirichlet concentration parameter for the prior over emission probabilities.
    """

    def __init__(self,
                 n_dims:int,
                 n_features:int,
                 alpha:float = 1.0):
        
        BaseEmission.__init__(self,n_dims,n_features)
        self.alpha = alpha
        self.params = self.sample_emission_params()

    @property
    def pdf(self) -> Categorical:
        return Categorical(logits=self.params.B)

    def sample_emission_params(self,X=None) -> nn.ParameterDict:
        if X is not None:
            emission_freqs = torch.bincount(X) / X.shape[0]
            emission_matrix = StochasticTensor(name='Emission', 
                                    logits=torch.log(emission_freqs.expand(self.n_dims,-1)))
        else:
            emission_matrix = StochasticTensor.from_dirichlet(name='Emission',
                                                   size=(self.n_dims,self.n_features),
                                                   prior=self.alpha)
        return nn.ParameterDict({
            'B': emission_matrix.logits
        })

    def map_emission(self,x):
        batch_shaped = x.repeat(self.n_dims,1).T
        return self.pdf.log_prob(batch_shaped)

    def update_emission_params(self,X,posterior,theta=None):
        self.params.update({'B':self._compute_emprobs(X,posterior,theta)})

    def _compute_emprobs(self,
                        X:List[torch.Tensor],
                        posterior:List[torch.Tensor],
                        theta:Optional[ContextualVariables]) -> torch.Tensor: 
        """Compute the emission probabilities for each hidden state."""
        emission_mat = torch.zeros(size=(self.n_dims, self.n_features),
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