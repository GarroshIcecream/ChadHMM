from typing import Optional,Sequence,List, Tuple, Literal
from abc import ABC, abstractmethod, abstractproperty

import torch
import torch.nn as nn
from torch.distributions import Categorical, Distribution
import numpy as np

from ..utils import (ContextualVariables, ConvergenceHandler, Observations, SeedGenerator,
log_normalize, sequence_generator, sample_probs, sample_A, is_valid_A, 
DECODERS, INFORM_CRITERIA) # type: ignore


class BaseHSMM(nn.Module,ABC):
    """
    Base Class for Hidden Semi-Markov Model (HSMM)
    ----------
    A Hidden Semi-Markov Model (HSMM) subclass that provides a foundation for building specific HMM models. HSMM is not assuming that the duration of each state is geometrically distributed, 
    but rather that it is distributed according to a general distribution. This duration is also reffered to as the sojourn time.
    """

    def __init__(self,
                 n_states:int,
                 max_duration:int,
                 alpha:float = 1.0,
                 seed:Optional[int] = None):
        
        super().__init__()
        self.n_states = n_states
        self.max_duration = max_duration
        self.alpha = alpha
        self._seed_gen = SeedGenerator(seed)
        self._params = self.sample_model_params(self.alpha)

    @property
    def seed(self):
        self._seed_gen.seed

    @property
    def A(self):
        return self._params.A.data
    
    @A.setter
    def A(self,logits:torch.Tensor):
        assert (o:=self.A.shape) == (f:=logits.shape), ValueError(f'Expected shape {o} but got {f}') 
        assert torch.allclose(logits.logsumexp(1),torch.ones(o)), ValueError(f'Probs do not sum to 1')
        assert is_valid_A(logits,self._A_type), ValueError(f'Transition Matrix is not satisfying the constraints given by its type {self._A_type}')
        self._params.A.data = logits 

    @property
    def pi(self):
        return self._params.pi.data
    
    @pi.setter
    def pi(self,logits:torch.Tensor):
        assert (o:=self.pi.shape) == (f:=logits.shape), ValueError(f'Expected shape {o} but got {f}')
        assert torch.allclose(logits.logsumexp(0),torch.ones(o)), ValueError(f'Probs do not sum to 1')
        self._params.pi.data = logits

    @property
    def D(self):
        return self._params.D.data
    
    @D.setter
    def D(self,logits:torch.Tensor):
        assert (o:=self.D.shape) == (f:=logits.shape), ValueError(f'Expected shape {o} but got {f}') 
        assert torch.allclose(logits.logsumexp(1),torch.ones(o)), ValueError(f'Probs do not sum to 1')
        self._params.D.data = logits

    @abstractproperty
    def pdf(self) -> Distribution:
        pass
    
    @abstractproperty
    def dof(self) -> int:
        """Returns the degrees of freedom of the model."""
        pass

    @abstractmethod
    def estimate_emission_params(self, 
                                 X:torch.Tensor, 
                                 posterior:torch.Tensor, 
                                 theta:Optional[ContextualVariables]) -> nn.ParameterDict:
        """Update the emission parameters where posterior is of shape (n_states,n_samples)"""
        pass

    @abstractmethod
    def sample_emission_params(self, X:Optional[torch.Tensor]=None) -> nn.ParameterDict:
        """Sample the emission parameters."""
        pass

    def sample_model_params(self, alpha:float = 1.0, X:Optional[torch.Tensor]=None) -> nn.ParameterDict:
        """Initialize the model parameters."""
        model_params = self.sample_emission_params(X)
        model_params.update(nn.ParameterDict({
            'pi':nn.Parameter(
                torch.log(sample_probs(alpha,(self.n_states,))),
                requires_grad=False
            ),
            'A':nn.Parameter(
                torch.log(sample_A(alpha,self.n_states,'semi')),
                requires_grad=False
            ),
            'D':nn.Parameter(
                torch.log(sample_probs(alpha,(self.n_states,self.max_duration))),
                requires_grad=False
            )
        }))

        return model_params
    
    def sample(self, size):
        return NotImplementedError('Sampling from HSMM yet to be implemented in next version')

    def map_emission(self, x:torch.Tensor) -> torch.Tensor:
        """Get emission probabilities for a given sequence of observations."""
        pdf_shape = self.pdf.batch_shape + self.pdf.event_shape
        b_size = torch.Size([torch.atleast_2d(x).size(0)]) + pdf_shape
        x_batched = x.unsqueeze(-len(pdf_shape)).expand(b_size)
        return self.pdf.log_prob(x_batched).squeeze()
    
    # TODO: torch.distributions.Distribution method _validate_sample does this but just on single observation
    def check_constraints(self, value:torch.Tensor) -> torch.Tensor:
        not_supported = value[torch.logical_not(self.pdf.support.check(value))].unique()
        events = self.pdf.event_shape
        event_dims = len(events)
        assert len(not_supported) == 0, ValueError(f'Values outside PDF support, got values: {not_supported.tolist()}')
        assert value.ndim == event_dims+1, ValueError(f'Expected number of dims differs from PDF constraints on event shape {events}')
        if event_dims > 0:
            assert value.shape[1:] == events, ValueError(f'PDF event shape differs, expected {events} but got {value.shape[1:]}')
        return value

    def to_observations(self, 
                        X:torch.Tensor, 
                        lengths:Optional[List[int]]=None) -> Observations:
        """Convert a sequence of observations to an Observations object."""
        X_valid = self.check_constraints(X).double()
        n_samples = X_valid.size(0)
        if lengths is not None:
            assert (s:=sum(lengths)) == n_samples, ValueError(f'Lenghts do not sum to total number of samples provided {s} != {n_samples}')
            seq_lengths = lengths
        else:
            seq_lengths = [n_samples]
        
        n_sequences = len(seq_lengths)
        seq_idx = torch.cumsum(torch.tensor([0]+seq_lengths, dtype=torch.int), dim=0)
        
        return Observations(
            data=X_valid,
            n_samples=n_samples,
            log_probs=self.map_emission(X_valid),
            start_idx=seq_idx[:-1],
            end_idx=seq_idx[1:]-1,
            lengths=seq_lengths,
            n_sequences=n_sequences
        )  
    
    def to_contextuals(self, 
                       theta:torch.Tensor, 
                       X:Observations) -> ContextualVariables:
        """Returns the parameters of the model."""
        if (n_dim:=theta.ndim) != 2:
            raise ValueError(f'Context must be 2-dimensional. Got {n_dim}.')
        elif theta.shape[1] not in (1, X.data.shape[0]):
            raise ValueError(f'Context must have shape (context_vars, 1) for time independent context or (context_vars,{X.data.shape[0]}) for time dependent. Got {theta.shape}.')
        else:
            n_context, n_observations = theta.shape
            time_dependent = n_observations == X.data.shape[0]
            adj_theta = torch.vstack((theta, torch.ones(size=(1,n_observations),
                                                        dtype=torch.float64)))
            if not time_dependent:
                adj_theta = adj_theta.expand(n_context+1, X.data.shape[0])

            context_matrix = torch.split(adj_theta,list(X.lengths),1)
            return ContextualVariables(n_context, context_matrix, time_dependent)

    def fit(self,
            X:torch.Tensor,
            tol:float=0.01,
            max_iter:int=15,
            n_init:int=1,
            post_conv_iter:int=1,
            ignore_conv:bool=False,
            sample_B_from_X:bool=False,
            verbose:bool=True,
            plot_conv:bool=False,
            lengths:Optional[List[int]]=None,
            theta:Optional[torch.Tensor]=None):
        """Fit the model to the given sequence using the EM algorithm."""
        if sample_B_from_X:
            self._params.update(self.sample_emission_params(X))
        X_valid = self.to_observations(X,lengths)
        valid_theta = self.to_contextuals(theta,X_valid) if theta is not None else None

        self.conv = ConvergenceHandler(tol=tol,
                                       max_iter=max_iter,
                                       n_init=n_init,
                                       post_conv_iter=post_conv_iter,
                                       verbose=verbose)

        for rank in range(n_init):
            if rank > 0:
                self._params.update(self.sample_model_params(self.alpha,X))
            
            self.conv.push_pull(self._compute_log_likelihood(X_valid).sum(),0,rank)
            for iter in range(1,self.conv.max_iter+1):
                # EM algorithm step
                self._params.update(self._estimate_model_params(X_valid,valid_theta))

                # remap emission probabilities after update of B
                X_valid.log_probs = self.map_emission(X_valid.data)
                
                curr_log_like = self._compute_log_likelihood(X_valid).sum()
                converged = self.conv.push_pull(curr_log_like,iter,rank)

                if converged and verbose and not ignore_conv:
                    break
        
        if plot_conv:
            self.conv.plot_convergence()

        return self
    
    def predict(self, 
                X:torch.Tensor, 
                lengths:Optional[List[int]] = None,
                algorithm:Literal['map','viterbi'] = 'viterbi') -> List[torch.Tensor]:
        """Predict the most likely sequence of hidden states. Returns log-likelihood and sequences"""
        if algorithm not in DECODERS:
            raise ValueError(f'Unknown decoder algorithm {algorithm}')
        
        decoder = {'viterbi': self._viterbi,
                   'map': self._map}[algorithm]
        
        X_valid = self.to_observations(X,lengths)
        decoded_path = decoder(X_valid)

        return decoded_path

    def score(self, 
              X:torch.Tensor,
              lengths:Optional[List[int]]=None,
              by_sample:bool=True) -> torch.Tensor:
        """Compute the joint log-likelihood"""
        log_likelihoods = self._compute_log_likelihood(self.to_observations(X,lengths))
        res = log_likelihoods if by_sample else log_likelihoods.sum(0,keepdim=True)
        return res

    def ic(self,
           X:torch.Tensor,
           lengths:Optional[List[int]] = None,
           criterion:Literal['AIC','BIC','HQC'] = 'AIC',
           by_sample:bool=True) -> torch.Tensor:
        """Calculates the information criteria for a given model."""
        if criterion not in INFORM_CRITERIA:
            raise NotImplementedError(f'{criterion} is not a valid information criterion. Valid criteria are: {INFORM_CRITERIA}')
        
        criterion_compute = {
            'AIC': lambda log_likelihood: -2.0 * log_likelihood + 2.0 * self.dof,
            'BIC': lambda log_likelihood: -2.0 * log_likelihood + self.dof * np.log(X.shape[0]),
            'HQC': lambda log_likelihood: -2.0 * log_likelihood + 2.0 * self.dof * np.log(np.log(X.shape[0]))
        }[criterion]
        
        log_likelihood = self.score(X,lengths,by_sample)
        log_likelihood.apply_(criterion_compute)
        return log_likelihood

    def _forward(self, X:Observations) -> torch.Tensor:
        """Forward pass of the forward-backward algorithm."""
        log_alpha = torch.zeros(size=(X.n_samples,self.n_states,self.max_duration),
                                dtype=torch.float64)
        
        # reshape into column vectors
        log_alpha[X.start_idx] = self._params.D + (self._params.pi + X.log_probs[X.start_idx]).reshape(-1,1)
        for t in range(X.n_samples):
            if t not in X.start_idx:
                trans_alpha_sum = torch.logsumexp(log_alpha[t-1,:,0].reshape(-1,1) + self._params.A, dim=0) + X.log_probs[t]

                log_alpha[t,:,-1] = trans_alpha_sum + self._params.D[:,-1]
                log_alpha[t,:,:-1] = torch.logaddexp(log_alpha[t-1,:,1:] + X.log_probs[t].reshape(-1,1),
                                                    trans_alpha_sum.reshape(-1,1) + self._params.D[:,:-1])
                
        return log_alpha

    def _backward(self, X:Observations) -> torch.Tensor:
        """Backward pass of the forward-backward algorithm."""
        beta_vec = []
        for seq_len in X.lengths: 
            log_beta = torch.zeros(size=(seq_len,self.n_states,self.max_duration),
                                   dtype=torch.float64)
            
            for t in reversed(range(seq_len-1)):
                beta_dur_sum = torch.logsumexp(log_beta[t+1] + self.params.D, dim=1)

                log_beta[t,:,0] = torch.logsumexp(self.params.A + X.log_probs[t+1] + beta_dur_sum, dim=1)
                log_beta[t,:,1:] = X.log_probs[t+1].reshape(-1,1) + log_beta[t+1,:,:-1]
            
            beta_vec.append(log_beta)

        return torch.cat(beta_vec)

    def _gamma(self, X:Observations, log_alpha:List[torch.Tensor], log_xi:List[torch.Tensor]) -> List[torch.Tensor]:
        """Compute the log-Gamma variable in Hidden Markov Model."""
        gamma_vec = []
        for (seq_len,_,_),alpha,xi in zip(sequence_generator(X),log_alpha,log_xi):
            log_gamma = torch.zeros(size=(seq_len,self.n_states), 
                                    dtype=torch.float64)

            real_xi = xi.exp()
            log_gamma[-1] = alpha[-1].logsumexp(1)
            for t in reversed(range(seq_len-1)):
                log_gamma[t] = torch.log(log_gamma[t+1].exp() + torch.sum(real_xi[t] - real_xi[t].transpose(-2,-1),dim=1))

            gamma_vec.append(log_normalize(log_gamma))

        return gamma_vec

    def _xi(self, X:Observations, log_alpha:torch.Tensor, log_beta:torch.Tensor) -> torch.Tensor:
        """Compute the log-Xi variable in Hidden Markov Model."""
        xi_vec = []
        for (seq_len,_,log_probs),alpha,beta in zip(sequence_generator(X),log_alpha,log_beta):
            log_xi = torch.zeros(size=(seq_len-1,self.n_states,self.n_states),
                                   dtype=torch.float64)
            
            for t in range(seq_len-1):
                beta_dur_sum = torch.logsumexp(beta[1:] + self._params.D.unsqueeze(0), dim=1)
                log_xi[t] = alpha[t,:,0].reshape(-1,1) + self.params.A + log_probs[t+1] + beta_dur_sum

            xi_vec.append(log_xi)

        return xi_vec
    
    def _eta(self, X:Observations, log_alpha:List[torch.Tensor], log_beta:List[torch.Tensor]) -> List[torch.Tensor]:
        """Compute the Eta variable in Hidden Markov Model."""
        eta_vec = []
        for (seq_len,_,log_probs),alpha,beta in zip(sequence_generator(X),log_alpha,log_beta):
            log_eta = torch.zeros(size=(seq_len-1,self.n_states,self.max_duration), 
                                  dtype=torch.float64)
            
            for t in range(seq_len-1):
                trains_alpha_sum = torch.logsumexp(alpha[t,:,0].reshape(-1,1) + self.params.A, dim=0)
                log_eta[t] = beta[t+1] + self.params.D + (log_probs[t+1] + trains_alpha_sum).reshape(-1, 1) 

            eta_vec.append(log_normalize(log_eta))
        
        return eta_vec

    def _compute_posteriors(self, X:Observations) -> Tuple[List[torch.Tensor],...]:
        """Execute the forward-backward algorithm and compute the log-Gamma, log-Xi and Log-Eta variables."""
        log_alpha = self._forward(X)
        log_beta = self._backward(X)
        log_xi = self._xi(X,log_alpha,log_beta)
        log_eta = self._eta(X,log_alpha,log_beta)
        log_gamma = self._gamma(X,log_alpha,log_xi)

        # Normalize the Xi after being used in gamma estimation
        log_xi = [log_normalize(xi,(1,2)) for xi in log_xi]

        return log_gamma, log_xi, log_eta
    
    def _accum_pi(self, log_gamma:List[torch.Tensor]) -> torch.Tensor:
        """Accumulate the statistics for the initial vector."""
        gamma_concat = torch.vstack([gamma[0] for gamma in log_gamma]).logsumexp(0)
        return log_normalize(gamma_concat,0)

    def _accum_A(self, log_xi:List[torch.Tensor]) -> torch.Tensor:
        """Accumulate the statistics for the transition matrix."""
        xi_concat = torch.vstack(log_xi).logsumexp(0)
        return log_normalize(xi_concat)

    def _accum_D(self, log_eta:List[torch.Tensor]) -> torch.Tensor:
        """Accumulate the statistics for the transition matrix."""
        eta_concat = torch.vstack(log_eta).logsumexp(0)
        return log_normalize(eta_concat)

    def _estimate_model_params(self, X:Observations, theta:Optional[ContextualVariables]) -> nn.ParameterDict:
        """Compute the updated parameters for the model."""
        log_gamma, log_xi, log_eta = self._compute_posteriors(X)

        new_params = self.estimate_emission_params(X.data,gamma,theta)
        new_params.update(nn.ParameterDict({
            'pi':nn.Parameter(
                self._accum_pi(log_gamma),
                requires_grad=False
            ),
            'A':nn.Parameter(
                self._accum_A(log_xi),
                requires_grad=False
            ),
            'D':nn.Parameter(
                self._accum_D(log_eta),
                requires_grad=False
            )
        }))
        
        return new_params

    def _viterbi(self, X:Observations) -> List[torch.Tensor]:
        """Viterbi algorithm for decoding the most likely sequence of hidden states."""
        raise NotImplementedError('Viterbi algorithm not yet implemented for HSMM')

    def _map(self, X:Observations) -> List[torch.Tensor]:
        """Compute the most likely (MAP) sequence of indiviual hidden states."""
        gamma,_ = self._compute_posteriors(X)
        map_paths = [gamma.argmax(1) for gamma in gamma]
        return map_paths

    def _compute_log_likelihood(self, X:Observations) -> torch.Tensor:
        """Compute the log-likelihood of the given sequence."""
        end_indices = torch.tensor(X.lengths,dtype=torch.int)-1
        scores = torch.index_select(self._forward(X),0,end_indices).logsumexp(1)
        return scores
