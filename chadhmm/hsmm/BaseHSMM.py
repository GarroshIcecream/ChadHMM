from abc import ABC, abstractmethod
from typing import Any, Optional, List, Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical, Distribution

from chadhmm.utils import ConvergenceHandler, SeedGenerator, constraints
from chadhmm.schemas import (
    Observations, 
    ContextualVariables, 
    DecodingAlgorithm,
    InformCriteria,
    Transitions
)


class BaseHSMM(nn.Module, ABC):
    """
    Base Class for Hidden Semi-Markov Model (HSMM)
    ----------
    A Hidden Semi-Markov Model (HSMM) subclass that provides a foundation
    for building specific HMM models. HSMM is not assuming that the duration
    of each state is geometrically distributed,
    but rather that it is distributed according to a general distribution.
    This duration is also reffered to as the sojourn time.
    """

    def __init__(
        self, n_states: int, max_duration: int, alpha: float, seed: Optional[int] = None
    ):
        super().__init__()
        self.n_states = n_states
        self.max_duration = max_duration
        self.alpha = alpha
        self._seed_gen = SeedGenerator(seed)
        self._params = self.sample_model_params()

    @property
    def seed(self):
        return self._seed_gen.seed

    @property
    def pdf(self) -> Any:
        return self._params.emission_pdf

    @property
    def A(self) -> torch.Tensor:
        return self._params.A.logits

    @A.setter
    def A(self, logits: torch.Tensor):
        assert (o := self.A.shape) == (f := logits.shape), ValueError(
            f"Expected shape {o} but got {f}"
        )
        assert torch.allclose(logits.logsumexp(1), torch.ones(o)), ValueError(
            "Probs do not sum to 1"
        )
        assert constraints.is_valid_A(logits, Transitions.SEMI), ValueError(
            "Semi-Transition Matrix is not satisfying the constraints"
        )
        self._params.A.logits = logits

    @property
    def pi(self) -> torch.Tensor:
        return self._params.pi.logits

    @pi.setter
    def pi(self, logits: torch.Tensor):
        assert (o := self.pi.shape) == (f := logits.shape), ValueError(
            f"Expected shape {o} but got {f}"
        )
        assert torch.allclose(logits.logsumexp(0), torch.ones(o)), ValueError(
            "Probs do not sum to 1"
        )
        self._params.pi.logits = logits

    @property
    def D(self) -> torch.Tensor:
        return self._params.D.logits

    @D.setter
    def D(self, logits: torch.Tensor):
        assert (o := self.D.shape) == (f := logits.shape), ValueError(
            f"Expected shape {o} but got {f}"
        )
        assert torch.allclose(logits.logsumexp(1), torch.ones(o)), ValueError(
            "Probs do not sum to 1"
        )
        self._params.D.logits = logits

    @property
    @abstractmethod
    def dof(self) -> int:
        """Returns the degrees of freedom of the model."""
        pass

    @abstractmethod
    def _estimate_emission_pdf(
        self,
        X: torch.Tensor,
        posterior: torch.Tensor,
        theta: Optional[ContextualVariables] = None,
    ) -> Distribution:
        """Update the emission parameters where posterior is of shape
        (n_states,n_samples)"""
        pass

    @abstractmethod
    def sample_emission_pdf(self, X: Optional[torch.Tensor] = None) -> Distribution:
        """Sample the emission parameters."""
        pass

    def sample_model_params(self, X: Optional[torch.Tensor] = None) -> nn.ParameterDict:
        """Initialize the model parameters."""
        sampled_pi = torch.log(constraints.sample_probs(self.alpha, (self.n_states,)))
        sampled_A = torch.log(
            constraints.sample_A(
                self.alpha, self.n_states, Transitions.SEMI
            )
        )
        sampled_D = torch.log(
            constraints.sample_probs(self.alpha, (self.n_states, self.max_duration))
        )

        return nn.ParameterDict(
            {
                "pi": Categorical(logits=sampled_pi),
                "A": Categorical(logits=sampled_A),
                "D": Categorical(logits=sampled_D),
                "emission_pdf": self.sample_emission_pdf(X),
            }
        )

    def sample(self, size: int) -> torch.Tensor:
        """Sample from underlying Markov chain"""
        sampled_path = torch.zeros(size, dtype=torch.int)
        sampled_path[0] = self._params.pi.sample([1])

        sample_chain = self._params.A.sample(torch.Size([size]))
        for idx in range(size - 1):
            sampled_path[idx + 1] = sample_chain[idx, sampled_path[idx]]

        return sampled_path

    def map_emission(self, x: torch.Tensor) -> torch.Tensor:
        """Get emission probabilities for a given sequence of observations."""
        pdf_shape = self.pdf.batch_shape + self.pdf.event_shape
        b_size = torch.Size([torch.atleast_2d(x).size(0)]) + pdf_shape
        x_batched = x.unsqueeze(-len(pdf_shape)).expand(b_size)
        return self.pdf.log_prob(x_batched).squeeze()

    def check_constraints(self, value: torch.Tensor) -> torch.Tensor:
        not_supported = value[torch.logical_not(self.pdf.support.check(value))].unique()
        events = self.pdf.event_shape
        event_dims = len(events)
        assert len(not_supported) == 0, ValueError(
            f"Values outside PDF support, got values: {not_supported.tolist()}"
        )
        assert value.ndim == event_dims + 1, ValueError(
            f"Expected number of dims differs from PDF constraints on event shape "
            f"{events}"
        )
        if event_dims > 0:
            assert value.shape[1:] == events, ValueError(
                f"PDF event shape differs, expected {events} but got {value.shape[1:]}"
            )
        return value

    def to_observations(
        self, X: torch.Tensor, lengths: Optional[List[int]] = None
    ) -> Observations:
        """Convert a sequence of observations to an utils.Observations object."""
        X_valid = self.check_constraints(X).double()
        n_samples = X_valid.size(0)
        if lengths is not None:
            assert (s := sum(lengths)) == n_samples, ValueError(
                f"Lenghts do not sum to total number of samples provided "
                f"{s} != {n_samples}"
            )
            seq_lengths = lengths
        else:
            seq_lengths = [n_samples]

        tensor_list = list(torch.split(X_valid, seq_lengths))
        tensor_probs = [self.map_emission(tens) for tens in tensor_list]

        return Observations(tensor_list, tensor_probs, seq_lengths)

    def to_contextuals(
        self, theta: torch.Tensor, X: Observations
    ) -> ContextualVariables:
        """Returns the parameters of the model."""
        if (n_dim := theta.ndim) != 2:
            raise ValueError(f"Context must be 2-dimensional. Got {n_dim}.")
        elif theta.shape[1] not in (1, sum(X.lengths)):
            raise ValueError(
                f"Context must have shape (context_vars, 1) for time independent "
                f"context or (context_vars,{sum(X.lengths)}) for time dependent. "
                f"Got {theta.shape}."
            )
        else:
            n_context, n_observations = theta.shape
            time_dependent = n_observations == sum(X.lengths)
            adj_theta = torch.vstack(
                (theta, torch.ones(size=(1, n_observations), dtype=torch.float64))
            )
            if not time_dependent:
                adj_theta = adj_theta.expand(n_context + 1, sum(X.lengths))

            context_matrix = torch.split(adj_theta, list(X.lengths), 1)
            return ContextualVariables(n_context, context_matrix, time_dependent)

    def fit(
        self,
        X: torch.Tensor,
        tol: float = 0.01,
        max_iter: int = 15,
        n_init: int = 1,
        post_conv_iter: int = 1,
        ignore_conv: bool = False,
        sample_B_from_X: bool = False,
        verbose: bool = True,
        plot_conv: bool = False,
        lengths: Optional[List[int]] = None,
        theta: Optional[torch.Tensor] = None,
    ):
        """Fit the model to the given sequence using the EM algorithm."""
        if sample_B_from_X:
            self._params.update({"emission_pdf": self.sample_emission_pdf(X)})

        X_valid = self.to_observations(X, lengths)
        valid_theta = self.to_contextuals(theta, X_valid) if theta is not None else None

        self.conv = ConvergenceHandler(
            tol=tol,
            max_iter=max_iter,
            n_init=n_init,
            post_conv_iter=post_conv_iter,
            verbose=verbose,
        )

        for rank in range(n_init):
            if rank > 0:
                self._params.update(self.sample_model_params(X))

            self.conv.push_pull(self._compute_log_likelihood(X_valid).sum(), 0, rank)
            for iter in range(1, self.conv.max_iter + 1):
                self._params.update(self._estimate_model_params(X_valid, valid_theta))

                X_valid.log_probs = [
                    self.map_emission(tens) for tens in X_valid.sequence
                ]

                curr_log_like = self._compute_log_likelihood(X_valid).sum()
                converged = self.conv.push_pull(curr_log_like, iter, rank)

                if converged and verbose and not ignore_conv:
                    break

        if plot_conv:
            self.conv.plot_convergence()

        return self

    def predict(
        self,
        X: torch.Tensor,
        algorithm: DecodingAlgorithm,
        lengths: Optional[List[int]] = None,
    ) -> List[torch.Tensor]:
        """Predict the most likely sequence of hidden states"""
        X_valid = self.to_observations(X, lengths)
        if algorithm == DecodingAlgorithm.MAP:
            decoded_path = self._map(X_valid)
        elif algorithm == DecodingAlgorithm.VITERBI:
            decoded_path = self._viterbi(X_valid)
        else:
            raise ValueError(f"Unknown decoder algorithm {algorithm}")

        return decoded_path

    def score(
        self, X: torch.Tensor, lengths: Optional[List[int]] = None, by_sample: bool = True
    ) -> torch.Tensor:
        """Compute the joint log-likelihood"""
        X_valid = self.to_observations(X, lengths)
        log_likelihoods = self._compute_log_likelihood(X_valid)

        if by_sample:
            return log_likelihoods
        else:
            return log_likelihoods.sum(0, keepdim=True)

    def ic(
        self,
        X: torch.Tensor,
        criterion: InformCriteria,
        lengths: Optional[List[int]] = None,
        by_sample: bool = True,
    ) -> torch.Tensor:
        """Calculates the information criteria for a given model."""
        log_likelihood = self.score(X, lengths, by_sample)
        return constraints.compute_information_criteria(
            X.shape[0], log_likelihood, self.dof, criterion
        )

    def _forward(self, X: Observations) -> List[torch.Tensor]:
        """Forward pass of the forward-backward algorithm."""
        alpha_vec: List[torch.Tensor] = []
        for seq_probs, seq_len in zip(X.log_probs, X.lengths, strict=False):
            log_alpha = torch.zeros(
                size=(seq_len, self.n_states, self.max_duration), dtype=torch.float64
            )

            log_alpha[0] = (self.pi + seq_probs[0]).unsqueeze(-1) + self.D
            for t in range(seq_len - 1):
                alpha_trans_sum = (
                    torch.logsumexp(log_alpha[t, :, 0].reshape(-1, 1) + self.A, 0)
                    + seq_probs[t + 1]
                )

                log_alpha[t + 1, :, -1] = alpha_trans_sum + self.D[:, -1]
                log_alpha[t + 1, :, :-1] = torch.logaddexp(
                    log_alpha[t, :, 1:] + seq_probs[t + 1].unsqueeze(-1),
                    self.D[:, :-1] + alpha_trans_sum.unsqueeze(-1),
                )

            alpha_vec.append(log_alpha)

        return alpha_vec

    def _backward(self, X: Observations) -> List[torch.Tensor]:
        """Backward pass of the forward-backward algorithm."""
        beta_vec: List[torch.Tensor] = []
        for seq_probs, seq_len in zip(X.log_probs, X.lengths, strict=False):
            log_beta = torch.zeros(
                size=(seq_len, self.n_states, self.max_duration), dtype=torch.float64
            )

            for t in reversed(range(seq_len - 1)):
                log_beta[t, :, 0] = torch.logsumexp(
                    self.A
                    + seq_probs[t + 1]
                    + torch.logsumexp(log_beta[t + 1] + self.D, 1),
                    1,
                )
                log_beta[t, :, 1:] = log_beta[t + 1, :, :-1] + seq_probs[
                    t + 1
                ].unsqueeze(-1)

            beta_vec.append(log_beta)

        return beta_vec

    def _compute_posteriors(
        self, X: Observations
    ) -> Tuple[List[torch.Tensor], ...]:
        """Execute the forward-backward algorithm and compute the log-Gamma,
        log-Xi and Log-Eta variables."""
        log_alpha = self._forward(X)
        log_beta = self._backward(X)

        xi_vec: List[torch.Tensor] = []
        eta_vec: List[torch.Tensor] = []
        gamma_vec: List[torch.Tensor] = []

        for seq_len, seq_probs, alpha, beta in zip(
            X.lengths, X.log_probs, log_alpha, log_beta, strict=False
        ):
            gamma = torch.zeros(size=(seq_len, self.n_states), dtype=torch.float64)

            trans_alpha = alpha[:-1, :, 0].unsqueeze(-1) + self.A.unsqueeze(0)
            dur_beta = beta[1:] + self.D.unsqueeze(0)

            log_xi = (seq_probs[1:] + torch.logsumexp(dur_beta, 2)).unsqueeze(
                1
            ) + trans_alpha
            normed_xi = constraints.log_normalize(log_xi, dim=(1, 2))
            log_eta = (torch.logsumexp(trans_alpha, 1) + seq_probs[1:]).unsqueeze(
                -1
            ) + dur_beta
            normed_eta = constraints.log_normalize(log_eta, dim=1)

            xi_real = normed_xi.exp()
            gamma[-1] = constraints.log_normalize(alpha[-1].logsumexp(1), dim=0).exp()
            for t in reversed(range(seq_len - 1)):
                gamma[t] = torch.clamp(
                    gamma[t + 1]
                    + torch.sum(xi_real[t] - xi_real[t].transpose(-2, -1), dim=0),
                    min=0.0,
                    max=1.0,
                )

            xi_vec.append(normed_xi)
            eta_vec.append(normed_eta)
            gamma_vec.append(constraints.log_normalize(gamma.log(), dim=1))

        return gamma_vec, xi_vec, eta_vec

    def _estimate_model_params(
        self, X: Observations, theta: Optional[ContextualVariables] = None
    ) -> nn.ParameterDict:
        """Compute the updated parameters for the model."""
        log_gamma, log_xi, log_eta = self._compute_posteriors(X)

        new_pi = constraints.log_normalize(
            matrix=torch.stack([tens[0] for tens in log_gamma], 1).logsumexp(1), 
            dim=0
        )
        new_A = constraints.log_normalize(
            matrix=torch.cat(log_xi).logsumexp(0), 
            dim=1
        )
        new_D = constraints.log_normalize(
            matrix=torch.cat(log_eta).logsumexp(0), 
            dim=1
        )
        new_pdf = self._estimate_emission_pdf(
            X=torch.cat(X.sequence), posterior=torch.cat(log_gamma).exp(), theta=theta
        )

        return nn.ParameterDict(
            {
                "pi": Categorical(logits=new_pi),
                "A": Categorical(logits=new_A),
                "D": Categorical(logits=new_D),
                "emission_pdf": new_pdf,
            }
        )

    # TODO: Implement Viterbi algorithm
    def _viterbi(self, X: Observations) -> List[torch.Tensor]:
        """Viterbi algorithm for decoding the most likely sequence of hidden states."""
        raise NotImplementedError("Viterbi algorithm not yet implemented for HSMM")

    def _map(self, X: Observations) -> List[torch.Tensor]:
        """Compute the most likely (MAP) sequence of indiviual hidden states."""
        gamma, _, _ = self._compute_posteriors(X)
        map_paths = [gamma.argmax(1) for gamma in gamma]
        return map_paths

    def _compute_log_likelihood(self, X: Observations) -> torch.Tensor:
        """Compute the log-likelihood of the given sequence."""
        log_alpha_vec = self._forward(X)
        concated_fwd = torch.stack([log_alpha[-1] for log_alpha in log_alpha_vec], 1)
        scores = concated_fwd.logsumexp(0)
        return scores
