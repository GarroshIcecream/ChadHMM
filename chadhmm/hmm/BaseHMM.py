from abc import ABC, abstractmethod
from typing import Any, Optional, TypeVar, List, Union

import torch
import torch.nn as nn
from torch.distributions import Categorical, Distribution

from chadhmm.utils import constraints, SeedGenerator, ConvergenceHandler
from chadhmm.schemas import (
    Observations, 
    ContextualVariables, 
    InformCriteria, 
    DecodingAlgorithm,
    Transitions
)


T = TypeVar("T", bound="BaseHMM")


class BaseHMM(nn.Module, ABC):
    """
    Base Abstract Class for HMM
    ----------
    Base Class of Hidden Markov Models (HMM) class that provides a foundation
    for building specific HMM models.
    """

    __slots__ = "n_states", "params"

    def __init__(
        self,
        n_states: int,
        transitions: Transitions,
        alpha: float,
        seed: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.n_states = n_states
        self.alpha = alpha
        self._seed_gen = SeedGenerator(seed)
        self._A_type = transitions
        self._device = device if device is not None else torch.device("cpu")
        self._params = self.sample_model_params()

    @property
    def device(self) -> torch.device:
        """Get current device of model parameters."""
        return self._device

    @property
    def seed(self) -> int:
        return self._seed_gen.seed

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
        assert constraints.is_valid_A(logits, self._A_type), ValueError(
            f"Transition Matrix is not satisfying the constraints given by its type "
            f"{self._A_type}"
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
    @abstractmethod
    def pdf(self) -> Any:
        pass

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

    def save_model(self, file_path: str) -> None:
        """Save the model's state dictionary to a file."""
        torch.save(self.state_dict(), file_path)

    def load_model(self, file_path: str) -> None:
        """Load the model's state dictionary from a file."""
        self.load_state_dict(torch.load(file_path))
        self.eval()

    def to(self, device: Union[str, torch.device]) -> T:
        """Move model to specified device (CPU/GPU).

        Args:
            device: Target device ('cpu', 'cuda', or torch.device)
        """
        if isinstance(device, str):
            device = torch.device(device)

        self._device = device
        for key, param in self._params.items():
            if hasattr(param, "to"):
                self._params[key] = param.to(device)

        return super().to(device)

    def train_val_split(
        self, X: torch.Tensor, val_ratio: float = 0.2
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split data into training and validation sets."""
        n_samples = X.shape[0]
        indices = torch.randperm(n_samples)
        val_size = int(val_ratio * n_samples)
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]

        return X[train_indices], X[val_indices]

    def sample_model_params(self, X: Optional[torch.Tensor] = None) -> nn.ParameterDict:
        """Initialize the model parameters."""
        sampled_pi = torch.log(constraints.sample_probs(self.alpha, (self.n_states,)))
        sampled_A = torch.log(
            constraints.sample_A(self.alpha, self.n_states, self._A_type)
        )

        return nn.ParameterDict(
            {
                "pi": Categorical(logits=sampled_pi),
                "A": Categorical(logits=sampled_A),
                "emission_pdf": self.sample_emission_pdf(X),
            }
        )

    def map_emission(self, x: torch.Tensor) -> torch.Tensor:
        """Get emission probabilities for a given sequence of observations."""
        pdf_shape = self.pdf.batch_shape + self.pdf.event_shape
        b_size = torch.Size([x.shape[0]]) + pdf_shape
        x_batched = x.unsqueeze(-len(pdf_shape)).expand(b_size)
        return self.pdf.log_prob(x_batched).squeeze()

    def sample(self, size: int) -> torch.Tensor:
        """Sample from underlying Markov chain"""
        sampled_path = torch.zeros(size, dtype=torch.int)
        sampled_path[0] = self._params.pi.sample([1])

        sample_chain = self._params.A.sample(torch.Size([size]))
        for idx in range(size - 1):
            sampled_path[idx + 1] = sample_chain[idx, sampled_path[idx]]

        return sampled_path

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
        """Convert a sequence of observations to an Observations object."""
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
        nested_tensor_probs = [self.map_emission(tens) for tens in tensor_list]

        return Observations(tensor_list, nested_tensor_probs, seq_lengths)

    def to_contextuals(
        self, theta: torch.Tensor, X: Observations,
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
    ) -> T:
        """Estimate model parameters using the EM algorithm.

        Args:
            X: Observation sequences of shape (n_samples, *feature_shape)
            tol: Convergence threshold for EM. Training stops when the log-likelihood
                improvement is less than tol. Default: 0.01
            max_iter: Maximum number of EM iterations. Default: 15
            n_init: Number of initializations to perform. The best initialization
                (highest log-likelihood) is kept. Default: 1
            post_conv_iter: Number of iterations to perform after convergence.
                Default: 1
            ignore_conv: If True, run for max_iter iterations regardless of
                convergence. Default: False
            sample_B_from_X: If True, initialize emission parameters using the
                data. Default: False
            verbose: If True, print convergence information. Default: True
            plot_conv: If True, plot convergence curve after training. Default: False
            lengths: Lengths of sequences if X contains multiple sequences.
                Must sum to n_samples. Default: None (single sequence)
            theta: Optional contextual variables for conditional models.
                Shape must be (n_context_vars, n_samples) or (n_context_vars, 1).
                Default: None

        Returns:
            self: Returns the fitted model instance.

        Raises:
            ValueError: If lengths don't sum to n_samples or theta has invalid shape.
        """
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
                new_params = self._estimate_model_params(X_valid, valid_theta)
                self._params.update(new_params)
                X_valid.log_probs = [
                    self.map_emission(tens) for tens in X_valid.sequence
                ]

                curr_log_like = self._compute_log_likelihood(X_valid).sum()
                converged = self.conv.push_pull(curr_log_like, iter, rank)

                if converged and not ignore_conv:
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
        """Predict the most likely sequence of hidden states.

        Args:
            X: Observation sequences of shape (n_samples, *feature_shape)
            lengths: Lengths of sequences if X contains multiple sequences.
                Must sum to n_samples. Default: None (single sequence)
            algorithm: Decoding algorithm to use. Options:
                - 'viterbi': Find the most likely sequence of states (maximizes P(z|x))
                - 'map': Maximum a posteriori prediction for each state individually
                Default: 'viterbi'

        Returns:
            List[torch.Tensor]: List of predicted state sequences for each input
                sequence.
                Each tensor contains integer indices corresponding to the most
                likely states.

        Raises:
            ValueError: If lengths don't sum to n_samples or algorithm is unknown.
            ValueError: If X contains values outside the support of the emission
                distribution.
        """
        with torch.inference_mode():
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
        information_criteria = constraints.compute_information_criteria(
            X.shape[0], log_likelihood, self.dof, criterion
        )
        return information_criteria

    @staticmethod
    @torch.jit.script
    def _forward_jit(
        n_states: int, log_probs: torch.Tensor, A: torch.Tensor, pi: torch.Tensor
    ) -> torch.Tensor:
        """JIT-compiled forward algorithm implementation."""
        seq_len = log_probs.shape[0]
        log_alpha = torch.zeros((seq_len, n_states), dtype=torch.float64)

        log_alpha[0] = pi + log_probs[0]
        for t in range(seq_len - 1):
            log_alpha[t + 1] = log_probs[t + 1] + torch.logsumexp(
                A + log_alpha[t].unsqueeze(-1), dim=0
            )

        return log_alpha

    def _forward(self, X: Observations) -> List[torch.Tensor]:
        """Forward pass of the forward-backward algorithm."""
        alpha_vec = []
        for log_probs in X.log_probs:
            alpha_vec.append(
                self._forward_jit(self.n_states, log_probs, self.A, self.pi)
            )

        return alpha_vec

    @staticmethod
    @torch.jit.script
    def _backward_jit(
        n_states: int, log_probs: torch.Tensor, A: torch.Tensor
    ) -> torch.Tensor:
        """JIT-compiled backward algorithm implementation."""
        seq_len = log_probs.shape[0]
        log_beta = torch.zeros((seq_len, n_states), dtype=torch.float64)

        for t in range(seq_len - 2, -1, -1):
            log_beta[t] = torch.logsumexp(A + log_probs[t + 1] + log_beta[t + 1], dim=1)

        return log_beta

    def _backward(self, X: Observations) -> List[torch.Tensor]:
        """Backward pass of the forward-backward algorithm."""
        beta_vec: List[torch.Tensor] = []
        for log_probs in X.log_probs:
            beta_vec.append(self._backward_jit(self.n_states, log_probs, self.A))

        return beta_vec

    @staticmethod
    @torch.jit.script
    def _compute_posteriors_jit(
        log_alpha: torch.Tensor,
        log_beta: torch.Tensor,
        log_probs: torch.Tensor,
        A: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """JIT-compiled computation of posterior probabilities.

        Args:
            log_alpha: Forward variables of shape (seq_len, n_states)
            log_beta: Backward variables of shape (seq_len, n_states)
            log_probs: Log emission probabilities of shape (seq_len, n_states)
            A: Log transition matrix of shape (n_states, n_states)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - log_gamma: State posteriors of shape (seq_len, n_states)
                - log_xi: Transition posteriors of shape (seq_len-1, n_states, n_states)
        """
        log_gamma = log_alpha + log_beta
        log_gamma = log_gamma - torch.logsumexp(log_gamma, dim=1, keepdim=True)

        trans_alpha = A.unsqueeze(0) + log_alpha[:-1].unsqueeze(-1)
        probs_beta = (log_probs[1:] + log_beta[1:]).unsqueeze(1)
        log_xi = trans_alpha + probs_beta
        log_xi = log_xi - torch.logsumexp(log_xi, dim=(1, 2), keepdim=True)

        return log_gamma, log_xi

    def _compute_posteriors(
        self, X: Observations
    ) -> tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Execute the forward-backward algorithm and compute the log-Gamma and
        log-Xi variables.

        Args:
            X: Observations object containing sequences and their log probabilities

        Returns:
            Tuple[List[torch.Tensor], List[torch.Tensor]]:
                - List of log_gamma tensors for each sequence
                - List of log_xi tensors for each sequence
        """
        gamma_vec: List[torch.Tensor] = []
        xi_vec: List[torch.Tensor] = []

        # Compute forward and backward variables
        log_alpha_vec = self._forward(X)
        log_beta_vec = self._backward(X)

        # Compute posteriors for each sequence
        for log_alpha, log_beta, log_probs in zip(
            log_alpha_vec, log_beta_vec, X.log_probs, strict=False
        ):
            log_gamma, log_xi = self._compute_posteriors_jit(
                log_alpha, log_beta, log_probs, self.A
            )
            gamma_vec.append(log_gamma)
            xi_vec.append(log_xi)

        return gamma_vec, xi_vec

    def _estimate_model_params(
        self, X: Observations, theta: Optional[ContextualVariables] = None
    ) -> nn.ParameterDict:
        """Compute the updated parameters for the model."""
        log_gamma, log_xi = self._compute_posteriors(X)

        new_pi = constraints.log_normalize(
            matrix=torch.stack([tens[0] for tens in log_gamma], 1).logsumexp(1), 
            dim=0
        )
        new_A = constraints.log_normalize(
            matrix=torch.cat(log_xi).logsumexp(0), 
            dim=1
        )
        new_pdf = self._estimate_emission_pdf(
            X=torch.cat(X.sequence), posterior=torch.cat(log_gamma).exp(), theta=theta
        )

        return nn.ParameterDict(
            {
                "pi": Categorical(logits=new_pi),
                "A": Categorical(logits=new_A),
                "emission_pdf": new_pdf,
            }
        )

    @staticmethod
    @torch.jit.script
    def _viterbi_jit(
        n_states: int, log_probs: torch.Tensor, A: torch.Tensor, pi: torch.Tensor
    ) -> torch.Tensor:
        """JIT-compiled Viterbi algorithm implementation.

        Args:
            log_probs: Log probabilities of shape (seq_len, n_states)
            A: Transition matrix of shape (n_states, n_states)
            pi: Initial state distribution of shape (n_states,)

        Returns:
            torch.Tensor: Most likely state sequence
        """
        seq_len = log_probs.shape[0]

        viterbi_path = torch.empty(
            size=(seq_len,), dtype=torch.int64, device=log_probs.device
        )
        viterbi_prob = torch.empty(
            size=(seq_len, n_states), dtype=log_probs.dtype, device=log_probs.device
        )
        psi = torch.empty_like(viterbi_prob)

        viterbi_prob[0] = log_probs[0] + pi
        for t in range(1, seq_len):
            trans_seq = A + (viterbi_prob[t - 1] + log_probs[t]).reshape(-1, 1)
            viterbi_prob[t] = torch.max(trans_seq, dim=0).values
            psi[t] = torch.argmax(trans_seq, dim=0)

        viterbi_path[-1] = torch.argmax(viterbi_prob[-1])
        for t in range(seq_len - 2, -1, -1):
            viterbi_path[t] = psi[t + 1, viterbi_path[t + 1]]

        return viterbi_path

    def _viterbi(self, X: Observations) -> List[torch.Tensor]:
        """Viterbi algorithm for decoding the most likely sequence of hidden states."""
        viterbi_vec = []
        for log_probs in X.log_probs:
            viterbi_path = self._viterbi_jit(self.n_states, log_probs, self.A, self.pi)
            viterbi_vec.append(viterbi_path)

        return viterbi_vec

    def _map(self, X: Observations) -> List[torch.Tensor]:
        """Compute the most likely (MAP) sequence of indiviual hidden states."""
        gamma, _ = self._compute_posteriors(X)
        map_paths = [gamma.argmax(1) for gamma in gamma]
        return map_paths

    def _compute_log_likelihood(self, X: Observations) -> torch.Tensor:
        """Compute the log-likelihood of the given sequence."""
        log_alpha_vec = self._forward(X)
        concated_fwd = torch.stack([log_alpha[-1] for log_alpha in log_alpha_vec], 1)
        scores = concated_fwd.logsumexp(0)
        return scores
