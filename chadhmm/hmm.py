import torch
import torch.nn as nn

from chadhmm.distributions import (
    BaseDistribution,
    InitialDistribution,
    TransitionMatrix,
)
from chadhmm.schemas import (
    ContextualVariables,
    DecodingAlgorithm,
    InformCriteria,
    Observations,
)
from chadhmm.utils import ConvergenceHandler, SeedGenerator, constraints


class HMM(nn.Module):
    """
    Base Class for HMM using composition pattern.

    This class implements the Hidden Markov Model framework and delegates
    emission distribution handling to a BaseDistribution instance.

    Parameters:
    ----------
    n_states : int
        Number of hidden states in the model.
    emission_pdf : BaseDistribution
        An instance of a distribution class (e.g., GaussianDistribution,
        MultinomialDistribution, PoissonDistribution).
    transition_matrix : TransitionMatrix
        Transition matrix of shape (n_states, n_states).
    initial_distribution : InitialDistribution
        Initial distribution of shape (n_states,).
    seed : int | None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_states: int,
        transition_matrix: TransitionMatrix,
        initial_distribution: InitialDistribution,
        emission_pdf: BaseDistribution,
        seed: int | None = None,
    ):
        super().__init__()
        self.n_states = n_states
        self._seed_gen = SeedGenerator(seed)
        self._params = nn.ParameterDict(
            {
                "pi": initial_distribution,
                "A": transition_matrix,
                "emission_pdf": emission_pdf,
            }
        )

    @property
    def seed(self) -> int:
        return self._seed_gen.seed

    @property
    def A(self) -> TransitionMatrix:
        return self._params.A

    @A.setter
    def A(self, value: TransitionMatrix):
        assert (o := self.A.param_shape) == (f := value.param_shape), ValueError(
            f"Expected shape {o} but got {f}"
        )
        self._params.A = value

    @property
    def pi(self) -> InitialDistribution:
        return self._params.pi

    @pi.setter
    def pi(self, value: InitialDistribution):
        assert (o := self.pi.param_shape) == (f := value.param_shape), ValueError(
            f"Expected shape {o} but got {f}"
        )
        self._params.pi = value

    @property
    def emission_pdf(self) -> BaseDistribution:
        return self._params.emission_pdf

    @emission_pdf.setter
    def emission_pdf(self, value: BaseDistribution):
        self._params.emission_pdf = value

    @property
    def params(self) -> nn.ParameterDict:
        return self._params

    @params.setter
    def params(self, params: nn.ParameterDict):
        self._params.update(params)

    @property
    def dof(self) -> int:
        """Returns the degrees of freedom of the model."""
        hmm_dof = self.n_states**2 - 1
        emission_dof = self.emission_pdf.dof
        return hmm_dof + emission_dof

    def save_model(self, file_path: str) -> None:
        """Save the model's state dictionary to a file."""
        torch.save(self.state_dict(), file_path)

    def load_model(self, file_path: str) -> None:
        """Load the model's state dictionary from a file."""
        self.load_state_dict(torch.load(file_path))
        self.eval()
    
    def sample(self, size: int) -> torch.Tensor:
        """Sample from underlying Markov chain"""
        sampled_path = torch.zeros(size, dtype=torch.int)
        sampled_path[0] = self._params.pi.sample([1])

        sample_chain = self._params.A.sample(torch.Size([size]))
        for idx in range(size - 1):
            sampled_path[idx + 1] = sample_chain[idx, sampled_path[idx]]

        return sampled_path

    def to_observations(
        self, X: torch.Tensor, lengths: list[int] | None = None
    ) -> Observations:
        """Convert a sequence of observations to an Observations object."""
        n_samples = X.size(0)
        if lengths is not None:
            assert (s := sum(lengths)) == n_samples, ValueError(
                f"Lenghts do not sum to total number of samples provided "
                f"{s} != {n_samples}"
            )
            seq_lengths = lengths
        else:
            seq_lengths = [n_samples]

        tensor_list = list(torch.split(X, seq_lengths))
        nested_tensor_probs = [self.emission_pdf.map_emission(tens) for tens in tensor_list]

        return Observations(tensor_list, nested_tensor_probs, seq_lengths)

    def to_contextuals(
        self,
        theta: torch.Tensor,
        X: Observations,
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
        lengths: list[int] | None = None,
        theta: torch.Tensor | None = None,
    ):
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
            self._params.emission_pdf = self.sample_emission_pdf(X)

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
            # if rank > 0:
            #     self._params.update(self.sample_model_params(observation_tensor=X))

            self.conv.push_pull(self._compute_log_likelihood(X_valid).sum(), 0, rank)
            for iter in range(1, self.conv.max_iter + 1):
                self._update_model_params(X_valid, valid_theta)
                X_valid.log_probs = [
                    self.emission_pdf.map_emission(tens) for tens in X_valid.sequence
                ]

                curr_log_like = self._compute_log_likelihood(X_valid).sum()
                converged = self.conv.push_pull(curr_log_like, iter, rank)

                if converged and not ignore_conv:
                    break

        if plot_conv:
            self.conv.plot_convergence()

    def predict(
        self,
        X: torch.Tensor,
        algorithm: DecodingAlgorithm,
        lengths: list[int] | None = None,
    ) -> list[torch.Tensor]:
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
        self, X: torch.Tensor, lengths: list[int] | None = None, by_sample: bool = True
    ) -> torch.Tensor:
        """Compute the joint log-likelihood"""
        with torch.inference_mode():
            X_valid = self.to_observations(X, lengths)
            log_likelihoods = self._compute_log_likelihood(X_valid)

        if not by_sample:
            log_likelihoods = log_likelihoods.sum(0, keepdim=True)

        return log_likelihoods

    def ic(
        self,
        X: torch.Tensor,
        criterion: InformCriteria,
        lengths: list[int] | None = None,
        by_sample: bool = True,
    ) -> torch.Tensor:
        """Calculates the information criteria for a given model."""
        with torch.inference_mode():
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

    def _forward(self, X: Observations) -> list[torch.Tensor]:
        """Forward pass of the forward-backward algorithm."""
        alpha_vec = []
        for log_probs in X.log_probs:
            alpha_vec.append(
                self._forward_jit(self.n_states, log_probs, self.A.logits, self.pi.logits)
            )

        return alpha_vec

    @staticmethod
    @torch.jit.script
    def _forward_jit_batched(
        n_states: int,
        log_probs: torch.Tensor,
        A: torch.Tensor,
        pi: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Batched JIT-compiled forward algorithm with masking for variable lengths.
        
        Args:
            n_states: Number of hidden states
            log_probs: Log emission probabilities of shape (batch_size, max_seq_len, n_states)
            A: Log transition matrix of shape (n_states, n_states)
            pi: Log initial distribution of shape (n_states,)
            lengths: Actual sequence lengths of shape (batch_size,)
            
        Returns:
            log_alpha: Forward variables of shape (batch_size, max_seq_len, n_states)
        """
        batch_size, max_seq_len, _ = log_probs.shape
        log_alpha = torch.zeros((batch_size, max_seq_len, n_states), dtype=torch.float64)
        
        log_alpha[:, 0] = pi + log_probs[:, 0]
        
        time_indices = torch.arange(max_seq_len, device=log_probs.device).unsqueeze(0)
        mask = time_indices < lengths.unsqueeze(1)
        
        for t in range(max_seq_len - 1):
            next_alpha = log_probs[:, t + 1] + torch.logsumexp(
                A.unsqueeze(0) + log_alpha[:, t].unsqueeze(-1), dim=1
            )
            
            active_mask = mask[:, t + 1].unsqueeze(-1)
            log_alpha[:, t + 1] = torch.where(
                active_mask, next_alpha, log_alpha[:, t + 1]
            )
        
        return log_alpha

    def _forward_batched(self, X: Observations) -> list[torch.Tensor]:
        """Batched forward pass using padding and masking."""
        if len(X.lengths) <= 1:
            return self._forward(X)
        
        max_len = max(X.lengths)
        padded_log_probs = []
        for log_probs in X.log_probs:
            pad_len = max_len - log_probs.shape[0]
            if pad_len > 0:
                padded = torch.nn.functional.pad(log_probs, (0, 0, 0, pad_len))
            else:
                padded = log_probs
            padded_log_probs.append(padded)
        
        batched_log_probs = torch.stack(padded_log_probs, dim=0)
        lengths_tensor = torch.tensor(X.lengths, dtype=torch.long, device=batched_log_probs.device)
        
        batched_alpha = self._forward_jit_batched(
            self.n_states, batched_log_probs, self.A.logits, self.pi.logits, lengths_tensor
        )
        
        alpha_vec = []
        for i, length in enumerate(X.lengths):
            alpha_vec.append(batched_alpha[i, :length, :])
        
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

    def _backward(self, X: Observations) -> list[torch.Tensor]:
        """Backward pass of the forward-backward algorithm."""
        beta_vec: list[torch.Tensor] = []
        for log_probs in X.log_probs:
            beta_vec.append(self._backward_jit(self.n_states, log_probs, self.A.logits))

        return beta_vec

    @staticmethod
    @torch.jit.script
    def _backward_jit_batched(
        n_states: int,
        log_probs: torch.Tensor,
        A: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Batched JIT-compiled backward algorithm with masking for variable lengths.
        
        Args:
            n_states: Number of hidden states
            log_probs: Log emission probabilities of shape (batch_size, max_seq_len, n_states)
            A: Log transition matrix of shape (n_states, n_states)
            lengths: Actual sequence lengths of shape (batch_size,)
            
        Returns:
            log_beta: Backward variables of shape (batch_size, max_seq_len, n_states)
        """
        batch_size, max_seq_len, _ = log_probs.shape
        log_beta = torch.zeros((batch_size, max_seq_len, n_states), dtype=torch.float64)
        time_indices = torch.arange(max_seq_len, device=log_probs.device).unsqueeze(0)
        mask = time_indices < lengths.unsqueeze(1)
        
        for t in range(max_seq_len - 2, -1, -1):
            next_beta = torch.logsumexp(
                A.unsqueeze(0) + log_probs[:, t + 1].unsqueeze(1) + log_beta[:, t + 1].unsqueeze(1),
                dim=2
            )
            
            active_mask = mask[:, t].unsqueeze(-1)
            log_beta[:, t] = torch.where(
                active_mask, next_beta, log_beta[:, t]
            )
        
        return log_beta

    def _backward_batched(self, X: Observations) -> list[torch.Tensor]:
        """Batched backward pass using padding and masking."""
        if len(X.log_probs) <= 1:
            return self._backward(X)
        
        max_len = max(X.lengths)
        padded_log_probs = []
        for log_probs in X.log_probs:
            pad_len = max_len - log_probs.shape[0]
            if pad_len > 0:
                padded = torch.nn.functional.pad(log_probs, (0, 0, 0, pad_len))
            else:
                padded = log_probs
            
            padded_log_probs.append(padded)
        
        batched_log_probs = torch.stack(padded_log_probs, dim=0)
        lengths_tensor = torch.tensor(X.lengths, dtype=torch.long, device=batched_log_probs.device)
        
        batched_beta = self._backward_jit_batched(
            self.n_states, batched_log_probs, self.A.logits, lengths_tensor
        )
        
        beta_vec = []
        for i, length in enumerate(X.lengths):
            beta_vec.append(batched_beta[i, :length, :])
        
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
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Execute the forward-backward algorithm and compute the log-Gamma and
        log-Xi variables.

        Args:
            X: Observations object containing sequences and their log probabilities
            
        Returns:
            Tuple[List[torch.Tensor], List[torch.Tensor]]:
                - List of log_gamma tensors for each sequence
                - List of log_xi tensors for each sequence
        """
        gamma_vec: list[torch.Tensor] = []
        xi_vec: list[torch.Tensor] = []

        log_alpha_vec = self._forward_batched(X)
        log_beta_vec = self._backward_batched(X)

        for log_alpha, log_beta, log_probs in zip(
            log_alpha_vec, log_beta_vec, X.log_probs, strict=False
        ):
            log_gamma, log_xi = self._compute_posteriors_jit(
                log_alpha, log_beta, log_probs, self.A.logits
            )
            gamma_vec.append(log_gamma)
            xi_vec.append(log_xi)

        return gamma_vec, xi_vec

    def _update_model_params(
        self, X: Observations, theta: ContextualVariables | None = None
    ) -> None:
        """Compute the updated parameters for the model."""
        log_gamma, log_xi = self._compute_posteriors(X)

        self.pi.logits = constraints.log_normalize(
            matrix=torch.stack([tens[0] for tens in log_gamma], 1).logsumexp(1), dim=0
        )
        self.A.logits = constraints.log_normalize(matrix=torch.cat(log_xi).logsumexp(0), dim=1)
        self.emission_pdf._update_posterior(
            X=torch.cat(X.sequence), posterior=torch.cat(log_gamma).exp(), theta=theta
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

    def _viterbi(self, X: Observations) -> list[torch.Tensor]:
        """Viterbi algorithm for decoding the most likely sequence of hidden states."""
        viterbi_vec = []
        for log_probs in X.log_probs:
            viterbi_path = self._viterbi_jit(self.n_states, log_probs, self.A.logits, self.pi.logits)
            viterbi_vec.append(viterbi_path)

        return viterbi_vec

    def _map(self, X: Observations) -> list[torch.Tensor]:
        """Compute the most likely (MAP) sequence of indiviual hidden states."""
        gamma_vec, _ = self._compute_posteriors(X)
        map_paths = [gamma.argmax(1) for gamma in gamma_vec]
        return map_paths

    def _compute_log_likelihood(self, X: Observations) -> torch.Tensor:
        """Compute the log-likelihood of the given sequence."""
        log_alpha_vec = self._forward(X)
        concated_fwd = torch.stack([log_alpha[-1] for log_alpha in log_alpha_vec], 1)
        scores = concated_fwd.logsumexp(0)
        return scores
