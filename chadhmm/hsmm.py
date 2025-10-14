import torch
import torch.nn as nn

from chadhmm.distributions import (
    BaseDistribution,
    DurationDistribution,
    InitialDistribution,
    TransitionMatrix,
)
from chadhmm.schemas import (
    ContextualVariables,
    DecodingAlgorithm,
    InformCriteria,
    Observations,
)
from chadhmm.utils import ConvergenceHandler, constraints


class HSMM(nn.Module):
    """
    Base Class for HSMM using composition pattern.

    This class implements the Hidden Semi-Markov Model framework and delegates
    emission distribution handling to a BaseDistribution instance.

    A Hidden Semi-Markov Model (HSMM) extends HMM by not assuming that the duration
    of each state is geometrically distributed, but rather that it is distributed
    according to a general distribution. This duration is also referred to as the
    sojourn time.

    Parameters:
    ----------
    transition_matrix : TransitionMatrix
        Transition matrix of shape (n_states, n_states).
    initial_distribution : InitialDistribution
        Initial distribution of shape (n_states,).
    duration_distribution : DurationDistribution
        Duration distribution of shape (n_states, max_duration).
    emission_pdf : BaseDistribution
        An instance of a distribution class (e.g., GaussianDistribution,
        MultinomialDistribution, PoissonDistribution).
    device : torch.device | None
        Device to place the model on.
    dtype : torch.dtype | None
        Data type for the model parameters.
    """

    def __init__(
        self,
        transition_matrix: TransitionMatrix,
        initial_distribution: InitialDistribution,
        duration_distribution: DurationDistribution,
        emission_pdf: BaseDistribution,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.n_states = initial_distribution.n_states
        self.max_duration = duration_distribution.max_duration
        self._validate(
            transition_matrix, initial_distribution, duration_distribution, emission_pdf
        )
        self.add_module("pi", initial_distribution)
        self.add_module("A", transition_matrix)
        self.add_module("D", duration_distribution)
        self.add_module("emission_pdf", emission_pdf)
        self.to(device, dtype)

    @property
    def dof(self) -> int:
        """Returns the degrees of freedom of the model."""
        hsmm_dof = (
            self.n_states**2 - self.n_states + self.n_states * self.max_duration - 1
        )
        emission_dof = self.emission_pdf.dof
        return hsmm_dof + emission_dof

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
        nested_tensor_probs = [
            self.emission_pdf.map_emission(tens) for tens in tensor_list
        ]

        return Observations(tensor_list, nested_tensor_probs, seq_lengths)

    def to_contextuals(
        self,
        theta: torch.Tensor,
        X: Observations,
    ) -> ContextualVariables:
        """Returns the parameters of the model."""
        if (n_dim := theta.ndim) != 2:
            raise ValueError(f"Context must be 2-dimensional. Got {n_dim}.")

        if theta.shape[1] not in (1, sum(X.lengths)):
            raise ValueError(
                f"Context must have shape (context_vars, 1) for time independent "
                f"context or (context_vars,{sum(X.lengths)}) for time dependent. "
                f"Got {theta.shape}."
            )

        n_context, n_observations = theta.shape
        time_dependent = n_observations == sum(X.lengths)
        adj_theta = torch.vstack(
            (
                theta,
                torch.ones(
                    size=(1, n_observations), dtype=theta.dtype, device=theta.device
                ),
            )
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
        X_valid = self.to_observations(X, lengths)
        match algorithm:
            case DecodingAlgorithm.MAP:
                decoded_path = self._map(X_valid)
            case DecodingAlgorithm.VITERBI:
                decoded_path = self._viterbi(X_valid)

        return decoded_path

    def score(
        self, X: torch.Tensor, lengths: list[int] | None = None, by_sample: bool = True
    ) -> torch.Tensor:
        """Compute the joint log-likelihood"""
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
        log_likelihood = self.score(X, lengths, by_sample)
        information_criteria = constraints.compute_information_criteria(
            X.shape[0], log_likelihood, self.dof, criterion
        )

        return information_criteria

    def sample(
        self, n_samples: int = 100, random_state: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate synthetic data from the fitted HSMM model.

        This method samples a sequence of hidden states with explicit durations
        from the model, then generates observations from the emission distribution
        for each state.

        In HSMMs, states have explicit durations sampled from the duration distribution,
        and the model enforces no self-transitions (semi-Markov property).

        Args:
            n_samples: Number of samples to generate. Default: 100
            random_state: Random seed for reproducibility. If None, uses the model's
                internal seed generator. Default: None

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - samples: Generated observations of shape (n_samples, *feature_shape)
                - states: Hidden state sequence of shape (n_samples,)

        Example:
            >>> samples, states = model.sample(n_samples=100)
            >>> print(samples.shape)  # torch.Size([100, n_features])
            >>> print(states.shape)   # torch.Size([100])
        """
        device = self.A.logits.device
        generator = torch.Generator(device=device)
        if random_state is not None:
            generator.manual_seed(random_state)
        else:
            generator.manual_seed(torch.random.seed())

        # Convert log probabilities to probabilities
        pi_probs = self.pi.logits.exp()
        A_probs = self.A.logits.exp()
        D_probs = self.D.logits.exp()

        states = []
        t = 0

        # Sample initial state
        current_state = torch.multinomial(
            pi_probs, num_samples=1, generator=generator
        ).item()

        while t < n_samples:
            # Sample duration for current state (add 1 since duration is 1-indexed)
            duration = (
                torch.multinomial(
                    D_probs[current_state], num_samples=1, generator=generator
                ).item()
                + 1
            )

            # Ensure we don't exceed n_samples
            actual_duration = min(duration, n_samples - t)

            # Fill in the current state for its duration
            states.extend([current_state] * actual_duration)
            t += actual_duration

            # Sample next state (excluding current state for semi-Markov property)
            if t < n_samples:
                # Create transition probabilities excluding self-transition
                trans_probs = A_probs[current_state].clone()
                trans_probs[current_state] = 0.0
                # Renormalize
                trans_probs = trans_probs / trans_probs.sum()

                current_state = torch.multinomial(
                    trans_probs, num_samples=1, generator=generator
                ).item()

        # Convert states list to tensor
        states_tensor = torch.tensor(states, dtype=torch.long, device=device)

        # Sample observations from emission distribution
        # Generate n_samples observations, one for each time step
        # Sample shape (n_samples, n_components, *event_shape)
        all_samples = self.emission_pdf.sample(torch.Size([n_samples]))

        # Select the sample from the corresponding state at each time step
        # Create indices for batch dimension
        batch_idx = torch.arange(n_samples, device=device)
        samples = all_samples[batch_idx, states_tensor]

        return samples, states_tensor

    @staticmethod
    def _forward_uncompiled(
        n_states: int,
        max_duration: int,
        log_probs: torch.Tensor,
        A: torch.Tensor,
        pi: torch.Tensor,
        D: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Batched forward algorithm with masking for variable lengths (non-compiled).

        Args:
            n_states: Number of hidden states
            max_duration: Maximum duration for any state
            log_probs: Log emission probabilities of shape
                (batch_size, max_seq_len, n_states)
            A: Log transition matrix of shape (n_states, n_states)
            pi: Log initial state distribution of shape (n_states,)
            D: Log duration distributions of shape (n_states, max_duration)
            lengths: Actual sequence lengths of shape (batch_size,)

        Returns:
            log_alpha: Forward variables of shape
                (batch_size, max_seq_len, n_states, max_duration)
        """
        batch_size, max_seq_len, _ = log_probs.shape
        log_alpha = torch.zeros(
            (batch_size, max_seq_len, n_states, max_duration),
            dtype=log_probs.dtype,
            device=A.device,
        )

        log_alpha[:, 0] = (pi + log_probs[:, 0]).unsqueeze(-1) + D

        time_indices = torch.arange(max_seq_len, device=A.device).unsqueeze(0)
        mask = time_indices < lengths.unsqueeze(1)

        for t in range(max_seq_len - 1):
            alpha_trans_sum = (
                torch.logsumexp(
                    log_alpha[:, t, :, 0].unsqueeze(-1) + A.unsqueeze(0), dim=1
                )
                + log_probs[:, t + 1]
            )

            next_alpha_last = alpha_trans_sum + D[:, -1].unsqueeze(0)
            next_alpha_rest = torch.logaddexp(
                log_alpha[:, t, :, 1:] + log_probs[:, t + 1].unsqueeze(-1),
                D[:, :-1].unsqueeze(0) + alpha_trans_sum.unsqueeze(-1),
            )

            active_mask = mask[:, t + 1].unsqueeze(-1).unsqueeze(-1)
            log_alpha[:, t + 1, :, -1] = torch.where(
                active_mask.squeeze(-1), next_alpha_last, log_alpha[:, t + 1, :, -1]
            )
            log_alpha[:, t + 1, :, :-1] = torch.where(
                active_mask, next_alpha_rest, log_alpha[:, t + 1, :, :-1]
            )

        return log_alpha

    @staticmethod
    @torch.compile
    def _forward_compiled(
        n_states: int,
        max_duration: int,
        log_probs: torch.Tensor,
        A: torch.Tensor,
        pi: torch.Tensor,
        D: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Batched compiled forward algorithm with masking for variable lengths.

        Args:
            n_states: Number of hidden states
            max_duration: Maximum duration for any state
            log_probs: Log emission probabilities of shape
                (batch_size, max_seq_len, n_states)
            A: Log transition matrix of shape (n_states, n_states)
            pi: Log initial state distribution of shape (n_states,)
            D: Log duration distributions of shape (n_states, max_duration)
            lengths: Actual sequence lengths of shape (batch_size,)

        Returns:
            log_alpha: Forward variables of shape
                (batch_size, max_seq_len, n_states, max_duration)
        """
        batch_size, max_seq_len, _ = log_probs.shape
        log_alpha = torch.zeros(
            (batch_size, max_seq_len, n_states, max_duration),
            dtype=log_probs.dtype,
            device=A.device,
        )

        log_alpha[:, 0] = (pi + log_probs[:, 0]).unsqueeze(-1) + D

        time_indices = torch.arange(max_seq_len, device=A.device).unsqueeze(0)
        mask = time_indices < lengths.unsqueeze(1)

        for t in range(max_seq_len - 1):
            alpha_trans_sum = (
                torch.logsumexp(
                    log_alpha[:, t, :, 0].unsqueeze(-1) + A.unsqueeze(0), dim=1
                )
                + log_probs[:, t + 1]
            )

            next_alpha_last = alpha_trans_sum + D[:, -1].unsqueeze(0)
            next_alpha_rest = torch.logaddexp(
                log_alpha[:, t, :, 1:] + log_probs[:, t + 1].unsqueeze(-1),
                D[:, :-1].unsqueeze(0) + alpha_trans_sum.unsqueeze(-1),
            )

            active_mask = mask[:, t + 1].unsqueeze(-1).unsqueeze(-1)
            log_alpha[:, t + 1, :, -1] = torch.where(
                active_mask.squeeze(-1), next_alpha_last, log_alpha[:, t + 1, :, -1]
            )
            log_alpha[:, t + 1, :, :-1] = torch.where(
                active_mask, next_alpha_rest, log_alpha[:, t + 1, :, :-1]
            )

        return log_alpha

    def forward(self, X: Observations) -> list[torch.Tensor]:
        """Forward pass using batched operations with padding and masking.

        Uses compiled version during training for speed, non-compiled during inference
        to avoid compilation overhead.

        Args:
            X: Observations object containing sequences and their log probabilities

        Returns:
            List of forward variable tensors (alpha) for each sequence
        """
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
        lengths_tensor = torch.tensor(
            X.lengths, dtype=torch.long, device=batched_log_probs.device
        )

        # Use compiled version during training, uncompiled during inference
        if self.training:
            batched_alpha = self._forward_compiled(
                self.n_states,
                self.max_duration,
                batched_log_probs,
                self.A.logits,
                self.pi.logits,
                self.D.logits,
                lengths_tensor,
            )
        else:
            batched_alpha = self._forward_uncompiled(
                self.n_states,
                self.max_duration,
                batched_log_probs,
                self.A.logits,
                self.pi.logits,
                self.D.logits,
                lengths_tensor,
            )

        alpha_vec = []
        for i, length in enumerate(X.lengths):
            alpha_vec.append(batched_alpha[i, :length, :, :])

        return alpha_vec

    @staticmethod
    def _backward_uncompiled(
        n_states: int,
        max_duration: int,
        log_probs: torch.Tensor,
        A: torch.Tensor,
        D: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Batched backward algorithm with masking for variable lengths (non-compiled).

        Args:
            n_states: Number of hidden states
            max_duration: Maximum duration for any state
            log_probs: Log emission probabilities of shape
                (batch_size, max_seq_len, n_states)
            A: Log transition matrix of shape (n_states, n_states)
            D: Log duration distributions of shape (n_states, max_duration)
            lengths: Actual sequence lengths of shape (batch_size,)

        Returns:
            log_beta: Backward variables of shape
                (batch_size, max_seq_len, n_states, max_duration)
        """
        batch_size, max_seq_len, _ = log_probs.shape
        log_beta = torch.zeros(
            (batch_size, max_seq_len, n_states, max_duration),
            dtype=log_probs.dtype,
            device=A.device,
        )

        time_indices = torch.arange(max_seq_len, device=A.device).unsqueeze(0)
        mask = time_indices < lengths.unsqueeze(1)

        for t in range(max_seq_len - 2, -1, -1):
            dur_beta_sum = torch.logsumexp(
                log_beta[:, t + 1] + D.unsqueeze(0), dim=-1
            ).unsqueeze(1)
            next_beta_zero = torch.logsumexp(
                A.unsqueeze(0) + log_probs[:, t + 1].unsqueeze(1) + dur_beta_sum,
                dim=2,
            )
            next_beta_rest = log_beta[:, t + 1, :, :-1] + log_probs[:, t + 1].unsqueeze(
                -1
            )

            active_mask = mask[:, t].unsqueeze(-1)
            log_beta[:, t, :, 0] = torch.where(
                active_mask.squeeze(-1), next_beta_zero, log_beta[:, t, :, 0]
            )
            log_beta[:, t, :, 1:] = torch.where(
                active_mask.unsqueeze(-1), next_beta_rest, log_beta[:, t, :, 1:]
            )

        return log_beta

    @staticmethod
    @torch.compile
    def _backward_compiled(
        n_states: int,
        max_duration: int,
        log_probs: torch.Tensor,
        A: torch.Tensor,
        D: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Batched compiled backward algorithm with masking for variable lengths.

        Args:
            n_states: Number of hidden states
            max_duration: Maximum duration for any state
            log_probs: Log emission probabilities of shape
                (batch_size, max_seq_len, n_states)
            A: Log transition matrix of shape (n_states, n_states)
            D: Log duration distributions of shape (n_states, max_duration)
            lengths: Actual sequence lengths of shape (batch_size,)

        Returns:
            log_beta: Backward variables of shape
                (batch_size, max_seq_len, n_states, max_duration)
        """
        batch_size, max_seq_len, _ = log_probs.shape
        log_beta = torch.zeros(
            (batch_size, max_seq_len, n_states, max_duration),
            dtype=log_probs.dtype,
            device=A.device,
        )

        time_indices = torch.arange(max_seq_len, device=A.device).unsqueeze(0)
        mask = time_indices < lengths.unsqueeze(1)

        for t in range(max_seq_len - 2, -1, -1):
            dur_beta_sum = torch.logsumexp(
                log_beta[:, t + 1] + D.unsqueeze(0), dim=-1
            ).unsqueeze(1)
            next_beta_zero = torch.logsumexp(
                A.unsqueeze(0) + log_probs[:, t + 1].unsqueeze(1) + dur_beta_sum,
                dim=2,
            )
            next_beta_rest = log_beta[:, t + 1, :, :-1] + log_probs[:, t + 1].unsqueeze(
                -1
            )

            active_mask = mask[:, t].unsqueeze(-1)
            log_beta[:, t, :, 0] = torch.where(
                active_mask.squeeze(-1), next_beta_zero, log_beta[:, t, :, 0]
            )
            log_beta[:, t, :, 1:] = torch.where(
                active_mask.unsqueeze(-1), next_beta_rest, log_beta[:, t, :, 1:]
            )

        return log_beta

    def backward(self, X: Observations) -> list[torch.Tensor]:
        """Backward pass using batched operations with padding and masking.

        Uses compiled version during training for speed, non-compiled during inference
        to avoid compilation overhead.

        Args:
            X: Observations object containing sequences and their log probabilities

        Returns:
            List of backward variable tensors (beta) for each sequence
        """
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
        lengths_tensor = torch.tensor(
            X.lengths, dtype=torch.long, device=batched_log_probs.device
        )

        # Use compiled version during training, uncompiled during inference
        if self.training:
            batched_beta = self._backward_compiled(
                self.n_states,
                self.max_duration,
                batched_log_probs,
                self.A.logits,
                self.D.logits,
                lengths_tensor,
            )
        else:
            batched_beta = self._backward_uncompiled(
                self.n_states,
                self.max_duration,
                batched_log_probs,
                self.A.logits,
                self.D.logits,
                lengths_tensor,
            )

        beta_vec = []
        for i, length in enumerate(X.lengths):
            beta_vec.append(batched_beta[i, :length, :, :])

        return beta_vec

    @staticmethod
    @torch.compile
    def _compute_posteriors_compiled(
        n_states: int,
        log_alpha: torch.Tensor,
        log_beta: torch.Tensor,
        log_probs: torch.Tensor,
        A: torch.Tensor,
        D: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compiled computation of posterior probabilities for HSMM.

        Args:
            n_states: Number of hidden states
            log_alpha: Forward variables of shape (seq_len, n_states, max_duration)
            log_beta: Backward variables of shape (seq_len, n_states, max_duration)
            log_probs: Log emission probabilities of shape (seq_len, n_states)
            A: Log transition matrix of shape (n_states, n_states)
            D: Log duration distributions of shape (n_states, max_duration)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - log_gamma: State posteriors of shape (seq_len, n_states)
                - log_xi: Transition posteriors of shape
                    (seq_len-1, n_states, n_states)
                - log_eta: Duration posteriors of shape
                    (seq_len-1, n_states, max_duration)
        """
        seq_len = log_probs.shape[0]
        gamma = torch.zeros(
            size=(seq_len, n_states), dtype=log_probs.dtype, device=A.device
        )

        trans_alpha = log_alpha[:-1, :, 0].unsqueeze(-1) + A.unsqueeze(0)
        dur_beta = log_beta[1:] + D.unsqueeze(0)

        log_xi = (log_probs[1:] + torch.logsumexp(dur_beta, dim=2)).unsqueeze(
            1
        ) + trans_alpha
        normed_xi = log_xi - torch.logsumexp(log_xi, dim=(1, 2), keepdim=True)

        log_eta = (torch.logsumexp(trans_alpha, dim=1) + log_probs[1:]).unsqueeze(
            -1
        ) + dur_beta
        normed_eta = log_eta - torch.logsumexp(log_eta, dim=1, keepdim=True)

        xi_real = normed_xi.exp()
        gamma[-1] = (log_alpha[-1].logsumexp(dim=1) - log_alpha[-1].logsumexp(1)).exp()

        for t in range(seq_len - 2, -1, -1):
            gamma[t] = torch.clamp(
                gamma[t + 1]
                + torch.sum(xi_real[t] - xi_real[t].transpose(-2, -1), dim=0),
                min=0.0,
                max=1.0,
            )

        log_gamma = gamma.log() - gamma.log().logsumexp(dim=1, keepdim=True)

        return log_gamma, normed_xi, normed_eta

    def compute_posteriors(
        self, X: Observations
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """Execute the forward-backward algorithm and compute posterior probabilities

        Computes the state posteriors (gamma), transition posteriors (xi), and
        duration posteriors (eta) for each sequence using the forward-backward algorithm

        Args:
            X: Observations object containing sequences and their log probabilities

        Returns:
            Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
                - log_gamma: List of state posterior tensors for each sequence
                  Each tensor has shape (seq_len, n_states) in log space
                - log_xi: List of transition posterior tensors for each sequence
                  Each tensor has shape (seq_len-1, n_states, n_states) in log space
                - log_eta: List of duration posterior tensors for each sequence
                  Each tensor has shape (seq_len-1, n_states, max_duration) in log space
        """
        gamma_vec: list[torch.Tensor] = []
        xi_vec: list[torch.Tensor] = []
        eta_vec: list[torch.Tensor] = []

        log_alpha_vec = self.forward(X)
        log_beta_vec = self.backward(X)

        for log_alpha, log_beta, log_probs in zip(
            log_alpha_vec, log_beta_vec, X.log_probs, strict=False
        ):
            log_gamma, log_xi, log_eta = self._compute_posteriors_compiled(
                self.n_states,
                log_alpha,
                log_beta,
                log_probs,
                self.A.logits,
                self.D.logits,
            )
            gamma_vec.append(log_gamma)
            xi_vec.append(log_xi)
            eta_vec.append(log_eta)

        return gamma_vec, xi_vec, eta_vec

    def _update_model_params(
        self, X: Observations, theta: ContextualVariables | None = None
    ) -> None:
        """Compute the updated parameters for the model."""
        log_gamma, log_xi, log_eta = self.compute_posteriors(X)

        self.pi._logits.data = constraints.log_normalize(
            matrix=torch.stack([tens[0] for tens in log_gamma], 1).logsumexp(1), dim=0
        )
        self.A._logits.data = constraints.log_normalize(
            matrix=torch.cat(log_xi).logsumexp(0), dim=1
        )
        self.D._logits.data = constraints.log_normalize(
            matrix=torch.cat(log_eta).logsumexp(0), dim=1
        )
        self.emission_pdf._update_posterior(
            X=torch.cat(X.sequence), posterior=torch.cat(log_gamma).exp(), theta=theta
        )

    @staticmethod
    @torch.compile
    def _viterbi_compiled(
        n_states: int,
        max_duration: int,
        log_probs: torch.Tensor,
        A: torch.Tensor,
        pi: torch.Tensor,
        D: torch.Tensor,
    ) -> torch.Tensor:
        """Compiled Viterbi algorithm implementation for HSMM.

        Reference: Similar to the C++ implementation in hsmmlearn and the algorithm
        described in Yu (2010) "Hidden semi-Markov models"

        Args:
            n_states: Number of hidden states
            max_duration: Maximum duration for any state
            log_probs: Log emission probabilities of shape (seq_len, n_states)
            A: Log transition matrix of shape (n_states, n_states)
            pi: Log initial state distribution of shape (n_states,)
            D: Log duration distributions of shape (n_states, max_duration)

        Returns:
            torch.Tensor: Most likely state sequence of shape (seq_len,)
        """
        seq_len = log_probs.shape[0]

        # alpha[t, j] = max log probability of ending in state j at time t
        alpha = torch.full(
            size=(seq_len, n_states),
            fill_value=float("-inf"),
            dtype=log_probs.dtype,
            device=log_probs.device,
        )

        # maxU[t, j] = duration that achieved maximum at (t, j)
        maxU = torch.zeros(
            size=(seq_len, n_states),
            dtype=torch.int64,
            device=log_probs.device,
        )

        # maxI[t, j] = previous state that achieved maximum at (t, j)
        maxI = torch.zeros(
            size=(seq_len, n_states),
            dtype=torch.int64,
            device=log_probs.device,
        )

        # Forward pass
        for t in range(seq_len):
            for j in range(n_states):
                # Consider all possible durations u that could end at time t
                max_val = float("-inf")
                best_u = 0
                best_i = 0

                for u in range(1, min(t + 1, max_duration) + 1):
                    # Accumulate observations over duration
                    obs_sum = torch.sum(log_probs[t - u + 1 : t + 1, j])

                    if u <= t:
                        # Transition from previous state i to current state j
                        for i in range(n_states):
                            if i != j:  # Semi-Markov: no self-transitions
                                val = alpha[t - u, i] + A[i, j] + D[j, u - 1] + obs_sum
                                if val > max_val:
                                    max_val = val
                                    best_u = u
                                    best_i = i
                    else:
                        # Initial state (duration spans from start)
                        val = pi[j] + D[j, u - 1] + obs_sum
                        if val > max_val:
                            max_val = val
                            best_u = u
                            best_i = -1  # -1 indicates initial state

                alpha[t, j] = max_val
                maxU[t, j] = best_u
                maxI[t, j] = best_i

        # Termination: find best final state
        best_final_val = float("-inf")
        best_final_state = 0
        for j in range(n_states):
            if alpha[seq_len - 1, j] > best_final_val:
                best_final_val = alpha[seq_len - 1, j]
                best_final_state = j

        # Backtracking
        path = torch.zeros(seq_len, dtype=torch.int64, device=log_probs.device)
        t = seq_len - 1
        current_state = best_final_state

        while t >= 0:
            duration = int(maxU[t, current_state])
            prev_state_idx = int(maxI[t, current_state])

            # Fill in the current state for its duration
            start_idx = max(0, t - duration + 1)
            for i in range(start_idx, t + 1):
                path[i] = current_state

            # Move to previous state
            t = t - duration
            if prev_state_idx >= 0:
                current_state = prev_state_idx
            # else: we've reached the initial state

        return path

    def _viterbi(self, X: Observations) -> list[torch.Tensor]:
        """Viterbi algorithm for decoding the most likely sequence of hidden states."""
        viterbi_vec = []
        for log_probs in X.log_probs:
            viterbi_path = self._viterbi_compiled(
                self.n_states,
                self.max_duration,
                log_probs,
                self.A.logits,
                self.pi.logits,
                self.D.logits,
            )
            viterbi_vec.append(viterbi_path)

        return viterbi_vec

    def _map(self, X: Observations) -> list[torch.Tensor]:
        """Compute the most likely (MAP) sequence of indiviual hidden states."""
        gamma_vec, _, _ = self.compute_posteriors(X)
        map_paths = [gamma.argmax(1) for gamma in gamma_vec]
        return map_paths

    def _compute_log_likelihood(self, X: Observations) -> torch.Tensor:
        """Compute the log-likelihood of the given sequence."""
        log_alpha_vec = self.forward(X)
        concated_fwd = torch.stack([log_alpha[-1] for log_alpha in log_alpha_vec], 1)
        scores = concated_fwd.logsumexp(0)
        return scores

    def _validate(
        self,
        transition_matrix: TransitionMatrix,
        initial_distribution: InitialDistribution,
        duration_distribution: DurationDistribution,
        emission_pdf: BaseDistribution,
    ) -> None:
        """Validate the parameters of the model."""
        if transition_matrix.n_states != initial_distribution.n_states:
            raise ValueError(
                "Transition matrix and initial distribution must have the same "
                "number of states"
            )

        if duration_distribution.n_states != initial_distribution.n_states:
            raise ValueError(
                "Duration distribution and initial distribution must have the "
                "same number of states"
            )

        if transition_matrix.n_states != emission_pdf.n_components:
            raise ValueError(
                "Transition matrix and emission distribution must have the same "
                "number of components"
            )
