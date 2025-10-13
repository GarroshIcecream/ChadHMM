import torch
import torch.nn as nn
from torch.distributions import Categorical

from chadhmm.schemas.common import Transitions


class InitialDistribution(nn.Module):
    """
    Initial state distribution π for HMM/HSMM.

    Represents the probability distribution over initial states.
    Uses nn.Parameter for learnable parameters.

    Args:
        logits: Log probabilities of shape (n_states,)
        probs: Probabilities of shape (n_states,) (alternative to logits)

    Properties:
        n_states: Number of states
        logits: Log probabilities (as nn.Parameter)
        probs: Probabilities (normalized)
    """

    def __init__(
        self,
        logits: torch.Tensor | None = None,
        probs: torch.Tensor | None = None,
    ):
        super().__init__()
        if logits is not None:
            self._logits = nn.Parameter(logits.clone(), requires_grad=False)
        elif probs is not None:
            self._logits = nn.Parameter(torch.log(probs.clone()), requires_grad=False)
        else:
            raise ValueError("Either logits or probs must be provided")
        self._validate()

    @property
    def logits(self) -> torch.Tensor:
        """Log probabilities."""
        return self._logits

    @property
    def probs(self) -> torch.Tensor:
        """Probabilities (normalized)."""
        return torch.softmax(self._logits, dim=-1)

    @property
    def n_states(self) -> int:
        """Number of states."""
        return self._logits.shape[0]

    @property
    def param_shape(self) -> torch.Size:
        """Shape of the parameters."""
        return self._logits.shape

    @property
    def device(self) -> torch.device:
        """Device of the parameters."""
        return self._logits.device

    def sample(self, sample_shape: torch.Size) -> torch.Tensor:
        """Sample from the distribution."""
        return Categorical(logits=self._logits).sample(sample_shape)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Compute log probability."""
        return Categorical(logits=self._logits).log_prob(value)

    def __repr__(self) -> str:
        return f"InitialDistribution(n_states={self.n_states})"

    @classmethod
    def sample_from_dirichlet(
        cls,
        prior: float,
        target_size: torch.Size,
    ) -> "InitialDistribution":
        """Sample from Dirichlet distribution."""
        alphas = torch.full(size=target_size, fill_value=prior)
        return cls(logits=torch.log(torch.distributions.Dirichlet(alphas).sample()))

    def _validate(self):
        """Validate the distribution parameters."""
        if self._logits.ndim != 1:
            raise ValueError(
                f"InitialDistribution must be 1-dimensional, "
                f"got shape {self._logits.shape}"
            )


class TransitionMatrix(nn.Module):
    """
    Transition matrix A for HMM/HSMM.

    Represents the probability distribution over next states given current state.
    Each row is a categorical distribution over next states.
    Uses nn.Parameter for learnable parameters.

    Args:
        logits: Log probabilities of shape (n_states, n_states)
        probs: Probabilities of shape (n_states, n_states) (alternative to logits)
        transition_type: Type of transition matrix (ergodic, semi, left-to-right)

    Properties:
        n_states: Number of states
        transition_type: Type of transition matrix
        logits: Log probabilities (as nn.Parameter)
        probs: Probabilities (normalized per row)
    """

    def __init__(
        self,
        transition_type: Transitions,
        logits: torch.Tensor | None = None,
        probs: torch.Tensor | None = None,
    ):
        super().__init__()
        if logits is not None:
            self._logits = nn.Parameter(logits.clone(), requires_grad=False)
        elif probs is not None:
            self._logits = nn.Parameter(torch.log(probs.clone()), requires_grad=False)
        else:
            raise ValueError("Either logits or probs must be provided")
        self._transition_type = transition_type
        self._validate()

    @property
    def logits(self) -> torch.Tensor:
        """Log probabilities."""
        return self._logits

    @property
    def probs(self) -> torch.Tensor:
        """Probabilities (normalized per row)."""
        return torch.softmax(self._logits, dim=-1)

    @property
    def transition_type(self) -> Transitions:
        """Transition type."""
        return self._transition_type

    @property
    def n_states(self) -> int:
        """Number of states."""
        return self._logits.shape[0]

    @property
    def param_shape(self) -> torch.Size:
        """Shape of the parameters."""
        return self._logits.shape

    @classmethod
    def sample_from_dirichlet(
        cls, prior: float, transition_type: Transitions, target_size: torch.Size
    ) -> "TransitionMatrix":
        """Sample from Dirichlet distribution."""
        alphas = torch.full(size=target_size, fill_value=prior)
        probs = torch.distributions.Dirichlet(alphas).sample()

        match transition_type:
            case Transitions.ERGODIC:
                pass
            case Transitions.SEMI:
                probs.fill_diagonal_(0)
                probs /= probs.sum(dim=-1, keepdim=True)
            case Transitions.LEFT_TO_RIGHT:
                probs = torch.triu(probs)
                probs /= probs.sum(dim=-1, keepdim=True)

        return cls(logits=torch.log(probs), transition_type=transition_type)

    def sample_next_state(self, current_state: torch.Tensor) -> torch.Tensor:
        """
        Sample next state given current state.

        Args:
            current_state: Current state index or indices

        Returns:
            Next state index or indices
        """
        if current_state.ndim == 0:
            return Categorical(logits=self._logits[current_state]).sample()
        else:
            return Categorical(logits=self._logits[current_state]).sample()

    def sample(self, sample_shape: torch.Size) -> torch.Tensor:
        """Sample from the distribution."""
        return Categorical(logits=self._logits).sample(sample_shape)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Compute log probability."""
        return Categorical(logits=self._logits).log_prob(value)

    def _validate(self):
        """Validate the distribution parameters."""
        if self._logits.ndim != 2:
            raise ValueError(
                f"TransitionMatrix must be 2-dimensional, "
                f"got shape {self._logits.shape}"
            )

        if self._logits.shape[0] != self._logits.shape[1]:
            raise ValueError(
                f"TransitionMatrix must be square, got shape {self._logits.shape}"
            )

        # Validate transition type constraints
        if self.transition_type == Transitions.SEMI:
            if not torch.allclose(
                torch.diag(self.probs),
                torch.zeros(
                    self.n_states, dtype=self.probs.dtype, device=self.probs.device
                ),
            ):
                raise ValueError(
                    "Semi-Markov transition matrix must have zero diagonal"
                )
        elif self.transition_type == Transitions.LEFT_TO_RIGHT:
            lower_tri = torch.tril(self.probs, diagonal=-1)
            if not torch.allclose(lower_tri, torch.zeros_like(lower_tri)):
                raise ValueError(
                    "Left-to-right transition matrix must be upper triangular"
                )

    def __repr__(self) -> str:
        return (
            f"TransitionMatrix(n_states={self.n_states}, "
            f"type={self.transition_type.value})"
        )


class DurationDistribution(nn.Module):
    """
    Duration distribution D for HSMM.

    Represents the probability distribution over state durations.
    Each row is a categorical distribution over possible durations for that state.
    Uses nn.Parameter for learnable parameters.

    Args:
        logits: Log probabilities of shape (n_states, max_duration)
        probs: Probabilities of shape (n_states, max_duration) (alternative to logits)

    Properties:
        n_states: Number of states
        max_duration: Maximum duration
        logits: Log probabilities (as nn.Parameter)
        probs: Probabilities (normalized per row)
    """

    def __init__(
        self,
        logits: torch.Tensor | None = None,
        probs: torch.Tensor | None = None,
    ):
        super().__init__()
        if logits is not None:
            self._logits = nn.Parameter(data=logits.clone(), requires_grad=False)
        elif probs is not None:
            self._logits = nn.Parameter(
                data=torch.log(probs.clone()), requires_grad=False
            )
        else:
            raise ValueError("Either logits or probs must be provided")

        self._validate()

    @property
    def logits(self) -> torch.Tensor:
        """Log probabilities."""
        return self._logits

    @property
    def probs(self) -> torch.Tensor:
        """Probabilities (normalized per row)."""
        return torch.softmax(self._logits, dim=-1)

    def _validate(self):
        """Validate the distribution parameters."""
        if self._logits.ndim != 2:
            raise ValueError(
                f"DurationDistribution must be 2-dimensional, "
                f"got shape {self._logits.shape}"
            )

    @property
    def n_states(self) -> int:
        """Number of states."""
        return self._logits.shape[0]

    @property
    def max_duration(self) -> int:
        """Maximum duration."""
        return self._logits.shape[1]

    def sample_duration(self, state: torch.Tensor) -> torch.Tensor:
        """
        Sample duration for a given state.

        Args:
            state: State index or indices

        Returns:
            Duration value or values
        """
        if state.ndim == 0:
            return Categorical(logits=self._logits[state]).sample()
        else:
            return Categorical(logits=self._logits[state]).sample()

    def mean_duration(self, state: int | None = None) -> torch.Tensor:
        """
        Compute mean duration for state(s).

        Args:
            state: Optional state index. If None, returns mean for all states.

        Returns:
            Mean duration(s). Shape (n_states,) if state is None, scalar otherwise.
        """
        # Duration values are 0-indexed (0 to max_duration-1)
        duration_values = torch.arange(
            self.max_duration, dtype=self.probs.dtype, device=self.probs.device
        )

        if state is None:
            # Return mean for all states
            return (self.probs * duration_values).sum(dim=1)
        else:
            # Return mean for specific state
            return (self.probs[state] * duration_values).sum()

    def sample(self, sample_shape: torch.Size) -> torch.Tensor:
        """Sample from the distribution."""
        return Categorical(logits=self._logits).sample(sample_shape)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Compute log probability."""
        return Categorical(logits=self._logits).log_prob(value)

    @classmethod
    def sample_from_dirichlet(
        cls,
        prior: float,
        target_size: torch.Size,
    ) -> "DurationDistribution":
        """Sample from Dirichlet distribution."""
        alphas = torch.full(size=target_size, fill_value=prior)
        return cls(logits=torch.log(torch.distributions.Dirichlet(alphas).sample()))

    def __repr__(self) -> str:
        return (
            f"DurationDistribution(n_states={self.n_states}, "
            f"max_duration={self.max_duration})"
        )
