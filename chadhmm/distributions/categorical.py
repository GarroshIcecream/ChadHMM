import torch
from torch.distributions import Categorical

from chadhmm.schemas.common import Transitions


class InitialDistribution(Categorical):
    logits: torch.Tensor  # Type hint for inherited attribute
    probs: torch.Tensor  # Type hint for inherited attribute
    """
    Initial state distribution π for HMM/HSMM.

    Represents the probability distribution over initial states.
    Inherits from torch.distributions.Categorical.

    Args:
        logits: Log probabilities of shape (n_states,)
        probs: Probabilities of shape (n_states,) (alternative to logits)

    Properties:
        n_states: Number of states
        logits: Log probabilities
        probs: Probabilities (normalized)
    """

    def __init__(
        self,
        logits: torch.Tensor | None = None,
        probs: torch.Tensor | None = None,
    ):
        super().__init__(logits=logits, probs=probs)
        self._validate()

    def __repr__(self) -> str:
        return f"InitialDistribution(n_states={self.n_states})"

    @classmethod
    def sample_from_dirichlet(
        cls,
        prior: float,
        target_size: tuple[int, ...] | torch.Size,
        dtype: torch.dtype = torch.float64,
    ) -> "InitialDistribution":
        """Sample from Dirichlet distribution."""
        alphas = torch.full(size=target_size, fill_value=prior, dtype=dtype)
        return cls(logits=torch.log(torch.distributions.Dirichlet(alphas).sample()))

    @property
    def n_states(self) -> int:
        """Number of states."""
        return self.logits.shape[0]

    @property
    def shape(self) -> torch.Size:
        """Shape of the distribution."""
        return self.logits.shape

    def _validate(self):
        """Validate the distribution parameters."""
        if self.logits.ndim != 1:
            raise ValueError(
                f"InitialDistribution must be 1-dimensional, "
                f"got shape {self.logits.shape}"
            )


class TransitionMatrix(Categorical):
    logits: torch.Tensor  # Type hint for inherited attribute
    probs: torch.Tensor  # Type hint for inherited attribute
    """
    Transition matrix A for HMM/HSMM.

    Represents the probability distribution over next states given current state.
    Each row is a categorical distribution over next states.
    Inherits from torch.distributions.Categorical.

    Args:
        logits: Log probabilities of shape (n_states, n_states)
        probs: Probabilities of shape (n_states, n_states) (alternative to logits)
        transition_type: Type of transition matrix (ergodic, semi, left-to-right)

    Properties:
        n_states: Number of states
        transition_type: Type of transition matrix
        logits: Log probabilities
        probs: Probabilities (normalized per row)
    """

    def __init__(
        self,
        transition_type: Transitions,
        logits: torch.Tensor | None = None,
        probs: torch.Tensor | None = None,
    ):
        super().__init__(logits=logits, probs=probs)
        self.transition_type = transition_type
        self._validate()

    def _validate(self):
        """Validate the distribution parameters."""
        if self.logits.ndim != 2:
            raise ValueError(
                f"TransitionMatrix must be 2-dimensional, got shape {self.logits.shape}"
            )

        if self.logits.shape[0] != self.logits.shape[1]:
            raise ValueError(
                f"TransitionMatrix must be square, got shape {self.logits.shape}"
            )

        # Validate transition type constraints
        probs = self.probs
        if self.transition_type == Transitions.SEMI:
            # Semi-Markov: diagonal should be zero
            if not torch.allclose(
                torch.diag(probs),
                torch.zeros(self.n_states, dtype=probs.dtype, device=probs.device),
            ):
                raise ValueError(
                    "Semi-Markov transition matrix must have zero diagonal"
                )
        elif self.transition_type == Transitions.LEFT_TO_RIGHT:
            # Left-to-right: lower triangle should be zero
            # (allowing some numerical tolerance)
            lower_tri = torch.tril(probs, diagonal=-1)
            if not torch.allclose(lower_tri, torch.zeros_like(lower_tri)):
                raise ValueError(
                    "Left-to-right transition matrix must be upper triangular"
                )

    @property
    def n_states(self) -> int:
        """Number of states."""
        return self.logits.shape[0]

    @property
    def shape(self) -> torch.Size:
        """Shape of the distribution."""
        return self.logits.shape

    @classmethod
    def sample_from_dirichlet(
        cls,
        prior: float,
        transition_type: Transitions,
        target_size: tuple[int, ...] | torch.Size,
        dtype: torch.dtype = torch.float64,
    ) -> "TransitionMatrix":
        """Sample from Dirichlet distribution."""
        alphas = torch.full(size=target_size, fill_value=prior, dtype=dtype)
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
            return Categorical(logits=self.logits[current_state]).sample()
        else:
            return Categorical(logits=self.logits[current_state]).sample()

    def __repr__(self) -> str:
        return (
            f"TransitionMatrix(n_states={self.n_states}, "
            f"type={self.transition_type.value})"
        )


class DurationDistribution(Categorical):
    logits: torch.Tensor  # Type hint for inherited attribute
    probs: torch.Tensor  # Type hint for inherited attribute
    """
    Duration distribution D for HSMM.

    Represents the probability distribution over state durations.
    Each row is a categorical distribution over possible durations for that state.
    Inherits from torch.distributions.Categorical.

    Args:
        logits: Log probabilities of shape (n_states, max_duration)
        probs: Probabilities of shape (n_states, max_duration) (alternative to logits)

    Properties:
        n_states: Number of states
        max_duration: Maximum duration
        logits: Log probabilities
        probs: Probabilities (normalized per row)
    """

    def __init__(
        self,
        logits: torch.Tensor | None = None,
        probs: torch.Tensor | None = None,
    ):
        super().__init__(logits=logits, probs=probs)
        self._validate()

    def _validate(self):
        """Validate the distribution parameters."""
        if self.logits.ndim != 2:
            raise ValueError(
                f"DurationDistribution must be 2-dimensional, "
                f"got shape {self.logits.shape}"
            )

    @property
    def n_states(self) -> int:
        """Number of states."""
        return self.logits.shape[0]

    @property
    def max_duration(self) -> int:
        """Maximum duration."""
        return self.logits.shape[1]

    def sample_duration(self, state: torch.Tensor) -> torch.Tensor:
        """
        Sample duration for a given state.

        Args:
            state: State index or indices

        Returns:
            Duration value or values
        """
        if state.ndim == 0:
            return Categorical(logits=self.logits[state]).sample()
        else:
            return Categorical(logits=self.logits[state]).sample()

    def mean_duration(self, state: int | None = None) -> torch.Tensor:
        """
        Compute mean duration for a state or all states.

        Args:
            state: Optional state index. If None, returns mean for all states.

        Returns:
            Mean duration(s)
        """
        durations = torch.arange(
            self.max_duration, dtype=torch.float64, device=self.logits.device
        )

        if state is not None:
            return (self.probs[state] * durations).sum()
        else:
            return (self.probs * durations.unsqueeze(0)).sum(dim=1)

    def __repr__(self) -> str:
        return (
            f"DurationDistribution(n_states={self.n_states}, "
            f"max_duration={self.max_duration})"
        )
