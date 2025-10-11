from chadhmm.distributions import PoissonDistribution
from chadhmm.hmm.BaseHMM import BaseHMM
from chadhmm.schemas import Transitions


class PoissonHMM(BaseHMM):
    """
    Poisson Hidden Markov Model using composition pattern.

    Hidden Markov model with Poisson emissions. Each state has independent
    Poisson distributions for each feature.

    Parameters:
    ----------
    n_states : int
        Number of hidden states in the model.
    n_features : int
        Number of independent Poisson-distributed features.
    alpha : float
        Dirichlet concentration parameter for the prior over initial
        distribution and transition probabilities.
    seed : int | None
        Random seed for reproducibility.

    Example:
    --------
    >>> model = PoissonHMM(n_states=3, n_features=2)
    >>> model.fit(X)
    >>> predictions = model.predict(X, algorithm='viterbi')
    """

    def __init__(
        self,
        n_states: int,
        n_features: int,
        transitions=None,  # For backward compatibility
        alpha: float = 1.0,
        seed: int | None = None,
    ):
        # Store parameters needed for distribution initialization
        self.n_features = n_features

        # Handle transitions parameter (backward compatibility)
        if transitions is None:
            transitions = Transitions.ERGODIC

        # Create initial emission distribution
        emission_dist = PoissonDistribution.sample_distribution(
            n_components=n_states, n_features=n_features, X=None
        )

        # Initialize BaseHMM with the emission distribution
        super().__init__(
            n_states=n_states,
            emission_dist=emission_dist,
            alpha=alpha,
            transitions=transitions,
            seed=seed,
        )

    def __str__(self) -> str:
        """Returns a string representation of the model."""
        return f"PoissonHMM(n_states={self.n_states}, n_features={self.n_features})"

    def __repr__(self) -> str:
        """Returns a detailed representation of the model."""
        return self.__str__()
